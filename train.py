"""
Train one (U, d, R) rung with the cyclic snapshot schedule and dump per-token
NLL of every snapshot on the val and calibration splits.

Usage:
    python train.py -U 100 -d 11 -R 8
    python train.py -U 100 -d 11 -R 8 -s 0,1,2,3
    python train.py -U 100 -d 11 -R 8 -elr 0.1 -wd 0.1
"""

import os
import csv
import json
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT models for data-constrained scaling")
parser.add_argument("-U", type=float, required=True, help="unique training tokens, in millions")
parser.add_argument("-d", type=int, required=True, help="depth on the scaling ladder")
parser.add_argument("-R", type=int, required=True, help="epochs: 1 or any even number")
parser.add_argument("-s", type=str, default="0", help="comma-separated seeds, trained one after another")
parser.add_argument("-elr", type=float, default=None, help="effective LR (LR multiplier x WD); default: our law")
parser.add_argument("-wd", type=float, default=None, help="peak weight decay; default: our law")
parser.add_argument("--device-batch-size", type=int, default=None)
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--runs-dir", type=str, default="runs")
parser.add_argument("--runs-csv", type=str, default="runs.csv")
args = parser.parse_args()

if not (args.R == 1 or (args.R >= 2 and args.R % 2 == 0)):
    parser.error(f"R must be 1 or even, got {args.R}")
if args.d < 6:
    parser.error("d must be at least 6 (the muP base width is 64 * 6)")

# =============================================================================
# Hyperparameters
# =============================================================================

SEQ_LEN = 2048
VOCAB_SIZE = 50257
PADDED_VOCAB = 50304
TOKENS_PER_STEP = 491_520   # 240 sequences, the same at every rung
MUP_BASE_WIDTH = 384        # d = 6

BASE_LR = dict(matrix=0.08, scalar=0.5, embedding=0.3, unembedding=0.004)
ADAM_BETAS = (0.8, 0.95)
MUON_MOMENTUM = 0.95

WARMUP_STEPS = 200
CYCLE_LR_FLOOR = 0.08
CYCLE_WD_FLOOR = 0.7


def num_params(d):
    return 49152 * d**3 + 3219456 * d


def hparam_law(U, N, R):
    elr = 3.953e4 * U**-1.0141 * N**0.2735 * R**0.1199
    wd = 107.5 * U**-0.8702 * N**0.3978 * R**0.6600
    return elr, wd


def default_device_batch_size(d):
    # fits an 80GB H100; must divide 240
    for max_d, bs in [(6, 60), (8, 48), (11, 40), (13, 30), (17, 24), (19, 20),
                      (21, 12), (24, 10), (27, 8), (29, 6), (32, 5)]:
        if d <= max_d:
            return bs
    return 4


U = int(args.U * 1e6)
N = num_params(args.d)
law_elr, law_wd = hparam_law(U, N, args.R)
elr = args.elr if args.elr is not None else float(f"{law_elr:.4g}")
wd = args.wd if args.wd is not None else float(f"{law_wd:.4g}")
lr_mult = elr / wd

device_batch_size = args.device_batch_size or default_device_batch_size(args.d)
assert (TOKENS_PER_STEP // SEQ_LEN) % device_batch_size == 0, "device batch size must divide 240"
grad_accum_steps = TOKENS_PER_STEP // (device_batch_size * SEQ_LEN)

# =============================================================================
# Flash Attention (FA3 on Hopper, SDPA elsewhere)
# =============================================================================

def _load_fa3():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        return None
    try:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception as e:
        print(f"Warning: failed to load FA3, falling back to SDPA: {e}")
        return None

_fa3 = _load_fa3()

def flash_attn_func(q, k, v, window_size):
    """q, k, v: (B, T, H, D)"""
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    T = q.size(1)
    i = torch.arange(T, device=q.device)
    mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] <= window_size[0])
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return y.transpose(1, 2)

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = SEQ_LEN
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    window_pattern: str = "SSSL"
    mup_base_n_embd: int = MUP_BASE_WIDTH

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)

@torch.compile(dynamic=False, fullgraph=True)
def softcap(logits, output_mult):
    return 15 * torch.tanh((logits * output_mult) / 15)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn_func(q, k, v, window_size=window_size)
        return self.c_proj(y.contiguous().view(B, T, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(PADDED_VOCAB, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, PADDED_VOCAB, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(PADDED_VOCAB, config.n_embd)
                                           for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, config.n_embd // config.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # muP readout: wider models shrink their logits so one set of LRs works for every width
        self.output_mult = config.mup_base_n_embd / config.n_embd

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.normal_(block.attn.c_proj.weight, mean=0.0, std=self.config.n_embd**-0.5)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.normal_(block.mlp.c_proj.weight, mean=0.0, std=self.config.n_embd**-0.5)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        self.cos, self.sin = self._precompute_rotary(self.rotary_seq_len, self.config.n_embd // self.config.n_head)
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32, device=device), inv_freq)
        return freqs.cos().bfloat16()[None, :, None, :], freqs.sin().bfloat16()[None, :, None, :]

    def _compute_window_sizes(self, config):
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[config.window_pattern[i % len(config.window_pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def num_params(self):
        # non-embedding params: transformer blocks + output projection
        return sum(p.numel() for p in self.transformer.h.parameters()) + self.lm_head.weight.numel()

    def setup_optimizer(self, lr_mult, weight_decay):
        adamw = lambda params, lr, betas=ADAM_BETAS, wd=weight_decay: dict(
            kind="adamw", params=params, lr=lr * lr_mult, betas=betas, eps=1e-10, weight_decay=wd)
        param_groups = [
            adamw(list(self.lm_head.parameters()), BASE_LR["unembedding"]),
            adamw(list(self.transformer.wte.parameters()), BASE_LR["embedding"]),
            adamw(list(self.value_embeds.parameters()), BASE_LR["embedding"]),
            adamw([self.resid_lambdas], BASE_LR["scalar"] * 0.01, wd=0.0),
            adamw([self.x0_lambdas], BASE_LR["scalar"], betas=(0.96, 0.95), wd=0.0),
        ]
        matrix_params = list(self.transformer.h.parameters())
        for shape in sorted({p.shape for p in matrix_params}):
            param_groups.append(dict(kind="muon", params=[p for p in matrix_params if p.shape == shape],
                                     lr=BASE_LR["matrix"] * lr_mult, momentum=MUON_MOMENTUM,
                                     ns_steps=5, beta2=0.95, weight_decay=weight_decay))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
            group["initial_wd"] = group["weight_decay"]
        return optimizer

    def forward(self, idx, targets, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = softcap(logits, self.output_mult)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction=reduction)

# =============================================================================
# Optimizer: MuonAdamW (Muon for matrices, AdamW for embeddings/scalars)
# =============================================================================

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    # coupled, conditional weight decay: only decay where it agrees with the update
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0)
        self._adamw_lr_t = torch.tensor(0.0)
        self._adamw_beta1_t = torch.tensor(0.0)
        self._adamw_beta2_t = torch.tensor(0.0)
        self._adamw_eps_t = torch.tensor(0.0)
        self._adamw_wd_t = torch.tensor(0.0)
        self._muon_momentum_t = torch.tensor(0.0)
        self._muon_lr_t = torch.tensor(0.0)
        self._muon_wd_t = torch.tensor(0.0)
        self._muon_beta2_t = torch.tensor(0.0)

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None: continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            adamw_step_fused(p, p.grad, state["exp_avg"], state["exp_avg_sq"],
                             self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                             self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group["params"]
        p = params[0]
        state = self.state[p]
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(len(params), *shape, dtype=dtype, device=device)
            s = (len(params), shape[-2], 1) if shape[-2] >= shape[-1] else (len(params), 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"])
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params, state["momentum_buffer"],
                        state["second_momentum_buffer"], self._muon_momentum_t, self._muon_lr_t,
                        self._muon_wd_t, self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw": self._step_adamw(group)
            elif group["kind"] == "muon": self._step_muon(group)

# =============================================================================
# Data: fixed-length rows over a flat uint16 token file
# =============================================================================

class TokenRows:
    def __init__(self, path, max_tokens=None):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        n = len(self.tokens) if max_tokens is None else min(max_tokens, len(self.tokens))
        self.num_rows = (n - 1) // SEQ_LEN

    def get(self, rows):
        starts = np.asarray(rows, dtype=np.int64) * SEQ_LEN
        return torch.from_numpy(self.tokens[starts[:, None] + np.arange(SEQ_LEN + 1)].astype(np.int64))


class DataLoader:
    """Reshuffles rows at every epoch; the seed sets the data order."""

    def __init__(self, path, B, max_tokens, seed, device="cuda"):
        self.data = TokenRows(path, max_tokens)
        self.B, self.seed, self.device = B, seed, device
        self.steps_per_epoch = self.data.num_rows // B
        self.epoch, self.pos = 0, self.steps_per_epoch

    def __next__(self):
        if self.pos >= self.steps_per_epoch:
            self.epoch += 1
            self.pos = 0
            g = torch.Generator()
            g.manual_seed(self.seed * 1000003 + self.epoch)
            self.perm = torch.randperm(self.data.num_rows, generator=g)
        rows = self.perm[self.pos * self.B:(self.pos + 1) * self.B].numpy()
        batch = self.data.get(rows).pin_memory().to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()

# =============================================================================
# Per-token NLL on a held-out split
# =============================================================================

@torch.no_grad()
def per_token_nll(model, data):
    model.eval()
    out = torch.empty(data.num_rows, SEQ_LEN, dtype=torch.float16)
    for i in range(0, data.num_rows, device_batch_size):
        rows = np.arange(i, min(i + device_batch_size, data.num_rows))
        batch = data.get(rows).cuda()
        if len(rows) < device_batch_size:  # pad so the compiled shape never changes
            batch = torch.cat([batch, batch[-1:].expand(device_batch_size - len(rows), -1)])
        with autocast_ctx:
            nll = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), reduction="none")
        out[i:i + len(rows)] = nll.view(device_batch_size, SEQ_LEN)[:len(rows)].half().cpu()
    model.train()
    return out.view(-1)

# =============================================================================
# Training
# =============================================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

steps_per_epoch = U // TOKENS_PER_STEP
U_eff = steps_per_epoch * TOKENS_PER_STEP   # trim U so epochs are whole optimizer steps
total_steps = steps_per_epoch * args.R
precycle_steps = total_steps // 2
cycle_steps = steps_per_epoch if args.R > 1 else total_steps - precycle_steps
warmup_steps = max(1, min(WARMUP_STEPS, total_steps // 10))
num_snapshots = (total_steps - precycle_steps) // cycle_steps


def get_lr_wd_multipliers(step):
    # step counts completed optimizer steps, starting at 1
    if step <= warmup_steps:
        return step / warmup_steps, step / warmup_steps
    if step <= precycle_steps:
        return 1.0, 1.0
    f = ((step - precycle_steps - 1) % cycle_steps + 1) / cycle_steps
    return 1 - (1 - CYCLE_LR_FLOOR) * f, CYCLE_WD_FLOOR + (1 - CYCLE_WD_FLOOR) * f


val_data = TokenRows(os.path.join(args.data_dir, "val.bin"))
calib_data = TokenRows(os.path.join(args.data_dir, "calib.bin"))
print(f"U={U_eff:,} d={args.d} N={N/1e6:.1f}M R={args.R} eLR={elr:g} wd={wd:g} lr_mult={lr_mult:.4f}")
print(f"steps: {total_steps} total, {steps_per_epoch}/epoch, warmup {warmup_steps}, "
      f"{num_snapshots} snapshots | batch {device_batch_size} x {grad_accum_steps} accum")


def train(seed):
    run_name = f"U{args.U:g}M_d{args.d}_R{args.R}_elr{elr:g}_wd{wd:g}_s{seed}"
    run_dir = os.path.join(args.runs_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    val_path, calib_path = os.path.join(run_dir, "val_nll.f16"), os.path.join(run_dir, "calib_nll.f16")
    for path in (val_path, calib_path):
        if os.path.exists(path):
            os.remove(path)
    print(f"\n=== {run_name} ===")

    torch.manual_seed(seed)
    config = GPTConfig(n_layer=args.d, n_head=args.d, n_embd=64 * args.d)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cuda")
    model.init_weights()
    assert abs(model.num_params() / N - 1) < 1e-3  # N leaves out the tiny value-embedding gates
    optimizer = model.setup_optimizer(lr_mult, wd)
    compiled = torch.compile(model, dynamic=False)

    info = dict(U=args.U, d=args.d, R=args.R, seed=seed, elr=elr, wd=wd, lr_mult=lr_mult, N=N,
                U_eff=U_eff, total_steps=total_steps, steps_per_epoch=steps_per_epoch,
                warmup_steps=warmup_steps, num_snapshots=num_snapshots,
                device_batch_size=device_batch_size, grad_accum_steps=grad_accum_steps,
                val_tokens=val_data.num_rows * SEQ_LEN, calib_tokens=calib_data.num_rows * SEQ_LEN)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(info, f, indent=2)

    train_loader = DataLoader(os.path.join(args.data_dir, "train.bin"), device_batch_size,
                              max_tokens=U_eff + SEQ_LEN, seed=seed)
    assert train_loader.steps_per_epoch == steps_per_epoch * grad_accum_steps
    x, y = next(train_loader)
    snapshots = []
    t0 = time.time()
    for step in range(1, total_steps + 1):
        lrm, wdm = get_lr_wd_multipliers(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            group["weight_decay"] = group["initial_wd"] * wdm
        train_loss = torch.zeros((), device="cuda")
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss = compiled(x, y)
            (loss / grad_accum_steps).backward()
            train_loss += loss.detach() / grad_accum_steps
            x, y = next(train_loader)
        optimizer.step()
        model.zero_grad(set_to_none=True)

        if step % steps_per_epoch == 0 or step == total_steps:
            train_loss = train_loss.item()
            print(f"step {step}/{total_steps} | epoch {step / steps_per_epoch:.2f} | "
                  f"train loss {train_loss:.4f} | {(time.time() - t0) / 60:.1f} min")
            if not math.isfinite(train_loss):
                print("diverged, stopping")
                return
        if step > precycle_steps and (step - precycle_steps) % cycle_steps == 0:
            val_nll, calib_nll = per_token_nll(compiled, val_data), per_token_nll(compiled, calib_data)
            with open(val_path, "ab") as f:
                f.write(val_nll.numpy().tobytes())
            with open(calib_path, "ab") as f:
                f.write(calib_nll.numpy().tobytes())
            snapshots.append(dict(step=step, val_loss=val_nll.float().mean().item(),
                                  calib_loss=calib_nll.float().mean().item()))
            print(f"snapshot {len(snapshots)}/{num_snapshots} | val {snapshots[-1]['val_loss']:.4f} "
                  f"| calib {snapshots[-1]['calib_loss']:.4f}")

    info.update(snapshots=snapshots, train_minutes=(time.time() - t0) / 60)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(info, f, indent=2)
    new_file = not os.path.exists(args.runs_csv)
    with open(args.runs_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["U", "d", "R", "seed", "eLR", "WD", "path"])
        writer.writerow([f"{args.U:g}", args.d, args.R, seed, f"{elr:g}", f"{wd:g}", run_dir])


for seed in [int(s) for s in args.s.split(",")]:
    train(seed)
