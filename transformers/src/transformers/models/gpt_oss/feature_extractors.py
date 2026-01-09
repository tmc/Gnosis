"""
PyTorch modules for extracting high-level features from LLM attention patterns,
hidden states, and confidence scores.  (Mask-free variant)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional

# ======================================================================================
# SECTION 1: UTILITIES & HELPERS
# ======================================================================================
def _safe_dtype_param(module: nn.Module):
    for p in module.parameters():
        return p.dtype
    return torch.bfloat16

# Safe, non-generator AMP-disabler: returns a context manager
try:
    from torch.cuda.amp import autocast as _autocast
except Exception:
    _autocast = None

def no_amp_fp32(enabled: bool = True):
    """Return a context manager that disables AMP to run a block in fp32 for numerical stability."""
    if not enabled or _autocast is None:
        return nullcontext()
    return _autocast(enabled=False)

def _num_groups(c: int, g: int = 8) -> int:
    """Choose a valid number of groups for GroupNorm / grouped convs."""
    for k in [g, 6, 4, 3, 2, 1]:
        if c % k == 0:
            return k
    return 1

def _module_param_dtype(mod: nn.Module) -> torch.dtype:
    for p in mod.parameters():
        return p.dtype
    return torch.float32

def percentile(x: torch.Tensor, q: float, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """q-th percentile via kthvalue (q in [0,1])."""
    n = x.shape[dim] if dim is not None else x.numel()
    k = max(1, int(n * q))
    if dim is None:
        vals, _ = torch.kthvalue(x.view(-1), k)
        return vals
    vals, _ = torch.kthvalue(x, k, dim=dim, keepdim=keepdim)
    return vals

# ======================================================================================
# SECTION 2: SET TRANSFORMER PRIMITIVES
# ======================================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, Tq, D = Q.shape
        Tk = K.size(1)

        q = self.q(Q).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k(K).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v(V).view(B, Tk, self.h, self.dk).transpose(1, 2)

        with no_amp_fp32(True):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
            attn = scores.softmax(dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.o(out)

class MAB(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, ff_mult: int = 2):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, pdrop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        x = self.ln1(Q + self.mha(Q, K, K))
        x = self.ln2(x + self.ff(x))
        return x

class SAB(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([MAB(d_model, n_heads, pdrop) for _ in range(num_layers)])
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for mab in self.layers:
            X = mab(X, X)
        return X

class PMA(nn.Module):
    def __init__(self, d_model: int, num_seeds: int = 4, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(num_seeds, d_model) / math.sqrt(d_model))
        self.mab = MAB(d_model, n_heads, pdrop)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        S = self.S.unsqueeze(0).expand(B, -1, -1)
        return self.mab(S, X)

# ======================================================================================
# SECTION 3: ATTENTION MAP FEATURE EXTRACTORS
# ======================================================================================

class ResNetBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_num_groups(out_c), out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_num_groups(out_c), out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_num_groups(out_c), out_c)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.gelu(out)

class AttnFeatureExtractorLite(nn.Module):
    """
    Per-(layer,head) multi-view CNN features + spectral/graph stats,
    mixed with a Set Transformer and pooled by PMA.
    """
    def __init__(
        self,
        D_ATT: int = 512, d_tok: int = 160, cnn_c: int = 64, K: int = 4,
        max_layers: int = 128, max_heads: int = 256, sab_layers: int = 1,
        sab_heads: int = 4, pdrop: float = 0.10,
    ):
        super().__init__()
        self.d_tok = d_tok

        def stem(in_c=3, c=cnn_c):
            return nn.Sequential(
                nn.Conv2d(in_c, c, 3, padding=1), nn.GroupNorm(_num_groups(c), c), nn.GELU(),
                nn.Conv2d(c, c, 3, padding=1),   nn.GroupNorm(_num_groups(c), c), nn.GELU()
            )
        self.cnn_s0, self.cnn_s1, self.cnn_s2 = stem(), stem(), stem()
        self.proj = nn.Linear(6 * cnn_c + 13, d_tok)
        self.layer_emb = nn.Embedding(max_layers, d_tok)
        self.head_emb  = nn.Embedding(max_heads, d_tok)
        self.sab = SAB(d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=sab_layers)
        self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=sab_heads, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_tok, 2 * d_tok), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_ATT)
        )

    def _coord(self, B_L_H: int, k: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B_L_H, -1, -1, -1)
        return coords

    @torch.no_grad()
    def _spectral_graph_stats(self, A: torch.Tensor) -> torch.Tensor:
        B, _, k, _ = A.shape
        A2 = A.squeeze(1)
        Xf = torch.fft.rfft2(A2, dim=(-2, -1))
        P  = (Xf.real**2 + Xf.imag**2) + 1e-12
        Psum = P.sum(dim=(-2, -1), keepdim=False)
        Pn = P / (Psum.view(B, 1, 1) + 1e-12)

        fy = torch.linspace(-0.5, 0.5, steps=k, device=A.device)
        fx = torch.linspace(0.0,  0.5, steps=(k // 2) + 1, device=A.device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        rad = torch.sqrt(yy**2 + xx**2)
        max_r = rad.max().clamp_min(1e-6)
        r1, r2, r3 = 0.15*max_r, 0.35*max_r, 0.60*max_r

        def band(mask):
            m = mask.to(P.dtype).unsqueeze(0)
            e = (P * m).sum(dim=(-2, -1))
            return (e / (Psum + 1e-12)).unsqueeze(-1)

        Pl = band(rad <= r1); Pm = band((rad > r1) & (rad <= r2)); Ph = band((rad > r2) & (rad <= r3)); Pv = band(rad > r3)
        sent = (-(Pn * Pn.log()).sum(dim=(-2, -1))).unsqueeze(-1)

        rows = A2.clamp_min(0); cols = rows.transpose(-1, -2)
        rsum = rows.sum(dim=-1); csum = cols.sum(dim=-1)
        rvar = rsum.var(dim=-1, unbiased=False).unsqueeze(-1)
        cvar = csum.var(dim=-1, unbiased=False).unsqueeze(-1)

        def _entropy(x, dim=-1):
            p = (x / (x.sum(dim=dim, keepdim=True) + 1e-8)).clamp_min(1e-8)
            return (-(p * p.log()).sum(dim=dim, keepdim=True))

        rent = _entropy(rows, dim=-1).mean(dim=-2)
        cent = _entropy(cols, dim=-1).mean(dim=-2)

        total = A2.abs().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1) + 1e-6
        diag  = torch.diagonal(A2, dim1=-2, dim2=-1).abs().sum(dim=-1, keepdim=True)
        diag_ratio = diag / total

        def band_energy(width: int):
            mask2d = torch.zeros(k, k, device=A2.device, dtype=A2.dtype)
            for d in range(-width, width + 1):
                diag_len = k - abs(d)
                if diag_len > 0:
                    mask2d += torch.diag(torch.ones(diag_len, device=A2.device, dtype=A2.dtype), diagonal=d)
            m = mask2d.clamp_max(1).unsqueeze(0)
            e = (A2.abs() * m).sum(dim=(-1, -2))
            return (e / total.squeeze(-1)).unsqueeze(-1)

        band_w1 = band_energy(max(1, k // 32)); band_w2 = band_energy(max(2, k // 16))
        D_trace  = rsum.sum(dim=-1, keepdim=True)
        A_trace  = A2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        lap_trace = D_trace - A_trace

        return torch.cat([sent, Pl, Pm, Ph, Pv, rvar, cvar, rent, cent, diag_ratio, band_w1, band_w2, lap_trace], dim=-1)

    def _cnn_gpool(self, x: torch.Tensor, stem: nn.Module) -> torch.Tensor:
        target_dtype = _module_param_dtype(stem)
        z = stem(x.to(target_dtype))
        gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
        gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
        return torch.cat([gavg, gmax], dim=-1)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        B, L, H, k, _ = attn.shape
        T = L * H
        maps = attn.reshape(B * T, 1, k, k)
        coords = self._coord(B * T, k, maps.device, dtype=maps.dtype)
        x = torch.cat([maps, coords], dim=1)

        s0, s1, s2 = x, F.avg_pool2d(x, 2, ceil_mode=True), F.avg_pool2d(x, 4, ceil_mode=True)

        # CNN in module's dtype to avoid mismatch; stats in fp32
        f0 = self._cnn_gpool(s0, self.cnn_s0)
        f1 = self._cnn_gpool(s1, self.cnn_s1)
        f2 = self._cnn_gpool(s2, self.cnn_s2)
        with no_amp_fp32(True):
            stats = self._spectral_graph_stats(maps.to(torch.float32))

        per_map = torch.cat([f0, f1, f2, stats.to(f0.dtype)], dim=-1)
        toks = self.proj(per_map).view(B, T, self.d_tok)

        l_idx = torch.arange(L, device=attn.device).repeat_interleave(H)
        h_idx = torch.arange(H, device=attn.device).repeat(L)
        pe = (self.layer_emb(l_idx) + self.head_emb(h_idx)).to(toks.dtype)
        toks = toks + pe.unsqueeze(0)

        toks = self.sab(toks)
        pooled = self.pma(toks)
        return self.out(pooled.flatten(1))

class AttnFeatureExtractorLite_D2(nn.Module):
    """
    Per-map CNN -> grid conv over (L,H) -> PMA.
    """
    def __init__(
        self,
        D_ATT: int = 512,
        d_grid: int = 128,
        cnn_channels: tuple = (32, 64, 128),
        grid_conv_layers: int = 2,
        K: int = 4,
        pdrop: float = 0.10,
    ):
        super().__init__()
        self.d_grid = d_grid
        num_stats = 13

        self.cnn_stem = nn.Sequential(
            nn.Conv2d(3, cnn_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_num_groups(cnn_channels[0]), cnn_channels[0]),
            nn.GELU()
        )
        self.cnn_body = nn.Sequential(
            ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
            ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2)
        )
        cnn_out_dim = cnn_channels[-1] * 2
        self.proj_per_map = nn.Linear(cnn_out_dim + num_stats, d_grid)

        grid_layers = [
            nn.Sequential(
                nn.Conv2d(d_grid, d_grid, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_num_groups(d_grid), d_grid),
                nn.GELU()
            ) for _ in range(grid_conv_layers)
        ]
        self.grid_processor = nn.ModuleList(grid_layers)

        self.pma = PMA(d_model=d_grid, num_seeds=K, n_heads=4, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_grid, 2 * d_grid), nn.GELU(), nn.LayerNorm(2 * d_grid), nn.Dropout(pdrop),
            nn.Linear(2 * d_grid, D_ATT)
        )

    def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    @torch.no_grad()
    def _spectral_graph_stats(self, A: torch.Tensor) -> torch.Tensor:
        B, _, k, _ = A.shape
        A2 = A.squeeze(1)
        Xf = torch.fft.rfft2(A2, dim=(-2, -1))
        P  = (Xf.real**2 + Xf.imag**2) + 1e-12
        Psum = P.sum(dim=(-2, -1), keepdim=False)
        Pn = P / (Psum.view(B, 1, 1) + 1e-12)

        fy = torch.linspace(-0.5, 0.5, steps=k, device=A.device)
        fx = torch.linspace(0.0,  0.5, steps=(k // 2) + 1, device=A.device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        rad = torch.sqrt(yy**2 + xx**2)
        max_r = rad.max().clamp_min(1e-6)
        r1, r2, r3 = 0.15*max_r, 0.35*max_r, 0.60*max_r

        def band(mask):
            m = mask.to(P.dtype).unsqueeze(0)
            e = (P * m).sum(dim=(-2, -1))
            return (e / (Psum + 1e-12)).unsqueeze(-1)

        Pl = band(rad <= r1); Pm = band((rad > r1) & (rad <= r2)); Ph = band((rad > r2) & (rad <= r3)); Pv = band(rad > r3)
        sent = (-(Pn * Pn.log()).sum(dim=(-2, -1))).unsqueeze(-1)

        rows = A2.clamp_min(0); cols = rows.transpose(-1, -2)
        rsum = rows.sum(dim=-1); csum = cols.sum(dim=-1)
        rvar = rsum.var(dim=-1, unbiased=False).unsqueeze(-1)
        cvar = csum.var(dim=-1, unbiased=False).unsqueeze(-1)

        def _entropy(x, dim=-1):
            p = (x / (x.sum(dim=dim, keepdim=True) + 1e-8)).clamp_min(1e-8)
            return (-(p * p.log()).sum(dim=dim, keepdim=True))

        rent = _entropy(rows, dim=-1).mean(dim=-2)
        cent = _entropy(cols, dim=-1).mean(dim=-2)

        total = A2.abs().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1) + 1e-6
        diag  = torch.diagonal(A2, dim1=-2, dim2=-1).abs().sum(dim=-1, keepdim=True)
        diag_ratio = diag / total

        def band_energy(width: int):
            mask2d = torch.zeros(k, k, device=A2.device, dtype=A2.dtype)
            for d in range(-width, width + 1):
                diag_len = k - abs(d)
                if diag_len > 0:
                    mask2d += torch.diag(torch.ones(diag_len, device=A2.device, dtype=A2.dtype), diagonal=d)
            m = mask2d.clamp_max(1).unsqueeze(0)
            e = (A2.abs() * m).sum(dim=(-1, -2))
            return (e / total.squeeze(-1)).unsqueeze(-1)

        band_w1 = band_energy(max(1, k // 32)); band_w2 = band_energy(max(2, k // 16))
        D_trace  = rsum.sum(dim=-1, keepdim=True)
        A_trace  = A2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        lap_trace = D_trace - A_trace

        return torch.cat([sent, Pl, Pm, Ph, Pv, rvar, cvar, rent, cent, diag_ratio, band_w1, band_w2, lap_trace], dim=-1)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        B, L, H, k, _ = attn.shape
        T = L * H

        maps = attn.reshape(B * T, 1, k, k)
        coords = self._coord(B * T, k, maps.device, maps.dtype)
        x_maps = torch.cat([maps, coords], dim=1)

        # Run CNN in module dtype to match weights; compute stats in fp32
        stem_dtype = _module_param_dtype(self.cnn_stem)
        cnn_features = self.cnn_stem(x_maps.to(stem_dtype))
        cnn_features = self.cnn_body(cnn_features)
        gavg = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
        gmax = F.adaptive_max_pool2d(cnn_features, 1).flatten(1)
        cnn_vec = torch.cat([gavg, gmax], dim=-1)

        with no_amp_fp32(True):
            stats_vec = self._spectral_graph_stats(maps.to(torch.float32))

        combined_vec = torch.cat([cnn_vec, stats_vec.to(cnn_vec.dtype)], dim=-1)
        grid_feats = self.proj_per_map(combined_vec)

        grid = grid_feats.view(B, L, H, self.d_grid).permute(0, 3, 1, 2).contiguous()
        for layer in self.grid_processor:
            grid = grid + layer(grid)

        pma_input = grid.flatten(2).transpose(1, 2)
        pooled = self.pma(pma_input)
        return self.out(pooled.flatten(1))

# Stronger hierarchical extractor (mask-free; fixed 256x256 input)
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        m = max(8, c // r)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.GELU(), nn.Linear(m, c), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        g = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * g

@torch.no_grad()
def spectral_graph_stats_13(A: torch.Tensor) -> torch.Tensor:
    B, _, k, _ = A.shape
    A2 = A.squeeze(1)
    Xf = torch.fft.rfft2(A2, dim=(-2, -1))
    P  = (Xf.real**2 + Xf.imag**2) + 1e-12
    Psum = P.sum(dim=(-2, -1))
    Pn = P / (Psum.view(B, 1, 1) + 1e-12)

    fy = torch.linspace(-0.5, 0.5, steps=k, device=A.device)
    fx = torch.linspace(0.0,  0.5, steps=(k // 2) + 1, device=A.device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rad = torch.sqrt(yy**2 + xx**2)
    max_r = rad.max().clamp_min(1e-6)
    r1, r2, r3 = 0.15*max_r, 0.35*max_r, 0.60*max_r

    def band(mask):
        m = mask.to(P.dtype).unsqueeze(0)
        e = (P * m).sum(dim=(-2, -1))
        return (e / (Psum + 1e-12)).unsqueeze(-1)

    Pl = band(rad <= r1)
    Pm = band((rad > r1) & (rad <= r2))
    Ph = band((rad > r2) & (rad <= r3))
    Pv = band(rad > r3)
    sent = (-(Pn * Pn.log()).sum(dim=(-2, -1))).unsqueeze(-1)

    rows = A2.clamp_min(0)
    cols = rows.transpose(-1, -2)
    rsum = rows.sum(dim=-1)
    csum = cols.sum(dim=-1)
    rvar = rsum.var(dim=-1, unbiased=False).unsqueeze(-1)
    cvar = csum.var(dim=-1, unbiased=False).unsqueeze(-1)

    def _entropy(x, dim=-1):
        p = (x / (x.sum(dim=dim, keepdim=True) + 1e-8)).clamp_min(1e-8)
        return (-(p * p.log()).sum(dim=dim, keepdim=True))

    rent = _entropy(rows, dim=-1).mean(dim=-2)
    cent = _entropy(cols, dim=-1).mean(dim=-2)

    total = A2.abs().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1) + 1e-6
    diag  = torch.diagonal(A2, dim1=-2, dim2=-1).abs().sum(dim=-1, keepdim=True)
    diag_ratio = diag / total

    def band_energy(width: int):
        mask2d = torch.zeros(k, k, device=A2.device, dtype=A2.dtype)
        for d in range(-width, width + 1):
            n = k - abs(d)
            if n > 0:
                mask2d += torch.diag(torch.ones(n, device=A2.device, dtype=A2.dtype), diagonal=d)
        m = mask2d.clamp_max(1).unsqueeze(0)
        e = (A2.abs() * m).sum(dim=(-1, -2))
        return (e / total.squeeze(-1)).unsqueeze(-1)

    band_w1 = band_energy(max(1, k // 32))
    band_w2 = band_energy(max(2, k // 16))

    D_trace  = rsum.sum(dim=-1, keepdim=True)
    A_trace  = A2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    lap_trace = D_trace - A_trace

    return torch.cat([sent, Pl, Pm, Ph, Pv, rvar, cvar, rent, cent,
                      diag_ratio, band_w1, band_w2, lap_trace], dim=-1)

# class AttnFeatureExtractorLite_D3(nn.Module):
#     """
#     Stronger variant of D2 (mask-free):
#       - Per-map CNN (+SE) on [map, coords]
#       - [optional] 13-d spectral/graph stats (no SVD), fixed 256x256
#       - Layer/Head embeddings
#       - Axial depthwise grid mixing (heads then layers), residual
#       - PMA pooling to D_ATT
#     """
#     def __init__(
#         self,
#         D_ATT: int = 512,
#         d_grid: int = 128,
#         cnn_channels: tuple = (32, 64, 128),
#         grid_conv_layers: int = 2,
#         K: int = 4,
#         pdrop: float = 0.10,
#         max_layers: int = 128,
#         max_heads: int = 256,
#         use_spectral: bool = False,  # <-- new toggle (default off)
#     ):
#         super().__init__()
#         self.d_grid = d_grid
#         self.use_spectral = use_spectral

#         # Per-map CNN (no mask channel)
#         in_c = 3
#         self.cnn_stem = nn.Sequential(
#             nn.Conv2d(in_c, cnn_channels[0], 3, 1, 1, bias=False),
#             nn.GroupNorm(_num_groups(cnn_channels[0]), cnn_channels[0]),
#             nn.GELU()
#         )
#         self.cnn_body = nn.Sequential(
#             ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
#             ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2),
#         )
#         self.se = SEBlock(cnn_channels[-1])
#         cnn_out_dim = cnn_channels[-1] * 2  # GAP || GMP

#         # Input dim depends on whether spectral stats are used
#         in_dim = cnn_out_dim + (13 if self.use_spectral else 0)
#         self.proj_per_map = nn.Linear(in_dim, d_grid)

#         # Embeddings
#         self.layer_emb = nn.Embedding(max_layers, d_grid)
#         self.head_emb  = nn.Embedding(max_heads, d_grid)
#         nn.init.normal_(self.layer_emb.weight, std=0.02)
#         nn.init.normal_(self.head_emb.weight,  std=0.02)

#         # Axial depthwise grid mixing
#         axial = []
#         for _ in range(grid_conv_layers):
#             axial += [
#                 nn.Conv2d(d_grid, d_grid, kernel_size=(1,3), padding=(0,1),
#                           groups=d_grid, bias=False),  # heads axis
#                 nn.GELU(),
#                 nn.Conv2d(d_grid, d_grid, kernel_size=(3,1), padding=(1,0),
#                           groups=d_grid, bias=False),  # layers axis
#                 nn.GELU(),
#                 nn.Conv2d(d_grid, d_grid, kernel_size=1, bias=False),  # pointwise fuse
#                 nn.GroupNorm(_num_groups(d_grid), d_grid),
#             ]
#         self.grid_processor = nn.Sequential(*axial)

#         # PMA head
#         self.pma = PMA(d_model=d_grid, num_seeds=K, n_heads=4, pdrop=pdrop)
#         self.out = nn.Sequential(
#             nn.Linear(K * d_grid, 2 * d_grid), nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(2 * d_grid, D_ATT)
#         )

#     def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
#         ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
#         xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
#         yy, xx = torch.meshgrid(ys, xs, indexing="ij")
#         return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

#     def forward(self, attn: torch.Tensor) -> torch.Tensor:
#         B, L, H, k, k2 = attn.shape
#         T = L * H
#         device = attn.device

#         # Per-map channels: [map, coords]
#         maps = attn.reshape(B * T, 1, k, k)
#         coords = self._coord(B * T, k, maps.device, maps.dtype)
#         x_maps = torch.cat([maps, coords], dim=1)

#         # CNN in module dtype
#         stem_dtype = _module_param_dtype(self.cnn_stem)
#         z = self.cnn_stem(x_maps.to(stem_dtype))
#         z = self.cnn_body(z)
#         z = self.se(z)
#         gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
#         gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
#         cnn_vec = torch.cat([gavg, gmax], dim=-1)

#         if self.use_spectral:
#             with no_amp_fp32(True):
#                 stats_vec = spectral_graph_stats_13(maps.to(torch.float32))
#             per_map = torch.cat([cnn_vec, stats_vec.to(cnn_vec.dtype)], dim=-1)
#         else:
#             per_map = cnn_vec

#         feats = self.proj_per_map(per_map)                       # (B*T, d_grid)

#         # (B, d, L, H) + embeddings
#         tok = feats.view(B, L, H, self.d_grid)
#         tok = tok + self.layer_emb(torch.arange(L, device=device)).view(1, L, 1, -1) \
#                   + self.head_emb(torch.arange(H, device=device)).view(1, 1, H, -1)
#         grid = tok.permute(0, 3, 1, 2).contiguous()              # (B, d, L, H)

#         grid = grid + self.grid_processor(grid)

#         pma_in = grid.flatten(2).transpose(1, 2)                 # (B, L*H, d)
#         pooled = self.pma(pma_in)                                # (B, K, d)
#         return self.out(pooled.flatten(1))                       # (B, D_ATT)


# class AttnFeatureExtractorLite_D3(nn.Module):
#     """
#     Stronger variant of D2 (mask-free):
#       - Per-map CNN (+SE) on [map, coords]  [enabled if feature_mode in {"cnn","both"}]
#       - 13-d spectral/graph stats (no SVD)  [enabled if feature_mode in {"spectral","both"}]
#       - Layer/Head embeddings
#       - Axial depthwise grid mixing (heads then layers), residual
#       - PMA pooling to D_ATT
#     """
#     def __init__(
#         self,
#         D_ATT: int = 512,
#         d_grid: int = 128,
#         cnn_channels: tuple = (32, 64, 128),
#         grid_conv_layers: int = 2,
#         K: int = 4,
#         pdrop: float = 0.10,
#         max_layers: int = 128,
#         max_heads: int = 256,
#         feature_mode: str = "cnn",        # "cnn" | "spectral" | "both"
#         use_spectral: Optional[bool] = None,  # legacy toggle: True -> "both", False -> "cnn"
#     ):
#         super().__init__()
#         # --- mode resolution (legacy flag supported) ---
#         if use_spectral is not None:
#             feature_mode = "both" if use_spectral else "cnn"
#         if feature_mode not in {"cnn", "spectral", "both"}:
#             raise ValueError(f"feature_mode must be one of 'cnn', 'spectral', 'both'; got {feature_mode!r}")
#         self.feature_mode = feature_mode
#         self.d_grid = d_grid

#         # Per-map CNN (no mask channel) — only used if mode includes CNN
#         in_c = 3  # [map, y, x]
#         self.cnn_stem = nn.Sequential(
#             nn.Conv2d(in_c, cnn_channels[0], 3, 1, 1, bias=False),
#             nn.GroupNorm(_num_groups(cnn_channels[0]), cnn_channels[0]),
#             nn.GELU()
#         )
#         self.cnn_body = nn.Sequential(
#             ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
#             ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2),
#         )
#         self.se = SEBlock(cnn_channels[-1])
#         cnn_out_dim = cnn_channels[-1] * 2  # GAP || GMP

#         # Input dim depends on mode
#         add_spec = 13 if self.feature_mode in {"spectral", "both"} else 0
#         use_cnn = self.feature_mode in {"cnn", "both"}
#         in_dim = (cnn_out_dim if use_cnn else 0) + add_spec
#         self.proj_per_map = nn.Linear(in_dim, d_grid)

#         # Embeddings
#         self.layer_emb = nn.Embedding(max_layers, d_grid)
#         self.head_emb  = nn.Embedding(max_heads, d_grid)
#         nn.init.normal_(self.layer_emb.weight, std=0.02)
#         nn.init.normal_(self.head_emb.weight,  std=0.02)

#         # Axial depthwise grid mixing
#         axial = []
#         for _ in range(grid_conv_layers):
#             axial += [
#                 nn.Conv2d(d_grid, d_grid, kernel_size=(1,3), padding=(0,1),
#                           groups=d_grid, bias=False),  # heads axis
#                 nn.GELU(),
#                 nn.Conv2d(d_grid, d_grid, kernel_size=(3,1), padding=(1,0),
#                           groups=d_grid, bias=False),  # layers axis
#                 nn.GELU(),
#                 nn.Conv2d(d_grid, d_grid, kernel_size=1, bias=False),  # pointwise fuse
#                 nn.GroupNorm(_num_groups(d_grid), d_grid),
#             ]
#         self.grid_processor = nn.Sequential(*axial)

#         # PMA head
#         self.pma = PMA(d_model=d_grid, num_seeds=K, n_heads=4, pdrop=pdrop)
#         self.out = nn.Sequential(
#             nn.Linear(K * d_grid, 2 * d_grid), nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(2 * d_grid, D_ATT)
#         )

#     def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
#         ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
#         xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
#         yy, xx = torch.meshgrid(ys, xs, indexing="ij")
#         return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

#     def forward(self, attn: torch.Tensor) -> torch.Tensor:
#         B, L, H, k, k2 = attn.shape
#         T = L * H
#         device = attn.device

#         per_chunks = []

#         # CNN branch
#         if self.feature_mode in {"cnn", "both"}:
#             maps = attn.reshape(B * T, 1, k, k)
#             coords = self._coord(B * T, k, maps.device, maps.dtype)
#             x_maps = torch.cat([maps, coords], dim=1)

#             stem_dtype = _module_param_dtype(self.cnn_stem)
#             z = self.cnn_stem(x_maps.to(stem_dtype))
#             z = self.cnn_body(z)
#             z = self.se(z)
#             gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
#             gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
#             cnn_vec = torch.cat([gavg, gmax], dim=-1)
#             per_chunks.append(cnn_vec)

#         # Spectral branch
#         if self.feature_mode in {"spectral", "both"}:
#             maps = attn.reshape(B * T, 1, k, k)  # ensure maps defined if CNN skipped
#             with no_amp_fp32(True):
#                 stats_vec = spectral_graph_stats_13(maps.to(torch.float32))
#             per_chunks.append(stats_vec.to(per_chunks[0].dtype if per_chunks else stats_vec.dtype))

#         # Concatenate selected chunks
#         per_map = per_chunks[0] if len(per_chunks) == 1 else torch.cat(per_chunks, dim=-1)

#         feats = self.proj_per_map(per_map)                       # (B*T, d_grid)

#         # (B, d, L, H) + embeddings
#         tok = feats.view(B, L, H, self.d_grid)
#         tok = tok + self.layer_emb(torch.arange(L, device=device)).view(1, L, 1, -1) \
#                   + self.head_emb(torch.arange(H, device=device)).view(1, 1, H, -1)
#         grid = tok.permute(0, 3, 1, 2).contiguous()              # (B, d, L, H)

#         grid = grid + self.grid_processor(grid)

#         pma_in = grid.flatten(2).transpose(1, 2)                 # (B, L*H, d)
#         pooled = self.pma(pma_in)                                # (B, K, d)
#         return self.out(pooled.flatten(1))                       # (B, D_ATT)

#************************************************************************************************
#D4, GPT

def _num_groups(c: int, g: int = 8) -> int:
    """Pick a valid GN group count that divides channels."""
    for k in [g, 6, 4, 3, 2, 1]:
        if c % k == 0:
            return k
    return 1

def _module_param_dtype(mod: nn.Module) -> torch.dtype:
    for p in mod.parameters(recurse=True):
        return p.dtype
    return torch.float32

def no_amp_fp32(enabled: bool = True):
    """Context manager: disable autocast to compute in fp32 (safer numerics)."""
    try:
        from torch.cuda.amp import autocast as _autocast
        return _autocast(enabled=False) if enabled else nullcontext()
    except Exception:
        return nullcontext()

class DropPath(nn.Module):
    """Stochastic depth."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

# =============================================================================
# Set Transformer bits (MHA/MAB/PMA)
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, Tq, D = Q.shape
        Tk = K.size(1)

        q = self.q(Q).view(B, Tq, self.h, self.dk).transpose(1, 2)  # (B,h,Tq,dk)
        k = self.k(K).view(B, Tk, self.h, self.dk).transpose(1, 2)  # (B,h,Tk,dk)
        v = self.v(V).view(B, Tk, self.h, self.dk).transpose(1, 2)  # (B,h,Tk,dk)

        with no_amp_fp32(True):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
            attn = scores.softmax(dim=-1)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v)                                # (B,h,Tq,dk)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)      # (B,Tq,D)
        return self.o(out)

class MAB(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, pdrop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        x = self.ln1(Q + self.mha(Q, K, K))
        x = self.ln2(x + self.ff(x))
        return x

class PMA(nn.Module):
    """Pooling by Multi-head Attention (Set Transformer)."""
    def __init__(self, d_model: int, num_seeds: int = 4, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(num_seeds, d_model) / math.sqrt(d_model))
        self.mab = MAB(d_model, n_heads, pdrop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        S = self.S.unsqueeze(0).expand(B, -1, -1)   # (B,K,d)
        return self.mab(S, X)                        # (B,K,d)

# =============================================================================
# ConvNeXt-lite encoder (shared across input scales)
# =============================================================================

class ConvNeXtLiteBlock(nn.Module):
    """
    Minimal ConvNeXt-style block:
      depthwise 7x7 -> pointwise MLP (expand 4x) -> GN -> residual
    """
    def __init__(self, c: int, mlp_mult: int = 4, pdrop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c, bias=False)
        self.gn = nn.GroupNorm(_num_groups(c), c)
        self.pw1 = nn.Conv2d(c, c * mlp_mult, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(c * mlp_mult, c, kernel_size=1, bias=False)
        self.drop = nn.Dropout2d(pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dw(x)
        z = self.gn(z)
        z = self.pw1(z)
        z = self.act(z)
        z = self.pw2(z)
        z = self.drop(z)
        return x + z

class ConvNeXtLite(nn.Module):
    """
    Shared encoder used at 3 input scales (weights shared across scales).
    """
    def __init__(self, in_c: int = 4, channels=(64, 128, 192), pdrop: float = 0.0):
        super().__init__()
        c1, c2, c3 = channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_c, c1, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(c1), c1),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            ConvNeXtLiteBlock(c1, pdrop=pdrop),
            nn.Conv2d(c1, c2, 2, stride=2, bias=False),  # downsample
        )
        self.stage2 = nn.Sequential(
            ConvNeXtLiteBlock(c2, pdrop=pdrop),
            nn.Conv2d(c2, c3, 2, stride=2, bias=False),  # downsample
        )
        self.stage3 = nn.Sequential(
            ConvNeXtLiteBlock(c3, pdrop=pdrop),
            ConvNeXtLiteBlock(c3, pdrop=pdrop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)
        return z  # (B, c3, h', w')

def _gpool_twice(z: torch.Tensor) -> torch.Tensor:
    gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
    gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
    return torch.cat([gavg, gmax], dim=-1)

# =============================================================================
# Stats: spectral/graph (13), geometric moments (6), entropy extras (4)
# =============================================================================

@torch.no_grad()
def spectral_graph_stats_13(A: torch.Tensor) -> torch.Tensor:
    """
    A: (B,1,k,k)
    Returns 13-d vector per map: spectral entropy bands + row/col stats + diag/band energies + Laplacian trace.
    """
    B, _, k, _ = A.shape
    A2 = A.squeeze(1)
    Xf = torch.fft.rfft2(A2, dim=(-2, -1))
    P  = (Xf.real**2 + Xf.imag**2) + 1e-12
    Psum = P.sum(dim=(-2, -1))
    Pn = P / (Psum.view(B, 1, 1) + 1e-12)

    fy = torch.linspace(-0.5, 0.5, steps=k, device=A.device, dtype=A.dtype)
    fx = torch.linspace(0.0,  0.5, steps=(k // 2) + 1, device=A.device, dtype=A.dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    rad = torch.sqrt(yy**2 + xx**2)
    max_r = rad.max().clamp_min(1e-6)
    r1, r2, r3 = 0.15*max_r, 0.35*max_r, 0.60*max_r

    def band(mask):
        m = mask.to(P.dtype).unsqueeze(0)
        e = (P * m).sum(dim=(-2, -1))
        return (e / (Psum + 1e-12)).unsqueeze(-1)

    Pl = band(rad <= r1); Pm = band((rad > r1) & (rad <= r2))
    Ph = band((rad > r2) & (rad <= r3)); Pv = band(rad > r3)
    sent = (-(Pn * Pn.log()).sum(dim=(-2, -1))).unsqueeze(-1)

    rows = A2.clamp_min(0); cols = rows.transpose(-1, -2)
    rsum = rows.sum(dim=-1); csum = cols.sum(dim=-1)
    rvar = rsum.var(dim=-1, unbiased=False).unsqueeze(-1)
    cvar = csum.var(dim=-1, unbiased=False).unsqueeze(-1)

    def _entropy(x, dim=-1):
        p = (x / (x.sum(dim=dim, keepdim=True) + 1e-8)).clamp_min(1e-8)
        return (-(p * p.log()).sum(dim=dim, keepdim=True))

    rent = _entropy(rows, dim=-1).mean(dim=-2)
    cent = _entropy(cols, dim=-1).mean(dim=-2)

    total = A2.abs().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1) + 1e-6
    diag  = torch.diagonal(A2, dim1=-2, dim2=-1).abs().sum(dim=-1, keepdim=True)
    diag_ratio = diag / total

    def band_energy(width: int):
        mask2d = torch.zeros(k, k, device=A2.device, dtype=A2.dtype)
        for d in range(-width, width + 1):
            n = k - abs(d)
            if n > 0:
                mask2d += torch.diag(torch.ones(n, device=A2.device, dtype=A2.dtype), diagonal=d)
        m = mask2d.clamp_max(1).unsqueeze(0)
        e = (A2.abs() * m).sum(dim=(-1, -2))
        return (e / total.squeeze(-1)).unsqueeze(-1)

    band_w1 = band_energy(max(1, k // 32)); band_w2 = band_energy(max(2, k // 16))
    D_trace  = rsum.sum(dim=-1, keepdim=True)
    A_trace  = A2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    lap_trace = D_trace - A_trace

    return torch.cat([sent, Pl, Pm, Ph, Pv, rvar, cvar, rent, cent,
                      diag_ratio, band_w1, band_w2, lap_trace], dim=-1)

@torch.no_grad()
def _moment_stats_6(A: torch.Tensor) -> torch.Tensor:
    """
    A: (B,1,k,k), non-negative map. Returns:
      [mu_y, mu_x, var_y, var_x, cov, anisotropy]
    """
    B, _, k, _ = A.shape
    A2 = A.squeeze(1).clamp_min_(0)
    s = A2.sum(dim=(-2, -1), keepdim=True).clamp_min_(1e-8)

    ys = torch.linspace(0.0, 1.0, steps=k, device=A.device, dtype=A.dtype).view(1, k, 1)
    xs = torch.linspace(0.0, 1.0, steps=k, device=A.device, dtype=A.dtype).view(1, 1, k)
    mu_y = (A2 * ys).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
    mu_x = (A2 * xs).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)

    dy = ys - mu_y.view(B, 1, 1)
    dx = xs - mu_x.view(B, 1, 1)
    var_y = (A2 * (dy ** 2)).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
    var_x = (A2 * (dx ** 2)).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
    cov   = (A2 * (dy * dx)).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)

    aniso = ((var_y - var_x).abs()) / (var_y + var_x + 1e-8)
    return torch.stack([mu_y, mu_x, var_y, var_x, cov, aniso], dim=-1)

@torch.no_grad()
def _attn_entropy_extras(A: torch.Tensor) -> torch.Tensor:
    """
    A: (B,1,k,k), non-negative attention (preferably post-softmax).
    Returns 4 scalars per map:
      [H_map_norm, H_row_norm, H_col_norm, MI_norm], each in ~[0,1]
    """
    B, _, k, _ = A.shape
    A = A.clamp_min(1e-12).squeeze(1)                      # (B,k,k)
    Z = A.sum(dim=(-2, -1), keepdim=True)
    P = (A / Z).clamp_min(1e-12)

    # Global entropy over k^2 cells
    H_map = -(P * P.log()).sum(dim=(-2, -1))               # (B,)
    H_map_norm = H_map / math.log(k * k)

    # Marginal entropies
    p_row = P.sum(dim=-1)                                  # (B,k)
    p_col = P.sum(dim=-2)                                  # (B,k)
    H_row = -(p_row * p_row.clamp_min(1e-12).log()).sum(dim=-1)
    H_col = -(p_col * p_col.clamp_min(1e-12).log()).sum(dim=-1)
    H_row_norm = H_row / math.log(k)
    H_col_norm = H_col / math.log(k)

    # Mutual information I(X;Y) = H_row + H_col - H_map
    I_xy = (H_row + H_col - H_map).clamp_min(0.0)
    I_xy_norm = I_xy / math.log(k)

    return torch.stack([H_map_norm, H_row_norm, H_col_norm, I_xy_norm], dim=-1)  # (B,4)

# =============================================================================
# Axial Transformer + TCN mixers over (L, H)
# =============================================================================

class AxialBlock(nn.Module):
    """
    Tokens shape: (B, L, H, d). Two self-attn passes (H-axis then L-axis) + FFN.
    """
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, drop_path: float = 0.0, ff_mult: int = 4):
        super().__init__()
        self.ln_h = nn.LayerNorm(d_model)
        self.ln_l = nn.LayerNorm(d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.mha_h = MultiHeadAttention(d_model, n_heads=n_heads, pdrop=pdrop)
        self.mha_l = MultiHeadAttention(d_model, n_heads=n_heads, pdrop=pdrop)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)
        self.dp3 = DropPath(drop_path)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, L, H, d = tok.shape

        # Heads-axis attention (fix L, attend over H)
        x = tok.reshape(B * L, H, d)
        x = x + self.dp1(self.mha_h(self.ln_h(x), self.ln_h(x), self.ln_h(x)))
        x = x.reshape(B, L, H, d)

        # Layers-axis attention (fix H, attend over L)
        y = x.permute(0, 2, 1, 3).reshape(B * H, L, d)
        y = y + self.dp2(self.mha_l(self.ln_l(y), self.ln_l(y), self.ln_l(y)))
        y = y.reshape(B, H, L, d).permute(0, 2, 1, 3)

        # FFN
        z = y + self.dp3(self.ff(self.ln_f(y)))
        return z

class TCNMix1D(nn.Module):
    """
    Depthwise-dilated 1D conv stacks along L and along H with fusion.
    Expects tokens (B, L, H, d).
    """
    def __init__(self, d_model: int, dilations=(1, 2, 4), pdrop: float = 0.0):
        super().__init__()
        self.dilations = tuple(dilations)

        def dw_stack():
            layers = []
            for dil in self.dilations:
                layers += [
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=dil, dilation=dil, groups=d_model, bias=False),
                    nn.GELU(),
                    nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                    nn.GroupNorm(_num_groups(d_model), d_model),
                ]
            return nn.Sequential(*layers)

        self.mix_L = dw_stack()  # along L
        self.mix_H = dw_stack()  # along H

        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        B, L, H, d = tok.shape
        base = tok

        # Along L (per head)
        x = tok.permute(0, 2, 3, 1)          # (B, H, d, L)
        x = x.reshape(B * H, d, L)
        x = self.mix_L(x)                    # (B*H, d, L)
        x = x.reshape(B, H, d, L).permute(0, 3, 1, 2)  # (B, L, H, d)

        # Along H (per layer)
        y = tok.permute(0, 1, 3, 2)          # (B, L, d, H)
        y = y.reshape(B * L, d, H)
        y = self.mix_H(y)                    # (B*L, d, H)
        y = y.reshape(B, L, d, H).permute(0, 1, 3, 2)  # (B, L, H, d)

        fused = torch.cat([x, y], dim=-1)    # (B, L, H, 2d)
        fused = self.fuse(fused)
        out = base + self.drop(self.ln(fused))
        return out

# =============================================================================
# D4 Extractor
# =============================================================================

class AttnFeatureExtractorLite_D4_gpt(nn.Module):
    """
    D4 (stronger than D1–D3):
      - Per-map multi-scale ConvNeXt-lite (shared weights) on channels [map, coords, diag]
      - Stats: 13 spectral/graph + 6 moment features + 4 entropy extras
      - Layer/Head embeddings
      - Axial Transformer blocks over (L,H) + DropPath
      - Dilated TCN mixers (both axes) for long-range progression patterns
      - PMA pooling to D_ATT

    Input:  attn (B, L, H, k, k)   (nonnegative attention probabilities)
    Output: (B, D_ATT)
    """
    def __init__(
        self,
        D_ATT: int = 512,
        d_tok: int = 192,
        cnn_channels: tuple = (64, 128, 192),
        axial_blocks: int = 3,
        axial_heads: int = 4,
        axial_drop_path: float = 0.10,
        tcn_dilations: tuple = (1, 2, 4),
        K: int = 6,
        pdrop: float = 0.10,
        max_layers: int = 128,
        max_heads: int = 256,
    ):
        super().__init__()
        self.d_tok = d_tok

        # Per-map encoder (shared across 3 scales). Input channels: 1 map + 2 coords + 1 diag = 4
        self.encoder = ConvNeXtLite(in_c=4, channels=cnn_channels, pdrop=pdrop)
        c_last = cnn_channels[-1]
        self._cnn_vec_dim = 2 * c_last                  # (GAP || GMP) per scale
        self._cnn_all_scales_dim = 3 * self._cnn_vec_dim  # s0,s1,s2 concatenated

        # Linear proj from [CNN multi-scale || stats(13+6+4)] -> d_tok
        self.num_stats = 13 + 6 + 4
        self.proj = nn.Linear(self._cnn_all_scales_dim + self.num_stats, d_tok)

        # Layer/Head embeddings
        self.layer_emb = nn.Embedding(max_layers, d_tok)
        self.head_emb  = nn.Embedding(max_heads, d_tok)
        nn.init.normal_(self.layer_emb.weight, std=0.02)
        nn.init.normal_(self.head_emb.weight,  std=0.02)

        # Axial Transformer stacks
        dpr = torch.linspace(0, axial_drop_path, steps=axial_blocks).tolist()
        self.axial = nn.ModuleList([
            AxialBlock(d_model=d_tok, n_heads=axial_heads, pdrop=pdrop, drop_path=dpr[i])
            for i in range(axial_blocks)
        ])

        # TCN mixers along L and H
        self.tcn = TCNMix1D(d_model=d_tok, dilations=tcn_dilations, pdrop=pdrop)

        # PMA pooling + head
        self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=4, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_tok, 2 * d_tok), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_ATT),
        )

    def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        # Return as (B,2,k,k)
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    def _diag_chan(self, B: int, k: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(k, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
        return eye.expand(B, -1, -1, -1)

    def _encode_cnn_multiscale(self, x4: torch.Tensor) -> torch.Tensor:
        """
        x4: (B*T, 4, k, k)
        Returns: (B*T, 3 * 2*c_last) = [s0||s1||s2] pooled vectors
        """
        stem_dtype = _module_param_dtype(self.encoder)
        s0 = x4
        s1 = F.avg_pool2d(x4, 2, ceil_mode=True)
        s2 = F.avg_pool2d(x4, 4, ceil_mode=True)

        z0 = self.encoder(s0.to(stem_dtype))
        z1 = self.encoder(s1.to(stem_dtype))
        z2 = self.encoder(s2.to(stem_dtype))

        v0 = _gpool_twice(z0)
        v1 = _gpool_twice(z1)
        v2 = _gpool_twice(z2)
        return torch.cat([v0, v1, v2], dim=-1)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: (B, L, H, k, k)  — pass non-negative attention (post-softmax preferred)
        """
        B, L, H, k, k2 = attn.shape
        assert k == k2, "attention maps must be square (k x k)"

        T = L * H
        maps = attn.reshape(B * T, 1, k, k)

        # Build channels: [map, coords, diag]
        coords = self._coord(B * T, k, maps.device, maps.dtype)   # (B*T, 2, k, k)
        diagc  = self._diag_chan(B * T, k, maps.device, maps.dtype)  # (B*T, 1, k, k)
        x4 = torch.cat([maps, coords, diagc], dim=1)                  # (B*T, 4, k, k)

        # Per-map multi-scale ConvNeXt-lite pooled vector
        cnn_vec = self._encode_cnn_multiscale(x4)                     # (B*T, 3*2*c_last)

        # Stats in fp32 for stability
        with no_amp_fp32(True):
            stats13 = spectral_graph_stats_13(maps.to(torch.float32)) # (B*T, 13)
            stats6  = _moment_stats_6(maps.to(torch.float32))         # (B*T, 6)
            ent4    = _attn_entropy_extras(maps.to(torch.float32))    # (B*T, 4)
        stats = torch.cat([stats13, stats6, ent4], dim=-1).to(cnn_vec.dtype)  # (B*T, 23)

        per_map = torch.cat([cnn_vec, stats], dim=-1)                 # (B*T, cnn+23)
        feats = self.proj(per_map)                                    # (B*T, d_tok)

        # (B, L, H, d_tok) + embeddings
        tok = feats.view(B, L, H, self.d_tok)
        device = tok.device
        tok = tok + self.layer_emb(torch.arange(L, device=device)).view(1, L, 1, -1) \
                  + self.head_emb(torch.arange(H, device=device)).view(1, 1, H, -1)

        # Axial Transformer stack
        for blk in self.axial:
            tok = blk(tok)

        # Dilated TCN mixers along L and H
        tok = self.tcn(tok)                                           # (B, L, H, d)

        # PMA pooling → (B, D_ATT)
        pma_in = tok.reshape(B, L * H, self.d_tok)

        pooled = self.pma(pma_in)                                     # (B, K, d)
        out = self.out(pooled.flatten(1))                             # (B, D_ATT)
        return out

class AttnFeatureExtractorLite_D4_gpt_NoPos(nn.Module):
    """
    D4 without positional embeddings:
      - Per-map multi-scale ConvNeXt-lite on [map, coords, diag]
      - Stats: 13 spectral/graph + 6 moments + 4 entropy extras
      - NO layer/head positional embeddings
      - Axial Transformer (content-only) + Dilated TCN along L and H
      - PMA pooling → D_ATT
    Input:  attn (B, L, H, k, k)  nonnegative (post-softmax preferred)
    Output: (B, D_ATT)
    """
    def __init__(
        self,
        D_ATT: int = 256,
        d_tok: int = 192,
        cnn_channels: tuple = (64, 128, 192),
        axial_blocks: int = 3,
        axial_heads: int = 4,
        axial_drop_path: float = 0.10,
        tcn_dilations: tuple = (1, 2, 4),
        K: int = 6,
        pdrop: float = 0.10,
        # keep these for interface compatibility; not used for embeddings
        max_layers: int = 128,
        max_heads: int = 256,
    ):
        super().__init__()
        self.d_tok = d_tok

        # ----- per-map multi-scale encoder -----
        self.encoder = ConvNeXtLite(in_c=4, channels=cnn_channels, pdrop=pdrop)
        c_last = cnn_channels[-1]
        self._cnn_vec_dim = 2 * c_last
        self._cnn_all_scales_dim = 3 * self._cnn_vec_dim

        # stats: 13 + 6 + 4
        self.num_stats = 13 + 6 + 4
        self.proj = nn.Linear(self._cnn_all_scales_dim + self.num_stats, d_tok)

        # ----- axial transformer (no positional embeddings) -----
        dpr = torch.linspace(0, axial_drop_path, steps=axial_blocks).tolist()
        self.axial = nn.ModuleList([
            AxialBlock(d_model=d_tok, n_heads=axial_heads, pdrop=pdrop, drop_path=dpr[i])
            for i in range(axial_blocks)
        ])

        # ----- TCN mixers (order-aware, no pos-emb needed) -----
        self.tcn = TCNMix1D(d_model=d_tok, dilations=tcn_dilations, pdrop=pdrop)

        # ----- PMA pooling + head -----
        self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=4, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_tok, 2 * d_tok), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_ATT),
        )

    def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,k,k)

    def _diag_chan(self, B: int, k: int, device, dtype) -> torch.Tensor:
        eye = torch.eye(k, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        return eye.expand(B, -1, -1, -1)  # (B,1,k,k)

    def _encode_cnn_multiscale(self, x4: torch.Tensor) -> torch.Tensor:
        stem_dtype = _module_param_dtype(self.encoder)
        s0 = x4
        s1 = F.avg_pool2d(x4, 2, ceil_mode=True)
        s2 = F.avg_pool2d(x4, 4, ceil_mode=True)

        z0 = self.encoder(s0.to(stem_dtype))
        z1 = self.encoder(s1.to(stem_dtype))
        z2 = self.encoder(s2.to(stem_dtype))

        v0 = _gpool_twice(z0)
        v1 = _gpool_twice(z1)
        v2 = _gpool_twice(z2)
        return torch.cat([v0, v1, v2], dim=-1)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        B, L, H, k, k2 = attn.shape
        assert k == k2, "attention maps must be square (k x k)"

        T = L * H
        maps = attn.reshape(B * T, 1, k, k)

        # [map, coords, diag]
        coords = self._coord(B * T, k, maps.device, maps.dtype)    # (B*T,2,k,k)
        diagc  = self._diag_chan(B * T, k, maps.device, maps.dtype) # (B*T,1,k,k)
        x4 = torch.cat([maps, coords, diagc], dim=1)                # (B*T,4,k,k)

        # per-map encoder
        cnn_vec = self._encode_cnn_multiscale(x4)                   # (B*T, 3*2*c_last)

        # stats (fp32)
        with no_amp_fp32(True):
            stats13 = spectral_graph_stats_13(maps.to(torch.float32))
            stats6  = _moment_stats_6(maps.to(torch.float32))
            ent4    = _attn_entropy_extras(maps.to(torch.float32))
        stats = torch.cat([stats13, stats6, ent4], dim=-1).to(cnn_vec.dtype)

        per_map = torch.cat([cnn_vec, stats], dim=-1)               # (B*T, cnn+23)
        feats = self.proj(per_map)                                  # (B*T, d_tok)

        # tokens over (L,H), NO positional embeddings
        tok = feats.reshape(B, L, H, self.d_tok)

        # axial transformer (content-only interactions)
        for blk in self.axial:
            tok = blk(tok)

        # order-aware mixers along L and H (no embeddings)
        tok = self.tcn(tok)

        # PMA pooling
        pma_in = tok.reshape(B, L * H, self.d_tok)
        pooled = self.pma(pma_in)
        out = self.out(pooled.flatten(1))
        return out


#*****************************************************************************************************************
# #gemini
# class AttnFeatureExtractorLite_D4_gemini(nn.Module):
#     """
#     D4: A more powerful hierarchical attention feature extractor.

#     Key Improvements over D3:
#     1.  **Richer Per-Map Features**: In addition to the CNN and spectral/graph stats,
#         this version computes explicit row-wise and column-wise statistics
#         (mean, std, entropy) to better capture token-level attention patterns.
#     2.  **Grid Transformer**: Replaces the axial convolutions with a full Transformer
#         encoder. This allows for global, long-range modeling of interactions between
#         attention maps across all layers and heads.
#     3.  **CLS Token Pooling**: Uses a learnable `[CLS]` token for global pooling,
#         a proven and powerful technique for sequence classification tasks, instead
#         of PMA. This token aggregates information from the entire (L,H) grid.
#     """
#     def __init__(
#         self,
#         D_ATT: int = 512,
#         d_grid: int = 192,
#         cnn_channels: tuple = (32, 64, 128),
#         grid_transformer_layers: int = 3,
#         grid_transformer_heads: int = 6,
#         K: int = 4, # Unused, kept for signature consistency if needed
#         pdrop: float = 0.1,
#         max_layers: int = 128,
#         max_heads: int = 256,
#     ):
#         super().__init__()
#         self.d_grid = d_grid
#         assert d_grid % grid_transformer_heads == 0, "d_grid must be divisible by heads"

#         # 1. Per-Map CNN Feature Extractor
#         self.cnn_stem = nn.Sequential(
#             nn.Conv2d(3, cnn_channels[0], 3, 1, 1, bias=False),
#             nn.GroupNorm(_num_groups(cnn_channels[0]), cnn_channels[0]),
#             nn.GELU()
#         )
#         self.cnn_body = nn.Sequential(
#             ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
#             ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2),
#         )
#         self.se = SEBlock(cnn_channels[-1])
#         cnn_out_dim = cnn_channels[-1] * 2  # GAP || GMP
#         num_spectral_stats = 13
#         num_rowcol_stats = 6 # mean/std/entropy for rows and cols
#         self.proj_per_map = nn.Linear(cnn_out_dim + num_spectral_stats + num_rowcol_stats, d_grid)

#         # 2. Grid Interaction & Pooling
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_grid))
#         self.layer_emb = nn.Embedding(max_layers, d_grid)
#         self.head_emb  = nn.Embedding(max_heads,  d_grid)
        
#         # Using the existing SAB as a standard Transformer encoder block
#         self.grid_transformer = SAB(
#             d_model=d_grid,
#             n_heads=grid_transformer_heads,
#             pdrop=pdrop,
#             num_layers=grid_transformer_layers
#         )

#         # 3. Final Output Head
#         self.out = nn.Sequential(
#             nn.Linear(d_grid, d_grid * 2), nn.GELU(), nn.LayerNorm(d_grid * 2),
#             nn.Dropout(pdrop),
#             nn.Linear(d_grid * 2, D_ATT)
#         )
#         self._init_weights()

#     def _init_weights(self):
#         nn.init.normal_(self.cls_token, std=0.02)
#         nn.init.normal_(self.layer_emb.weight, std=0.02)
#         nn.init.normal_(self.head_emb.weight, std=0.02)

#     def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
#         ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
#         xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
#         yy, xx = torch.meshgrid(ys, xs, indexing="ij")
#         return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
    
#     @torch.no_grad()
#     def _row_col_stats(self, A: torch.Tensor) -> torch.Tensor:
#         A2 = A.squeeze(1).clamp_min(0) # (B, k, k)
        
#         # Row stats (how a query attends to keys)
#         rsum = A2.sum(dim=-1)
#         rp = rsum / (rsum.sum(dim=-1, keepdim=True) + 1e-8)
#         rent = (-(rp.clamp_min(1e-8) * rp.clamp_min(1e-8).log()).sum(dim=-1)).unsqueeze(-1)
#         rmean = rsum.mean(dim=-1).unsqueeze(-1)
#         rstd = rsum.std(dim=-1, unbiased=False).unsqueeze(-1)

#         # Column stats (how a key is attended by queries)
#         csum = A2.sum(dim=-2)
#         cp = csum / (csum.sum(dim=-1, keepdim=True) + 1e-8)
#         cent = (-(cp.clamp_min(1e-8) * cp.clamp_min(1e-8).log()).sum(dim=-1)).unsqueeze(-1)
#         cmean = csum.mean(dim=-1).unsqueeze(-1)
#         cstd = csum.std(dim=-1, unbiased=False).unsqueeze(-1)
        
#         return torch.cat([rmean, rstd, rent, cmean, cstd, cent], dim=-1)


#     def forward(self, attn: torch.Tensor) -> torch.Tensor:
#         B, L, H, k, _ = attn.shape
#         T = L * H
#         device, dtype = attn.device, attn.dtype

#         maps = attn.reshape(B * T, 1, k, k)
#         coords = self._coord(B * T, k, device, maps.dtype)
#         x_maps = torch.cat([maps, coords], dim=1)

#         # === 1. Per-Map Feature Extraction ===
#         stem_dtype = _safe_dtype_param(self.cnn_stem)
#         z = self.cnn_stem(x_maps.to(stem_dtype))
#         z = self.cnn_body(z)
#         z = self.se(z)
#         gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
#         gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
#         cnn_vec = torch.cat([gavg, gmax], dim=-1)

#         with no_amp_fp32(True):
#             maps_fp32 = maps.to(torch.float32)
#             spectral_vec = spectral_graph_stats_13(maps_fp32)
#             rowcol_vec = self._row_col_stats(maps_fp32)

#         per_map_vec = torch.cat([
#             cnn_vec,
#             spectral_vec.to(cnn_vec.dtype),
#             rowcol_vec.to(cnn_vec.dtype)
#         ], dim=-1)
        
#         grid_toks = self.proj_per_map(per_map_vec).view(B, T, self.d_grid)

#         # === 2. Grid Transformer Processing ===
#         # Add layer and head embeddings
#         l_idx = torch.arange(L, device=device).repeat_interleave(H)
#         h_idx = torch.arange(H, device=device).repeat(L)
#         pos_emb = self.layer_emb(l_idx) + self.head_emb(h_idx)
#         grid_toks = grid_toks + pos_emb.unsqueeze(0)

#         # Prepend CLS token
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         transformer_in = torch.cat((cls_tokens, grid_toks), dim=1)

#         # Process with transformer
#         transformer_out = self.grid_transformer(transformer_in)

#         # === 3. Pooling and Final Projection ===
#         cls_output = transformer_out[:, 0] # Select the [CLS] token's final state
#         return self.out(cls_output)








#**************************************************************************************************************************
#**************************************************************************************************************************
#**************************************************************************************************************************
#**************************************************************************************************************************
#**************************************************************************************************************************
#**************************************************************************************************************************
#**************************************************************************************************************************
#**************************************************************************************************************************
#D3_V2
import math
from typing import Optional, Sequence, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# === Helper registry for ablation and flexible selection
# -------------------------------------------------------

ATOMIC_GROUP_DIMS: Dict[str, int] = {
    "spec_bands5": 5,   # [sent, Pl, Pm, Ph, Pv]
    "rowcol4":     4,   # [rvar, cvar, rent, cent]
    "structure3":  3,   # [diag_ratio, band_w1, band_w2]
    "lap1":        1,   # [lap_trace]
    "moments6":    6,   # [mu_y, mu_x, var_y, var_x, cov, anisotropy]
    "entropy4":    4,   # [H_map_norm, H_row_norm, H_col_norm, MI_norm]
}

ALIASES: Dict[str, Sequence[str]] = {
    "spec13": ("spec_bands5", "rowcol4", "structure3", "lap1"),
    "all":    tuple(ATOMIC_GROUP_DIMS.keys()),
}

def _resolve_groups(groups: Sequence[str]) -> List[str]:
    out: List[str] = []
    for g in groups:
        if g in ALIASES:
            out.extend(ALIASES[g])
        else:
            out.append(g)
    seen, uniq = set(), []
    for g in out:
        if g not in seen:
            if g not in ATOMIC_GROUP_DIMS:
                raise ValueError(f"Unknown stat group: {g}")
            uniq.append(g)
            seen.add(g)
    return uniq

def _groups_dim(groups: Sequence[str]) -> int:
    return sum(ATOMIC_GROUP_DIMS[g] for g in _resolve_groups(groups))


# -------------------------------------------------------
# === Core statistics (moment, entropy, spectral, etc.)
# -------------------------------------------------------

@torch.no_grad()
def _moment_stats_6(A: torch.Tensor) -> torch.Tensor:
    B, _, k, _ = A.shape
    A2 = A.squeeze(1).clamp_min_(0)
    s = A2.sum(dim=(-2, -1), keepdim=True).clamp_min_(1e-8)

    ys = torch.linspace(0.0, 1.0, steps=k, device=A.device, dtype=A.dtype).view(1, k, 1)
    xs = torch.linspace(0.0, 1.0, steps=k, device=A.device, dtype=A.dtype).view(1, 1, k)
    mu_y = (A2 * ys).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
    mu_x = (A2 * xs).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)

    dy = ys - mu_y.view(B, 1, 1)
    dx = xs - mu_x.view(B, 1, 1)
    var_y = (A2 * (dy ** 2)).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
    var_x = (A2 * (dx ** 2)).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)
    cov   = (A2 * (dy * dx)).sum(dim=(-2, -1)) / s.squeeze(-1).squeeze(-1)

    aniso = ((var_y - var_x).abs()) / (var_y + var_x + 1e-8)
    return torch.stack([mu_y, mu_x, var_y, var_x, cov, aniso], dim=-1)


@torch.no_grad()
def _attn_entropy_extras(A: torch.Tensor) -> torch.Tensor:
    B, _, k, _ = A.shape
    A = A.clamp_min(1e-12).squeeze(1)
    Z = A.sum(dim=(-2, -1), keepdim=True)
    P = (A / Z).clamp_min(1e-12)

    H_map = -(P * P.log()).sum(dim=(-2, -1))
    H_map_norm = H_map / math.log(k * k)

    p_row = P.sum(dim=-1)
    p_col = P.sum(dim=-2)
    H_row = -(p_row * p_row.clamp_min(1e-12).log()).sum(dim=-1)
    H_col = -(p_col * p_col.clamp_min(1e-12).log()).sum(dim=-1)
    H_row_norm = H_row / math.log(k)
    H_col_norm = H_col / math.log(k)

    I_xy = (H_row + H_col - H_map).clamp_min(0.0)
    I_xy_norm = I_xy / math.log(k)
    return torch.stack([H_map_norm, H_row_norm, H_col_norm, I_xy_norm], dim=-1)


@torch.no_grad()
def compute_attn_stats(
    A: torch.Tensor,
    groups: Sequence[str] = ("spec13",),
    *,
    spec_radii: Tuple[float, float, float] = (0.15, 0.35, 0.60),
    band_widths: Tuple[Optional[int], Optional[int]] = (None, None),
) -> torch.Tensor:
    G = _resolve_groups(groups)
    B, _, k, _ = A.shape
    A2 = A.squeeze(1)
    chunks: List[torch.Tensor] = []

    need_rowcol = any(g in G for g in ("rowcol4", "structure3", "lap1"))
    if need_rowcol:
        rows = A2.clamp_min(0)
        cols = rows.transpose(-1, -2)
        rsum = rows.sum(dim=-1)
        csum = cols.sum(dim=-1)
        total = A2.abs().sum(dim=(-1, -2), keepdim=False).unsqueeze(-1) + 1e-6
        diag  = torch.diagonal(A2, dim1=-2, dim2=-1).abs().sum(dim=-1, keepdim=True)

    if "spec_bands5" in G:
        Xf = torch.fft.rfft2(A2.to(torch.float32), dim=(-2, -1))
        P  = (Xf.real**2 + Xf.imag**2) + 1e-12
        Psum = P.sum(dim=(-2, -1))
        Pn = P / (Psum.view(B, 1, 1) + 1e-12)
        fy = torch.linspace(-0.5, 0.5, steps=k, device=A.device)
        fx = torch.linspace(0.0,  0.5, steps=(k // 2) + 1, device=A.device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        rad = torch.sqrt(yy**2 + xx**2)
        max_r = rad.max().clamp_min(1e-6)
        r1, r2, r3 = (spec_radii[0]*max_r, spec_radii[1]*max_r, spec_radii[2]*max_r)

        def band(mask):
            m = mask.to(P.dtype).unsqueeze(0)
            e = (P * m).sum(dim=(-2, -1))
            return (e / (Psum + 1e-12)).unsqueeze(-1)

        Pl = band(rad <= r1)
        Pm = band((rad > r1) & (rad <= r2))
        Ph = band((rad > r2) & (rad <= r3))
        Pv = band(rad > r3)
        sent = (-(Pn * Pn.log()).sum(dim=(-2, -1))).unsqueeze(-1)
        chunks.append(torch.cat([sent, Pl, Pm, Ph, Pv], dim=-1).to(A.dtype))

    if "rowcol4" in G:
        def _entropy(x, dim=-1):
            p = (x / (x.sum(dim=dim, keepdim=True) + 1e-8)).clamp_min(1e-8)
            return (-(p * p.log()).sum(dim=dim, keepdim=True))
        rvar = rsum.var(dim=-1, unbiased=False).unsqueeze(-1)
        cvar = csum.var(dim=-1, unbiased=False).unsqueeze(-1)
        rent = _entropy(rows, dim=-1).mean(dim=-2)
        cent = _entropy(cols, dim=-1).mean(dim=-2)
        chunks.append(torch.cat([rvar, cvar, rent, cent], dim=-1).to(A.dtype))

    if "structure3" in G:
        def band_energy(width: int):
            mask2d = torch.zeros(k, k, device=A2.device, dtype=A2.dtype)
            for d in range(-width, width + 1):
                n = k - abs(d)
                if n > 0:
                    mask2d += torch.diag(torch.ones(n, device=A2.device, dtype=A2.dtype), diagonal=d)
            m = mask2d.clamp_max(1).unsqueeze(0)
            e = (A2.abs() * m).sum(dim=(-1, -2))
            return (e / total.squeeze(-1)).unsqueeze(-1)

        w1 = (k // 32) if band_widths[0] is None else int(band_widths[0])
        w2 = (k // 16) if band_widths[1] is None else int(band_widths[1])
        diag_ratio = (diag / total)
        chunks.append(torch.cat([diag_ratio, band_energy(w1), band_energy(w2)], dim=-1).to(A.dtype))

    if "lap1" in G:
        D_trace  = rsum.sum(dim=-1, keepdim=True)
        A_trace  = A2.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        lap_trace = D_trace - A_trace
        chunks.append(lap_trace.to(A.dtype))

    if "moments6" in G:
        chunks.append(_moment_stats_6(A).to(A.dtype))

    if "entropy4" in G:
        chunks.append(_attn_entropy_extras(A).to(A.dtype))

    if not chunks:
        raise ValueError("No stat groups selected.")
    return torch.cat(chunks, dim=-1)


@torch.no_grad()
def spectral_graph_stats_13(A: torch.Tensor) -> torch.Tensor:
    """Legacy exact 13-D compatibility"""
    return compute_attn_stats(A, groups=("spec13",))


# -------------------------------------------------------
# === AttnFeatureExtractorLite_D3 (CNN optional)
# -------------------------------------------------------

class AttnFeatureExtractorLite_D3(nn.Module):
    def __init__(
        self,
        D_ATT: int = 512,
        d_grid: int = 128,
        cnn_channels: tuple = (32, 64, 128),
        grid_conv_layers: int = 2,
        K: int = 4,
        pdrop: float = 0.10,
        max_layers: int = 128,
        max_heads: int = 256,
        feature_mode: str = "cnn",                   # "cnn" | "spectral" | "both"
        stats_groups: Sequence[str] = ("spec13",),
        spec_radii: Tuple[float, float, float] = (0.15, 0.35, 0.60),
        band_widths: Tuple[Optional[int], Optional[int]] = (None, None),
        use_spectral: Optional[bool] = None,
    ):
        super().__init__()

        if use_spectral is not None:
            feature_mode = "both" if use_spectral else "cnn"
        if feature_mode not in {"cnn", "spectral", "both"}:
            raise ValueError(f"Invalid feature_mode: {feature_mode}")

        self.feature_mode = feature_mode
        self.stats_groups = tuple(stats_groups)
        self.spec_radii   = spec_radii
        self.band_widths  = band_widths
        self.d_grid       = d_grid

        self.has_cnn   = feature_mode in {"cnn", "both"}
        self.has_stats = feature_mode in {"spectral", "both"}

        # Conditional CNN creation
        self.cnn_stem = self.cnn_body = self.se = None
        cnn_out_dim = 0
        if self.has_cnn:
            in_c = 3
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(in_c, cnn_channels[0], 3, 1, 1, bias=False),
                nn.GroupNorm(max(1, cnn_channels[0] // 8), cnn_channels[0]),
                nn.GELU()
            )
            self.cnn_body = nn.Sequential(
                ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
                ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2),
            )
            self.se = SEBlock(cnn_channels[-1])
            cnn_out_dim = cnn_channels[-1] * 2

        stats_dim = _groups_dim(self.stats_groups) if self.has_stats else 0
        in_dim = cnn_out_dim + stats_dim
        if in_dim <= 0:
            raise ValueError("No input features selected")

        self.proj_per_map = nn.Linear(in_dim, d_grid)
        self.layer_emb = nn.Embedding(max_layers, d_grid)
        self.head_emb  = nn.Embedding(max_heads, d_grid)
        nn.init.normal_(self.layer_emb.weight, std=0.02)
        nn.init.normal_(self.head_emb.weight,  std=0.02)

        axial = []
        for _ in range(grid_conv_layers):
            axial += [
                nn.Conv2d(d_grid, d_grid, (1,3), padding=(0,1), groups=d_grid, bias=False),
                nn.GELU(),
                nn.Conv2d(d_grid, d_grid, (3,1), padding=(1,0), groups=d_grid, bias=False),
                nn.GELU(),
                nn.Conv2d(d_grid, d_grid, 1, bias=False),
                nn.GroupNorm(max(1, d_grid // 8), d_grid),
            ]
        self.grid_processor = nn.Sequential(*axial)

        self.pma = PMA(d_model=d_grid, num_seeds=K, n_heads=4, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_grid, 2 * d_grid), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2 * d_grid, D_ATT)
        )

    def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        B, L, H, k, _ = attn.shape
        T = L * H
        device = attn.device
        maps = attn.reshape(B * T, 1, k, k)
        per_chunks: List[torch.Tensor] = []

        if self.has_cnn:
            coords = self._coord(B * T, k, device=maps.device, dtype=maps.dtype)
            x_maps = torch.cat([maps, coords], dim=1)
            z = self.cnn_stem(x_maps)
            z = self.cnn_body(z)
            z = self.se(z)
            gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
            gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
            per_chunks.append(torch.cat([gavg, gmax], dim=-1))

        if self.has_stats:
            stats_vec = compute_attn_stats(
                maps.to(torch.float32),
                groups=self.stats_groups,
                spec_radii=self.spec_radii,
                band_widths=self.band_widths,
            )
            per_chunks.append(stats_vec.to(per_chunks[0].dtype if per_chunks else maps.dtype))

        per_map = per_chunks[0] if len(per_chunks) == 1 else torch.cat(per_chunks, dim=-1)
        feats = self.proj_per_map(per_map)

        tok = feats.view(B, L, H, self.d_grid)
        tok = tok + self.layer_emb(torch.arange(L, device=device)).view(1, L, 1, -1) \
                  + self.head_emb(torch.arange(H, device=device)).view(1, 1, H, -1)
        grid = tok.permute(0, 3, 1, 2).contiguous()
        grid = grid + self.grid_processor(grid)

        pma_in = grid.flatten(2).transpose(1, 2)
        pooled = self.pma(pma_in)
        return self.out(pooled.flatten(1))




# D4 GEMINI — uses the SAME stats engine as D3 (compute_attn_stats)
# Assumes you already have:
#   - compute_attn_stats, _groups_dim
#   - SAB (Transformer block), PMA (unused here), ResNetBlock, SEBlock
#   - _num_groups, _module_param_dtype, no_amp_fp32
# If your helper names differ, adjust imports accordingly.

class AttnFeatureExtractorLite_D4_gemini(nn.Module):
    """
    D4: Hierarchical attention feature extractor with CLS pooling and grid Transformer.

    - CNN branch on [map, coords] is optional and constructed ONLY if enabled.
    - Stats branch is IDENTICAL to D3 (via compute_attn_stats with the same groups/aliases).
    - Global modeling via Transformer encoder (SAB) over the (L,H) grid + [CLS] token pooling.
    """
    def __init__(
        self,
        D_ATT: int = 512,
        d_grid: int = 192,
        cnn_channels: tuple = (32, 64, 128),
        grid_transformer_layers: int = 3,
        grid_transformer_heads: int = 6,
        K: int = 4,                         # kept for signature parity; unused here
        pdrop: float = 0.10,
        max_layers: int = 128,
        max_heads: int = 256,

        # SAME interface as D3:
        feature_mode: str = "both",         # {"cnn","spectral","both"}
        stats_groups: Sequence[str] = ("spec13",),
        spec_radii: Tuple[float, float, float] = (0.15, 0.35, 0.60),
        band_widths: Tuple[Optional[int], Optional[int]] = (None, None),
        # legacy toggle: True→"both", False→"cnn"
        use_spectral: Optional[bool] = None,
    ):
        super().__init__()
        assert d_grid % grid_transformer_heads == 0, "d_grid must be divisible by heads"

        # ---- resolve mode (legacy flag supported) ----
        if use_spectral is not None:
            feature_mode = "both" if use_spectral else "cnn"
        if feature_mode not in {"cnn", "spectral", "both"}:
            raise ValueError(f"feature_mode must be 'cnn','spectral','both'; got {feature_mode!r}")

        self.d_grid       = d_grid
        self.feature_mode = feature_mode
        self.has_cnn      = feature_mode in {"cnn", "both"}
        self.has_stats    = feature_mode in {"spectral", "both"}

        self.stats_groups = tuple(stats_groups)
        self.spec_radii   = spec_radii
        self.band_widths  = band_widths

        # ---- 1) Per-map branches (CNN conditional) ----
        self.cnn_stem = self.cnn_body = self.se = None
        cnn_out_dim = 0
        if self.has_cnn:
            in_c = 3  # [map, y, x]
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(in_c, cnn_channels[0], 3, 1, 1, bias=False),
                nn.GroupNorm(_num_groups(cnn_channels[0]), cnn_channels[0]),
                nn.GELU()
            )
            self.cnn_body = nn.Sequential(
                ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
                ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2),
            )
            self.se = SEBlock(cnn_channels[-1])
            cnn_out_dim = cnn_channels[-1] * 2  # GAP || GMP

        stats_dim = _groups_dim(self.stats_groups) if self.has_stats else 0
        in_dim = cnn_out_dim + stats_dim
        if in_dim <= 0:
            raise ValueError("No inputs selected: enable CNN and/or choose non-empty stats_groups.")

        self.proj_per_map = nn.Linear(in_dim, d_grid)

        # ---- 2) Grid embeddings & Transformer ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_grid))
        self.layer_emb = nn.Embedding(max_layers, d_grid)
        self.head_emb  = nn.Embedding(max_heads,  d_grid)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.layer_emb.weight, std=0.02)
        nn.init.normal_(self.head_emb.weight,  std=0.02)

        self.grid_transformer = SAB(
            d_model=d_grid,
            n_heads=grid_transformer_heads,
            pdrop=pdrop,
            num_layers=grid_transformer_layers,
        )

        # ---- 3) Output head on [CLS] ----
        self.out = nn.Sequential(
            nn.Linear(d_grid, 2 * d_grid),
            nn.GELU(),
            nn.LayerNorm(2 * d_grid),
            nn.Dropout(pdrop),
            nn.Linear(2 * d_grid, D_ATT),
        )

    def _coord(self, B: int, k: int, device, dtype) -> torch.Tensor:
        ys = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, steps=k, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: (B, L, H, k, k)
        """
        B, L, H, k, _ = attn.shape
        T = L * H
        device = attn.device

        maps = attn.reshape(B * T, 1, k, k)
        per_chunks: List[torch.Tensor] = []

        # ---- CNN branch (only if constructed) ----
        if self.has_cnn:
            coords  = self._coord(B * T, k, device=maps.device, dtype=maps.dtype)
            x_maps  = torch.cat([maps, coords], dim=1)
            stem_ty = _module_param_dtype(self.cnn_stem)
            z = self.cnn_stem(x_maps.to(stem_ty))
            z = self.cnn_body(z)
            z = self.se(z)
            gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
            gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
            per_chunks.append(torch.cat([gavg, gmax], dim=-1))

        # ---- Stats branch (IDENTICAL to D3) ----
        if self.has_stats:
            with no_amp_fp32(True):
                stats_vec = compute_attn_stats(
                    maps.to(torch.float32),
                    groups=self.stats_groups,       # <- same groups/aliases as D3
                    spec_radii=self.spec_radii,
                    band_widths=self.band_widths,
                )
            per_chunks.append(stats_vec)

        # ---- Concatenate enabled features & project ----
        per_map_vec = per_chunks[0] if len(per_chunks) == 1 else torch.cat(per_chunks, dim=-1)
        per_map_vec = per_map_vec.to(self.proj_per_map.weight.dtype)
        grid_toks = self.proj_per_map(per_map_vec).view(B, T, self.d_grid)

        # ---- Add (L,H) positional embeddings ----
        l_idx = torch.arange(L, device=device).repeat_interleave(H)   # (T,)
        h_idx = torch.arange(H, device=device).repeat(L)              # (T,)
        pos_emb = (self.layer_emb(l_idx) + self.head_emb(h_idx)).unsqueeze(0)  # (1,T,d)
        grid_toks = grid_toks + pos_emb

        # ---- Prepend [CLS] and run transformer ----
        cls_tokens = self.cls_token.expand(B, 1, self.d_grid)         # (B,1,d)
        transformer_in = torch.cat([cls_tokens, grid_toks], dim=1)    # (B,1+T,d)
        transformer_out = self.grid_transformer(transformer_in)       # (B,1+T,d)

        # ---- Pool via [CLS] ----
        cls_out = transformer_out[:, 0]                                # (B,d)
        return self.out(cls_out)
































# # ======================================================================================
# # SECTION 4: OTHER FEATURE EXTRACTORS (mask-free)
# # ======================================================================================

# class HiddenFeatureExtractorLite(nn.Module):
#     """
#     Features from hidden-state sequences via gated dilated 1D convs + SAB + PMA.
#     """
#     def __init__(
#         self, D_model: int, D_HID: int = 512, d_tok: int = 192, k_hid: int = 192,
#         groups: int = 8, K: int = 3, sab_layers: int = 1, sab_heads: int = 4, pdrop: float = 0.10,
#     ):
#         super().__init__()
#         self.k_hid, self.d_tok = int(k_hid), int(d_tok)
#         self.norm, self.proj = nn.LayerNorm(D_model), nn.Linear(D_model, d_tok)

#         g = _num_groups(d_tok)  # ensure divisibility
#         def dw_block(dil):
#             return nn.Sequential(
#                 nn.Conv1d(d_tok, d_tok, 5, padding=2*dil, dilation=dil, groups=g),
#                 nn.GroupNorm(_num_groups(d_tok), d_tok), nn.GELU(),
#             )
#         self.dw1, self.dw2, self.dw3 = dw_block(1), dw_block(2), dw_block(4)
#         self.gate = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]), requires_grad=True)
#         self.se1d = nn.Sequential(
#             nn.Conv1d(d_tok, max(8, d_tok//8), 1), nn.GELU(),
#             nn.Conv1d(max(8, d_tok//8), d_tok, 1), nn.Sigmoid()
#         )
#         self.drop = nn.Dropout(pdrop)
#         self.pos = nn.Parameter(torch.randn(1, self.k_hid, d_tok) / math.sqrt(d_tok))
#         self.sab = SAB(d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=sab_layers)
#         self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=sab_heads, pdrop=pdrop)
#         self.out = nn.Sequential(
#             nn.Linear(K * d_tok + 10, 2 * d_tok), nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(2 * d_tok, D_HID)
#         )

#     @torch.no_grad()
#     def _dyn_fft_stats(self, seq_mean: torch.Tensor) -> torch.Tensor:
#         B, k = seq_mean.shape
#         mean, var = seq_mean.mean(dim=-1, keepdim=True), seq_mean.var(dim=-1, unbiased=False, keepdim=True)
#         if k >= 2:
#             d = seq_mean[:, 1:] - seq_mean[:, :-1]
#             tv = d.abs().sum(dim=-1, keepdim=True)
#             p90d = percentile(d.abs(), 0.90, dim=-1, keepdim=True)
#         else:
#             tv, p90d = torch.zeros(B, 1, device=seq_mean.device), torch.zeros(B, 1, device=seq_mean.device)
#         w = max(1, k // 3)
#         el = (seq_mean[:, :w].mean(dim=-1, keepdim=True) - seq_mean[:, -w:].mean(dim=-1, keepdim=True))
#         last = seq_mean[:, -1:].contiguous()
#         P = (torch.fft.rfft(seq_mean, dim=-1).abs()**2)
#         M = P.shape[-1]; q1, q2, q3 = M//4, M//2, 3*M//4
#         Pl, Pm = P[:, :q1].mean(dim=-1, keepdim=True), P[:, q1:q2].mean(dim=-1, keepdim=True)
#         Ph, Pv = P[:, q2:q3].mean(dim=-1, keepdim=True), P[:, q3:].mean(dim=-1, keepdim=True)
#         return torch.cat([mean, var, tv, p90d, el, last, Pl, Pm, Ph, Pv], dim=-1)

#     def forward(self, last_hidden: torch.Tensor, mask_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
#         B, S, _ = last_hidden.shape
#         x = self.proj(self.norm(last_hidden)).permute(0, 2, 1).contiguous()

#         # Mask-free downsampling (fp32 for pooling only)
#         with no_amp_fp32(True):
#             x_ds = F.adaptive_avg_pool1d(x.to(torch.float32), self.k_hid)
#         x_ds = x_ds.to(_module_param_dtype(self.dw1))  # match conv dtype

#         y1, y2, y3 = self.dw1(x_ds), self.dw2(x_ds), self.dw3(x_ds)
#         g = torch.softmax(self.gate, dim=0)
#         mix = g[0]*y1 + g[1]*y2 + g[2]*y3
#         z = self.drop(mix * self.se1d(mix) + x_ds)

#         tok = z.permute(0, 2, 1).contiguous() + self.pos.to(z.dtype)
#         tok = self.sab(tok)
#         pooled = self.pma(tok).flatten(1)

#         with no_amp_fp32(True):
#             stats = self._dyn_fft_stats(tok.to(torch.float32).mean(dim=-1))
#         vec = torch.cat([pooled, stats.to(pooled.dtype)], dim=-1)
#         return self.out(vec)

#no stat
class HiddenFeatureExtractorLite(nn.Module):
    """
    Features from hidden-state sequences via gated dilated 1D convs + SAB + PMA.
    """
    def __init__(
        self, D_model: int, D_HID: int = 512, d_tok: int = 192, k_hid: int = 192,
        groups: int = 8, K: int = 3, sab_layers: int = 1, sab_heads: int = 4, pdrop: float = 0.10,
    ):
        super().__init__()
        self.k_hid, self.d_tok = int(k_hid), int(d_tok)
        self.norm, self.proj = nn.LayerNorm(D_model), nn.Linear(D_model, d_tok)

        g = _num_groups(d_tok)  # ensure divisibility
        def dw_block(dil):
            return nn.Sequential(
                nn.Conv1d(d_tok, d_tok, 5, padding=2*dil, dilation=dil, groups=g),
                nn.GroupNorm(_num_groups(d_tok), d_tok), nn.GELU(),
            )
        self.dw1, self.dw2, self.dw3 = dw_block(1), dw_block(2), dw_block(4)
        self.gate = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]), requires_grad=True)
        self.se1d = nn.Sequential(
            nn.Conv1d(d_tok, max(8, d_tok//8), 1), nn.GELU(),
            nn.Conv1d(max(8, d_tok//8), d_tok, 1), nn.Sigmoid()
        )
        self.drop = nn.Dropout(pdrop)
        self.pos = nn.Parameter(torch.randn(1, self.k_hid, d_tok) / math.sqrt(d_tok))
        self.sab = SAB(d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=sab_layers)
        self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=sab_heads, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_tok, 2 * d_tok), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_HID)
        )

    def forward(self, last_hidden: torch.Tensor, mask_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = last_hidden.shape
        x = self.proj(self.norm(last_hidden)).permute(0, 2, 1).contiguous()

        # Mask-free downsampling (fp32 for pooling only)
        with no_amp_fp32(True):
            x_ds = F.adaptive_avg_pool1d(x.to(torch.float32), self.k_hid)
        x_ds = x_ds.to(_module_param_dtype(self.dw1))  # match conv dtype

        y1, y2, y3 = self.dw1(x_ds), self.dw2(x_ds), self.dw3(x_ds)
        g = torch.softmax(self.gate, dim=0)
        mix = g[0]*y1 + g[1]*y2 + g[2]*y3
        z = self.drop(mix * self.se1d(mix) + x_ds)

        tok = z.permute(0, 2, 1).contiguous() + self.pos.to(z.dtype)
        tok = self.sab(tok)
        pooled = self.pma(tok).flatten(1)

        return self.out(pooled)

# ===============================================
# v5: Stability + SDPA + DropPath + richer stats
# ===============================================

# ---- DropPath (stochastic depth) ----
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

# ======================================================================================
# SECTION 2: SET TRANSFORMER PRIMITIVES (v5)
# - SDPA-based MHA for speed/stability
# - Optional DropPath on residuals
# ======================================================================================

class MultiHeadAttentionV5(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.pdrop = float(pdrop)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, Tq, D = Q.shape
        Tk = K.size(1)

        q = self.q(Q).view(B, Tq, self.h, self.dk).transpose(1, 2)  # [B,h,Tq,dk]
        k = self.k(K).view(B, Tk, self.h, self.dk).transpose(1, 2)  # [B,h,Tk,dk]
        v = self.v(V).view(B, Tk, self.h, self.dk).transpose(1, 2)  # [B,h,Tk,dk]

        # F.scaled_dot_product_attention handles scaling, masking, dropout (training only)
        with no_amp_fp32(True):
            out = F.scaled_dot_product_attention(
                q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
                attn_mask=None, dropout_p=self.pdrop if self.training else 0.0, is_causal=False
            ).to(q.dtype)

        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.o(out)

class MABV5(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, ff_mult: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.mha = MultiHeadAttentionV5(d_model, n_heads, pdrop)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        x = Q + self.dp1(self.mha(Q, K, K))
        x = self.ln1(x)
        x = x + self.dp2(self.ff(x))
        x = self.ln2(x)
        return x

class SABV5(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, num_layers: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([MABV5(d_model, n_heads, pdrop, drop_path=drop_path) for _ in range(num_layers)])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for mab in self.layers:
            X = mab(X, X)
        return X

class PMAV5(nn.Module):
    def __init__(self, d_model: int, num_seeds: int = 4, n_heads: int = 4, pdrop: float = 0.1, drop_path: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.randn(num_seeds, d_model) / math.sqrt(d_model))
        self.mab = MABV5(d_model, n_heads, pdrop, drop_path=drop_path)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        S = self.S.unsqueeze(0).expand(B, -1, -1)
        return self.mab(S, X)

# ======================================================================================
# SECTION 3: HiddenFeatureExtractorLiteV5
# - DW-PW blocks + SE + GLU
# - Multi-scale temporal pooling (k and k//2)
# - Richer fp32 stats: FFT bands + geometric trajectory
# - Optional mask branch (disabled by default)
# ======================================================================================

class DWBlock1d(nn.Module):
    """Depthwise 1D conv + SE + optional pointwise mixing with a tiny GLU gate."""
    def __init__(self, c: int, k: int = 5, dilation: int = 1, use_pw: bool = True):
        super().__init__()
        g = _num_groups(c)
        pad = (k // 2) * dilation
        self.dw = nn.Conv1d(c, c, k, padding=pad, dilation=dilation, groups=g, bias=False)
        self.gn = nn.GroupNorm(_num_groups(c), c)
        self.act = nn.GELU()
        mid = max(8, c // 8)
        self.se = nn.Sequential(nn.Conv1d(c, mid, 1), nn.GELU(), nn.Conv1d(mid, c, 1), nn.Sigmoid())
        self.use_pw = use_pw
        if use_pw:
            self.pw_in = nn.Conv1d(c, 2 * c, 1, bias=False)  # GLU
            self.pw_out = nn.Conv1d(c, c, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x)
        y = self.gn(y)
        y = self.act(y)
        y = y * self.se(y)
        if self.use_pw:
            # GLU
            a, b = self.pw_in(y).chunk(2, dim=1)
            y = F.glu(torch.cat([a, b], dim=1), dim=1)  # or: a * torch.sigmoid(b)
            y = self.pw_out(y)
        return x + y  # residual

class HiddenFeatureExtractorLite_V5(nn.Module):
    """
    v5 upgrades:
      • SDPA-based set transformer blocks (SAB/PMAv5) with DropPath
      • Multi-scale temporal pooling (k_hid and k_hid//2) before tokenization
      • DW-PW blocks with SE+GLU gating for stronger local mixing
      • Rich fp32 stats: FFT (low/mid/high/very-high) + geometric norms & curvature
      • Optional masked branch (disabled by default to preserve 'mask-free' behavior)

    Inputs:
      last_hidden: [B, S, D_model]
      mask_tokens (optional): [B, S] boolean/int; used only if use_mask_branch=True
    """
    def __init__(
        self,
        D_model: int,
        D_HID: int = 512,
        d_tok: int = 192,
        k_hid: int = 192,
        groups: int = 8,
        K: int = 3,
        sab_layers: int = 1,
        sab_heads: int = 4,
        pdrop: float = 0.10,
        drop_path: float = 0.05,
        use_mask_branch: bool = False,
        use_multi_scale: bool = True,
    ):
        super().__init__()
        self.k_hid = int(k_hid)
        self.d_tok = int(d_tok)
        self.use_mask_branch = bool(use_mask_branch)
        self.use_multi_scale = bool(use_multi_scale)

        # Token projection
        self.norm = nn.LayerNorm(D_model)
        self.proj = nn.Linear(D_model, d_tok)

        # Three dilations at base scale
        self.dw1 = DWBlock1d(d_tok, k=5, dilation=1, use_pw=True)
        self.dw2 = DWBlock1d(d_tok, k=5, dilation=2, use_pw=True)
        self.dw3 = DWBlock1d(d_tok, k=5, dilation=4, use_pw=True)
        self.gate = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]), requires_grad=True)

        self.drop = nn.Dropout(pdrop)

        # Positions per scale
        self.pos_main = nn.Parameter(torch.randn(1, self.k_hid, d_tok) / math.sqrt(d_tok))
        k_half = max(16, self.k_hid // 2)
        self.k_half = k_half
        if self.use_multi_scale:
            self.pos_half = nn.Parameter(torch.randn(1, k_half, d_tok) / math.sqrt(d_tok))

        # SAB/PMA with DropPath
        self.sab = SABV5(d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=sab_layers, drop_path=drop_path)
        self.pma = PMAV5(d_model=d_tok, num_seeds=K, n_heads=sab_heads, pdrop=pdrop, drop_path=drop_path)

        # If we ever enable mask branch, project/fuse its pooled vector
        if self.use_mask_branch:
            self.mask_sab = SABV5(d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=max(1, sab_layers//2), drop_path=drop_path)
            self.mask_pma = PMAV5(d_model=d_tok, num_seeds=max(1, K//2) or 1, n_heads=sab_heads, pdrop=pdrop, drop_path=drop_path)

        # Output head: [pooled_tokens || stats] -> D_HID
        # stats dims: FFT(4) + dyn (mean,var,tv,p90d,edge,last)(6) + geom (norm_mean,norm_std,cos_mean,cos_p10,cos_p90)(5) = 15
        self._n_stats = 15
        # If multi-scale: we concatenate pooled_main and pooled_half
        pooled_dim = (K * d_tok) * (2 if self.use_multi_scale else 1)
        if self.use_mask_branch:
            pooled_dim += (max(1, K//2) * d_tok)  # masked pooled vector

        self.out = nn.Sequential(
            nn.Linear(pooled_dim + self._n_stats, 2 * d_tok),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_HID),
        )

    @torch.no_grad()
    def _dyn_fft_stats(self, seq_mean: torch.Tensor) -> torch.Tensor:
        """
        seq_mean: [B, k] (mean across channels)
        Returns 10 dims: mean,var,tv,p90d,edge,last, Pl,Pm,Ph,Pv
        """
        B, k = seq_mean.shape
        mean = seq_mean.mean(dim=-1, keepdim=True)
        var  = seq_mean.var(dim=-1, unbiased=False, keepdim=True)
        if k >= 2:
            d = seq_mean[:, 1:] - seq_mean[:, :-1]
            tv = d.abs().sum(dim=-1, keepdim=True)
            p90d = percentile(d.abs(), 0.90, dim=-1, keepdim=True)
        else:
            tv  = torch.zeros(B, 1, device=seq_mean.device)
            p90d = torch.zeros(B, 1, device=seq_mean.device)
        w = max(1, k // 3)
        edge = (seq_mean[:, :w].mean(dim=-1, keepdim=True) - seq_mean[:, -w:].mean(dim=-1, keepdim=True))
        last = seq_mean[:, -1:].contiguous()

        P = (torch.fft.rfft(seq_mean, dim=-1).abs() ** 2)
        M = P.shape[-1]; q1, q2, q3 = M//4, M//2, 3*M//4
        Pl = P[:, :q1].mean(dim=-1, keepdim=True) if q1 > 0 else torch.zeros(B,1,device=seq_mean.device)
        Pm = P[:, q1:q2].mean(dim=-1, keepdim=True) if q2 > q1 else torch.zeros(B,1,device=seq_mean.device)
        Ph = P[:, q2:q3].mean(dim=-1, keepdim=True) if q3 > q2 else torch.zeros(B,1,device=seq_mean.device)
        Pv = P[:, q3:].mean(dim=-1, keepdim=True) if q3 < M   else torch.zeros(B,1,device=seq_mean.device)
        return torch.cat([mean, var, tv, p90d, edge, last, Pl, Pm, Ph, Pv], dim=-1)

    @torch.no_grad()
    def _geom_stats(self, tok: torch.Tensor) -> torch.Tensor:
        """
        tok: [B, k, d_tok]
        Returns 5 dims: norm_mean, norm_std, cos_mean, cos_p10, cos_p90
        """
        B, k, d = tok.shape
        norms = tok.norm(dim=-1)  # [B,k]
        n_mean = norms.mean(dim=-1, keepdim=True)
        n_std  = norms.std(dim=-1, unbiased=False, keepdim=True)

        if k >= 2:
            cos = F.cosine_similarity(tok[:, :-1, :], tok[:, 1:, :], dim=-1)  # [B,k-1]
            c_mean = cos.mean(dim=-1, keepdim=True)
            c_p10  = percentile(cos, 0.10, dim=-1, keepdim=True)
            c_p90  = percentile(cos, 0.90, dim=-1, keepdim=True)
        else:
            c_mean = torch.zeros(B,1,device=tok.device)
            c_p10  = torch.zeros(B,1,device=tok.device)
            c_p90  = torch.zeros(B,1,device=tok.device)

        return torch.cat([n_mean, n_std, c_mean, c_p10, c_p90], dim=-1)

    def _downsample_seq(self, x: torch.Tensor, k: int) -> torch.Tensor:
        # AMP-off for pooling, then cast back
        with no_amp_fp32(True):
            x_ds = F.adaptive_avg_pool1d(x.to(torch.float32), k)
        return x_ds.to(_module_param_dtype(self))

    def _tok_pipeline(self, x_ds: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_ds: [B, d_tok, k]
        y1, y2, y3 = self.dw1(x_ds), self.dw2(x_ds), self.dw3(x_ds)
        g = torch.softmax(self.gate, dim=0)
        mix = g[0]*y1 + g[1]*y2 + g[2]*y3
        z = self.drop(mix) + x_ds  # extra residual to stabilize early training
        tok = z.permute(0, 2, 1).contiguous() + pos.to(z.dtype)  # [B,k,d_tok]
        tok = self.sab(tok)
        pooled = self.pma(tok).flatten(1)
        return tok, pooled

    def forward(self, last_hidden: torch.Tensor, mask_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        last_hidden: [B, S, D_model]
        mask_tokens: [B, S] (ignored unless use_mask_branch=True and provided)
        """
        B, S, _ = last_hidden.shape
        x = self.proj(self.norm(last_hidden)).permute(0, 2, 1).contiguous()  # [B,d_tok,S]

        # ---- main scale ----
        x_main = self._downsample_seq(x, self.k_hid)               # [B,d_tok,k]
        tok_main, pooled_main = self._tok_pipeline(x_main, self.pos_main)

        pooled_list = [pooled_main]

        # ---- half scale (optional) ----
        if self.use_multi_scale:
            x_half = self._downsample_seq(x, self.k_half)
            tok_half, pooled_half = self._tok_pipeline(x_half, self.pos_half)
            pooled_list.append(pooled_half)
            tok_for_stats = tok_main  # keep single source for stats to avoid overweight
        else:
            tok_for_stats = tok_main

        # ---- optional masked branch (defaults OFF to keep mask-free behavior) ----
        if self.use_mask_branch and (mask_tokens is not None):
            # crude masked average before downsampling to fixed k
            # (keeps things simple; for ragged exactness, use per-batch slicing pipeline)
            m = (mask_tokens > 0).float()[:, None, :]  # [B,1,S]
            eps = 1e-6
            x_masked = (x * m) / (m + eps)            # zero elsewhere, scale where mask>0
            x_mask_ds = self._downsample_seq(x_masked, max(8, self.k_hid // 2))
            # separate (lighter) SAB/PMA for mask branch
            z1, z2, z3 = self.dw1(x_mask_ds), self.dw2(x_mask_ds), self.dw3(x_mask_ds)
            g = torch.softmax(self.gate, dim=0)
            mix_m = g[0]*z1 + g[1]*z2 + g[2]*z3
            tok_m = (self.drop(mix_m) + x_mask_ds).permute(0,2,1).contiguous()
            tok_m = self.mask_sab(tok_m)
            pooled_mask = self.mask_pma(tok_m).flatten(1)
            pooled_list.append(pooled_mask)

        pooled = torch.cat(pooled_list, dim=-1)

        # ---- stats (fp32) ----
        with no_amp_fp32(True):
            seq_mean = tok_for_stats.to(torch.float32).mean(dim=-1)  # [B,k]
            fft_stats = self._dyn_fft_stats(seq_mean)                # [B,10]
            geom_stats = self._geom_stats(tok_for_stats.to(torch.float32))  # [B,5]
            stats = torch.cat([fft_stats, geom_stats], dim=-1)       # [B,15]

        vec = torch.cat([pooled, stats.to(pooled.dtype)], dim=-1)
        return self.out(vec)




# ======================================================================================
# SECTION 1: POSITIONAL ENCODING (Standard Helper)
# ======================================================================================

class PositionalEncoding(nn.Module):
    """
    Standard sine/cosine positional encoding.
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 30000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        
        # Use .float() for compatibility even in AMP
        pe[0, :, 0::2] = torch.sin(position * div_term).float()
        pe[0, :, 1::2] = torch.cos(position * div_term).float()
        
        # register_buffer makes it part of the model's state but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # x.size(1) is the seq_len
        # self.pe is [1, max_len, d_model], so we slice [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ======================================================================================
# SECTION 2: THE FEATURE EXTRACTOR
# ======================================================================================

class HiddenFeatureExtractorLite_V6(nn.Module):
    """
    Extracts features from an LLM's hidden states using a Transformer
    Encoder focused on the *answer tokens*.

    This model processes the last hidden states from an LLM. It focuses
    exclusively on the tokens corresponding to the generated *answer*,
    ignoring the prompt and padding. It uses a [CLS] token and a Transformer
    Encoder to build a representative feature vector.

    Args:
        d_llm (int): The hidden dimension of the input LLM (e.g., 4096).
        D_HID (int): The final output feature dimension for the extractor.
        d_model (int): The internal dimension of this classifier model (e.g., 256 or 512).
        n_heads (int): Number of attention heads for the Transformer.
        num_layers (int): Number of Transformer Encoder layers.
        d_ff (int): Dimension of the feed-forward network in the Transformer.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        d_llm: int,
        D_HID: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Input Projection
        # Project the large LLM hidden state to our model's dimension
        self.input_proj = nn.Linear(d_llm, d_model)
        
        # 2. [CLS] Token
        # This is a learnable parameter that will act as the sequence summary
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) / math.sqrt(d_model))
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=30000)
        
        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,  # Expects [B, S, D]
            norm_first=True,   # Pre-LayerNorm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) # Final norm
        )
        
        # 5. Feature Head
        # Takes the final [CLS] token representation and maps to D_HID
        self.feature_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, D_HID)
        )

    def forward(
        self,
        last_hidden: torch.Tensor,
        mask_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            last_hidden (torch.Tensor):
                The last hidden state from the LLM.
                Shape: [B, S, d_llm] (Batch, SeqLen, LLM_Dim)
            
            mask_tokens (torch.Tensor):
                A boolean or integer mask. 
                1s or True for *answer tokens*.
                0s or False for *prompt and padding tokens*.
                Shape: [B, S] (Batch, SeqLen)

        Returns:
            torch.Tensor: Feature vector. Shape: [B, D_HID]
        """
        B, S, _ = last_hidden.shape
        
        # 1. Project input to our working dimension
        # [B, S, d_llm] -> [B, S, d_model]
        x = self.input_proj(last_hidden)
        
        # 2. Prepend the [CLS] token
        # [1, 1, d_model] -> [B, 1, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # [B, S, d_model] -> [B, S+1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 3. Create the padding mask for the Transformer
        # The Transformer expects True for tokens to *ignore* (mask out).
        
        if mask_tokens is None:
            # If no mask provided, assume all tokens (except CLS) are valid
            padding_mask = torch.zeros(B, S + 1, dtype=torch.bool, device=x.device)
        else:
            # We want to *ignore* everything that is NOT an answer token.
            # True where answer_mask is 0 (i.e., at prompt/padding)
            ignore_mask = ~(mask_tokens.bool()) 
            
            # We must also add a mask for the [CLS] token.
            # The [CLS] token should *never* be ignored.
            # [B, 1] of all False
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            
            # [B, 1] + [B, S] -> [B, S+1]
            padding_mask = torch.cat([cls_mask, ignore_mask], dim=1)
        
        # 4. Add positional encoding
        # [B, S+1, d_model]
        x = self.pos_encoder(x)
        
        # 5. Pass through Transformer
        # The padding_mask ensures attention is only paid *to* and *from*
        # the [CLS] token and the valid answer tokens.
        transformer_out = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )
        
        # 6. Get the final [CLS] token representation
        # This vector has aggregated information from the answer sequence.
        # [B, S+1, d_model] -> [B, d_model]
        cls_embedding = transformer_out[:, 0]
        
        # 7. Project to final feature dimension
        # [B, d_model] -> [B, D_HID]
        features = self.feature_head(cls_embedding)
        
        return features

# #************************************************************************************************
# #V2
# class _LSEPool1d(nn.Module):
#     """
#     Learned log-sum-exp pooling over time.
#     Interpolates mean (alpha→0) and max (alpha→+∞), stable for signed inputs.
#     """
#     def __init__(self, init_alpha: float = 4.0):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, T]
#         B, C, T = x.shape
#         # clamp alpha for stability
#         alpha = torch.clamp(self.alpha, 0.0, 20.0)
#         if alpha < 1e-3:
#             return x.mean(dim=-1)  # [B, C]
#         # (1/alpha) * log(mean_t exp(alpha * x))
#         m = torch.logsumexp(alpha * x, dim=-1) - math.log(T)
#         return m / alpha  # [B, C]

# def _tail_preserving_pool(x: torch.Tensor, k_prefix: int, k_tail: int) -> torch.Tensor:
#     """
#     Keep the last k_tail tokens verbatim; average-pool the prefix to k_prefix bins.
#     x: [B, C, S]  ->  [B, C, k_prefix + k_tail]
#     """
#     B, C, S = x.shape
#     k_tail = max(0, min(k_tail, S))
#     k_prefix = max(0, k_prefix)
#     if k_tail == 0:
#         return F.adaptive_avg_pool1d(x, k_prefix) if k_prefix > 0 else x
#     tail = x[..., S - k_tail :]                       # [B, C, k_tail]
#     if S - k_tail <= 0 or k_prefix == 0:
#         return tail
#     prefix = x[..., : S - k_tail]                     # [B, C, S - k_tail]
#     prefix_ds = F.adaptive_avg_pool1d(prefix, k_prefix)
#     return torch.cat([prefix_ds, tail], dim=-1)

# class _TCNBlock(nn.Module):
#     """
#     Minimal residual TCN block: depthwise 1D conv (dilated) + GELU + pointwise 1x1 + dropout + residual.
#     """
#     def __init__(self, channels: int, dilation: int, k: int = 5, pdrop: float = 0.10):
#         super().__init__()
#         pad = dilation * (k // 2)
#         self.dw = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=dilation, groups=channels)
#         self.pw = nn.Conv1d(channels, channels, kernel_size=1)
#         self.act = nn.GELU()
#         self.drop = nn.Dropout(pdrop)
#         self.res_scale = nn.Parameter(torch.tensor(1.0))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.dw(x)
#         y = self.act(y)
#         y = self.pw(y)
#         y = self.drop(y)
#         return x + self.res_scale * y

# class HiddenFeatureExtractorLite_V2(nn.Module):
#     """
#     Simpler, stronger readout (no attention, no masks):
#       - LN + Linear to d_tok
#       - Tail-preserving downsample to k_hid = k_prefix + k_tail
#       - 3 residual TCN blocks with dilations [1, 2, 4]
#       - Global pooling: LSE(mean↔max) + last-window average
#       - 2-layer MLP to D_HID
#     """
#     def __init__(
#         self,
#         D_model: int,
#         D_HID:   int = 512,
#         d_tok:   int = 192,
#         k_hid:   int = 192,
#         k_tail:  int = 64,         # how many raw tail tokens to preserve
#         pdrop:   float = 0.10,
#         last_m:  int = 16,         # window size for last-window average
#         conv_k:  int = 5,
#     ):
#         super().__init__()
#         assert k_tail <= k_hid, "k_tail must be <= k_hid"
#         self.k_prefix = int(k_hid - k_tail)
#         self.k_tail   = int(k_tail)
#         self.last_m   = int(max(1, last_m))

#         self.norm = nn.LayerNorm(D_model)
#         self.proj = nn.Linear(D_model, d_tok)

#         self.tcn = nn.Sequential(
#             _TCNBlock(d_tok, dilation=1, k=conv_k, pdrop=pdrop),
#             _TCNBlock(d_tok, dilation=2, k=conv_k, pdrop=pdrop),
#             _TCNBlock(d_tok, dilation=4, k=conv_k, pdrop=pdrop),
#         )
#         self.drop = nn.Dropout(pdrop)

#         self.lse_pool = _LSEPool1d(init_alpha=4.0)  # learns mean↔max
#         # Head: [LSE(C) || last_avg(C)] -> 2*d_tok -> D_HID
#         self.out = nn.Sequential(
#             nn.Linear(2 * d_tok, 2 * d_tok),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(2 * d_tok, D_HID),
#         )

#     def forward(self, last_hidden: torch.Tensor, mask_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # last_hidden: [B, S, D_model]
#         B, S, _ = last_hidden.shape

#         # Project to token channels and go to [B, C, S]
#         x = self.proj(self.norm(last_hidden))          # [B, S, d_tok]
#         x = x.permute(0, 2, 1).contiguous()           # [B, d_tok, S]

#         # Tail-preserving downsample (fp32 for pooling only)
#         x_ds = _tail_preserving_pool(x.to(torch.float32), self.k_prefix, self.k_tail)  # [B, d_tok, k_hid]
#         x_ds = x_ds.to(x.dtype)

#         # Tiny TCN stack
#         z = self.tcn(x_ds)
#         z = self.drop(z)                               # [B, d_tok, k_hid]

#         # Global pooling: learned LSE + last-window average (no masks)
#         lse = self.lse_pool(z)                         # [B, d_tok]
#         m   = min(self.last_m, z.size(-1))
#         last_avg = z[..., -m:].mean(dim=-1)            # [B, d_tok]

#         vec = torch.cat([lse, last_avg], dim=-1)       # [B, 2*d_tok]
#         return self.out(vec)                           # [B, D_HID]


# #************************************************************************************************
# class DropPath(nn.Module):
#     """Stochastic depth per-sample (keep prob = 1 - p)."""
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = float(drop_prob)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.drop_prob == 0.0 or not self.training:
#             return x
#         keep = 1 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
#         mask = rand.floor() / keep
#         return x * mask

# # --------- TCN block (wider, with expansion + SE) ---------
# class _SE1d(nn.Module):
#     def __init__(self, c: int, r: int = 8):
#         super().__init__()
#         hidden = max(r, c // r)
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(c, hidden, 1), nn.GELU(),
#             nn.Conv1d(hidden, c, 1), nn.Sigmoid()
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.se(x)

# class _TCNBlockWide(nn.Module):
#     """
#     Depthwise-dilated conv -> PW expand -> GELU -> PW project -> SE -> Dropout -> Stochastic-Depth residual
#     """
#     def __init__(self, channels: int, dilation: int, k: int = 7, expand: int = 4, pdrop: float = 0.10, drop_path: float = 0.0):
#         super().__init__()
#         pad = dilation * (k // 2)
#         self.dw = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=dilation, groups=channels)
#         self.pw1 = nn.Conv1d(channels, channels * expand, kernel_size=1)
#         self.act = nn.GELU()
#         self.pw2 = nn.Conv1d(channels * expand, channels, kernel_size=1)
#         self.se  = _SE1d(channels)
#         self.drop = nn.Dropout(pdrop)
#         self.dpath = DropPath(drop_path)
#         self.res_scale = nn.Parameter(torch.tensor(1.0))
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.dw(x)
#         y = self.pw1(y)
#         y = self.act(y)
#         y = self.pw2(y)
#         y = self.se(y)
#         y = self.drop(y)
#         return x + self.dpath(self.res_scale * y)

# # --------- Tiny Transformer mixer (pre-LN, no masks) ---------
# class _TinyTFBlock(nn.Module):
#     def __init__(self, d_model: int, n_heads: int = 8, ff_mult: int = 4, pdrop: float = 0.10):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(d_model)
#         self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=pdrop, batch_first=True)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.ff  = nn.Sequential(
#             nn.Linear(d_model, ff_mult * d_model),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(ff_mult * d_model, d_model),
#         )
#         self.drop = nn.Dropout(pdrop)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, T, C]
#         h = self.ln1(x)
#         a, _ = self.attn(h, h, h, need_weights=False)   # no masks
#         x = x + self.drop(a)
#         h = self.ln2(x)
#         x = x + self.drop(self.ff(h))
#         return x

# # --------- The extractor (V3, heavier; optional Transformer) ---------
# class HiddenFeatureExtractorLite_V3(nn.Module):
#     """
#     Heavier V2 (no masks):
#       - LN + Linear to d_tok (wider)
#       - Tail-preserving downsample to k_hid
#       - 6 residual TCN blocks (dilations [1,2,4,8,16,32]) with expansion + SE
#       - Optional tiny Transformer mixer (L layers)
#       - Global pooling: LSE(mean↔max) + last-window average
#       - 3-layer MLP to D_HID
#     """
#     def __init__(
#         self,
#         D_model: int,
#         D_HID:   int = 768,
#         d_tok:   int = 256,
#         k_hid:   int = 256,
#         k_tail:  int = 96,          # raw tail tokens to preserve
#         pdrop:   float = 0.10,
#         last_m:  int = 24,          # last-window avg size
#         conv_k:  int = 7,
#         expand:  int = 4,           # TCN expansion ratio
#         drop_path: float = 0.10,    # stochastic depth across TCN stack
#         attn_layers: int = 2,       # set 0 to disable Transformer mixer
#         attn_heads:  int = 8,
#         attn_ff_mult: int = 4,
#         use_pos_embed: bool = True, # learned absolute pos for Transformer stage
#     ):
#         super().__init__()
#         assert k_tail <= k_hid, "k_tail must be <= k_hid"
#         self.k_prefix = int(k_hid - k_tail)
#         self.k_tail   = int(k_tail)
#         self.last_m   = int(max(1, last_m))
#         self.attn_layers = int(attn_layers)
#         self.use_pos = bool(use_pos_embed)

#         # input proj
#         self.norm = nn.LayerNorm(D_model)
#         self.proj = nn.Linear(D_model, d_tok)

#         # deeper TCN stack
#         dilations = [1, 2, 4, 8, 16, 32]
#         dpaths = torch.linspace(0.0, drop_path, steps=len(dilations)).tolist()
#         self.tcn = nn.Sequential(*[
#             _TCNBlockWide(d_tok, d, k=conv_k, expand=expand, pdrop=pdrop, drop_path=dpaths[i])
#             for i, d in enumerate(dilations)
#         ])
#         self.drop = nn.Dropout(pdrop)

#         # optional Transformer mixer
#         if self.attn_layers > 0:
#             self.pos = nn.Parameter(torch.randn(1, k_hid, d_tok) / math.sqrt(d_tok)) if self.use_pos else None
#             self.tf = nn.ModuleList([
#                 _TinyTFBlock(d_model=d_tok, n_heads=attn_heads, ff_mult=attn_ff_mult, pdrop=pdrop)
#                 for _ in range(self.attn_layers)
#             ])

#         # pooling & head
#         self.lse_pool = _LSEPool1d(init_alpha=4.0)
#         self.out = nn.Sequential(
#             nn.Linear(2 * d_tok, 3 * d_tok),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(3 * d_tok, 2 * d_tok),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(2 * d_tok, D_HID),
#         )

#     def forward(self, last_hidden: torch.Tensor, mask_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
#         # last_hidden: [B, S, D_model]
#         x = self.proj(self.norm(last_hidden))          # [B, S, d_tok]
#         x = x.permute(0, 2, 1).contiguous()           # [B, d_tok, S]

#         # tail-preserving downsample (fp32 for pooling only)
#         x_ds = _tail_preserving_pool(x.to(torch.float32), self.k_prefix, self.k_tail)  # [B, d_tok, k_hid]
#         x_ds = x_ds.to(x.dtype)

#         # TCN
#         z = self.tcn(x_ds)
#         z = self.drop(z)                               # [B, d_tok, k_hid]

#         # optional tiny Transformer mixer
#         if self.attn_layers > 0:
#             tok = z.permute(0, 2, 1).contiguous()      # [B, k_hid, d_tok]
#             if self.use_pos:
#                 tok = tok + self.pos.to(tok.dtype)
#             for blk in self.tf:
#                 tok = blk(tok)                         # [B, k_hid, d_tok]
#             z = tok.permute(0, 2, 1).contiguous()      # back to [B, d_tok, k_hid]

#         # pooling
#         lse = self.lse_pool(z)                         # [B, d_tok]
#         m   = min(self.last_m, z.size(-1))
#         last_avg = z[..., -m:].mean(dim=-1)            # [B, d_tok]

#         vec = torch.cat([lse, last_avg], dim=-1)       # [B, 2*d_tok]
#         return self.out(vec)                           # [B, D_HID]


# #************************************************************************************************
# #V4
# # ----- helpers -----
# # best_hidden_no_pos_safe_pool.py
# # Self-contained: safe pooling + best-effort no-pos encoder (+ optional classifier)

# import math
# from typing import Optional
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ============================================================
# # Safe, differentiable pooling (no AdaptiveAvgPool1d kernels)
# # ============================================================

# def _safe_adaptive_avg_pool1d_channel(x: torch.Tensor, k_out: int) -> torch.Tensor:
#     """
#     x: [B, C, S] -> [B, C, k_out]
#     Scatter-add segment mean (CUDA-friendly). Differentiable w.r.t x.
#     """
#     B, C, S = x.shape
#     if k_out <= 0 or S == 0:
#         return x
#     if S == k_out:
#         return x
#     if k_out == 1:
#         return x.mean(dim=-1, keepdim=True)

#     device = x.device
#     idx = (torch.arange(S, device=device) * k_out) // S           # [S], bin index per source position
#     idx[-1] = k_out - 1                                           # ensure last bin covered

#     out = x.new_zeros(B, C, k_out)                                # sums
#     out.scatter_add_(-1, idx.view(1, 1, S).expand(B, C, S), x)

#     counts = torch.bincount(idx, minlength=k_out).to(x.dtype)     # [k_out]
#     counts = counts.clamp_min(1).view(1, 1, k_out)

#     return out / counts

# def _safe_tail_preserving_pool_channel(x: torch.Tensor, k_prefix: int, k_tail: int) -> torch.Tensor:
#     """
#     x: [B, C, S] -> [B, C, k_prefix + k_tail]
#     Keeps last k_tail tokens verbatim; pools prefix to k_prefix bins (safe).
#     """
#     B, C, S = x.shape
#     k_tail   = max(0, min(k_tail, S))
#     k_prefix = max(0, k_prefix)

#     if k_tail == 0:
#         return _safe_adaptive_avg_pool1d_channel(x, k_prefix) if k_prefix > 0 else x

#     tail = x[..., S - k_tail:]  # [B, C, k_tail]
#     if S - k_tail <= 0 or k_prefix == 0:
#         return tail

#     prefix   = x[..., : S - k_tail]
#     prefix_d = _safe_adaptive_avg_pool1d_channel(prefix, k_prefix)  # [B, C, k_prefix]
#     return torch.cat([prefix_d, tail], dim=-1)

# def safe_adaptive_avg_pool1d_seq(x_seq: torch.Tensor, k_out: int) -> torch.Tensor:
#     """Sequence-first wrapper: x_seq [B, S, C] -> [B, k_out, C]."""
#     return _safe_adaptive_avg_pool1d_channel(x_seq.permute(0, 2, 1).contiguous(), k_out).permute(0, 2, 1).contiguous()

# def tail_preserve_downsample_seq_safe(x_seq: torch.Tensor, k_prefix: int, k_tail: int) -> torch.Tensor:
#     """Sequence-first wrapper: x_seq [B, S, C] -> [B, k_prefix+k_tail, C]."""
#     return _safe_tail_preserving_pool_channel(x_seq.permute(0, 2, 1).contiguous(), k_prefix, k_tail).permute(0, 2, 1).contiguous()

# # =========================
# # Building blocks (no pos)
# # =========================

# class DropPath(nn.Module):
#     def __init__(self, drop_prob: float = 0.0):
#         super().__init__()
#         self.drop_prob = float(drop_prob)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.drop_prob == 0.0 or not self.training:
#             return x
#         keep = 1.0 - self.drop_prob
#         shape = (x.shape[0],) + (1,) * (x.ndim - 1)
#         mask = (keep + torch.rand(shape, dtype=x.dtype, device=x.device)).floor() / keep
#         return x * mask

# class LSEPoolSeq(nn.Module):
#     """Log-Sum-Exp pooling over sequence dim (B, S, C) -> (B, C)."""
#     def __init__(self, init_alpha: float = 4.0):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, S, C = x.shape
#         a = torch.clamp(self.alpha, 0.0, 20.0)
#         if a < 1e-3:
#             return x.mean(dim=1)
#         m = torch.logsumexp(a * x, dim=1) - math.log(max(1, S))
#         return m / a

# class SE1d(nn.Module):
#     def __init__(self, c: int, r: int = 8):
#         super().__init__()
#         hidden = max(r, c // r)
#         self.net = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(c, hidden, 1), nn.GELU(),
#             nn.Conv1d(hidden, c, 1), nn.Sigmoid()
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * self.net(x)

# class TCNBlockWide(nn.Module):
#     """Depthwise-dilated conv -> PW expand -> GELU -> PW project -> SE -> Dropout -> Stochastic-Depth residual."""
#     def __init__(self, channels: int, dilation: int, k: int = 7, expand: int = 4, pdrop: float = 0.10, drop_path: float = 0.0):
#         super().__init__()
#         pad = dilation * (k // 2)
#         self.dw   = nn.Conv1d(channels, channels, kernel_size=k, padding=pad, dilation=dilation, groups=channels)
#         self.pw1  = nn.Conv1d(channels, channels * expand, kernel_size=1)
#         self.act  = nn.GELU()
#         self.pw2  = nn.Conv1d(channels * expand, channels, kernel_size=1)
#         self.se   = SE1d(channels)
#         self.drop = nn.Dropout(pdrop)
#         self.dp   = DropPath(drop_path)
#         self.res_scale = nn.Parameter(torch.tensor(1.0))
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.dw(x)
#         y = self.pw1(y); y = self.act(y)
#         y = self.pw2(y)
#         y = self.se(y)
#         y = self.drop(y)
#         return x + self.dp(self.res_scale * y)

# class TinyTFBlock(nn.Module):
#     """Pre-LN Transformer block (no masks, no positional embeddings)."""
#     def __init__(self, d_model: int, n_heads: int = 8, ff_mult: int = 4, pdrop: float = 0.10):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(d_model)
#         self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=pdrop, batch_first=True)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.ff  = nn.Sequential(
#             nn.Linear(d_model, ff_mult * d_model),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(ff_mult * d_model, d_model),
#         )
#         self.drop = nn.Dropout(pdrop)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         h = self.ln1(x)
#         a, _ = self.attn(h, h, h, need_weights=False)
#         x = x + self.drop(a)
#         h = self.ln2(x)
#         x = x + self.drop(self.ff(h))
#         return x

# class SeedAttentionPool(nn.Module):
#     """K learned queries attend over sequence to produce K pooled slots (no pos enc)."""
#     def __init__(self, d_model: int, K: int = 6, n_heads: int = 6, pdrop: float = 0.10):
#         super().__init__()
#         self.seeds = nn.Parameter(torch.randn(K, d_model) / math.sqrt(d_model))
#         self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=pdrop, batch_first=True)
#         self.drop  = nn.Dropout(pdrop)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, _, C = x.shape
#         Q = self.seeds.unsqueeze(0).expand(B, -1, -1)  # [B, K, C]
#         out, _ = self.attn(Q, x, x, need_weights=False)
#         return self.drop(out)  # [B, K, C]

# # ==============================================
# # Best-effort encoder (no positional embeddings)
# # ==============================================

# class HiddenFeatureExtractorLite_V4(nn.Module):
#     """
#     Best-effort encoder without positional embeddings:
#       - LN + Linear to d_tok
#       - Tail-preserving downsample to k_hid (safe pooling)
#       - Dual trunk: TCN (multi-dilation) + Tiny Transformer (2L, no pos)
#       - Gated fusion (token-wise)
#       - Seed attention pooling + LSE + last-window + last-token
#       - Strong MLP to D_HID
#     """
#     def __init__(
#         self,
#         D_model: int,
#         D_HID:   int = 768,
#         d_tok:   int = 384,
#         k_hid:   int = 256,
#         k_tail:  int = 96,
#         pdrop:   float = 0.10,
#         conv_k:  int = 7,
#         tcn_expand: int = 4,
#         tcn_drop_path: float = 0.10,
#         tf_layers: int = 2,
#         tf_heads:  int = 8,
#         tf_ff_mult: int = 4,
#         seeds_K: int = 6,
#         seeds_heads: int = 6,
#         last_m:   int = 32,
#     ):
#         super().__init__()
#         assert k_tail <= k_hid
#         self.k_prefix = int(k_hid - k_tail)
#         self.k_tail   = int(k_tail)
#         self.last_m   = int(max(1, last_m))
#         self.d_tok    = int(d_tok)
#         self.out_dim  = int(D_HID)

#         # Input projection
#         self.norm = nn.LayerNorm(D_model)
#         self.proj = nn.Linear(D_model, d_tok)

#         # TCN trunk (channel-first)
#         dilations = [1, 2, 4, 8, 16, 32]
#         dpaths = torch.linspace(0.0, tcn_drop_path, steps=len(dilations)).tolist()
#         self.tcn = nn.Sequential(*[
#             TCNBlockWide(d_tok, d, k=conv_k, expand=tcn_expand, pdrop=pdrop, drop_path=dpaths[i])
#             for i, d in enumerate(dilations)
#         ])
#         self.tcn_drop = nn.Dropout(pdrop)

#         # Tiny Transformer trunk (sequence-first, no pos)
#         self.tf_layers = int(tf_layers)
#         if self.tf_layers > 0:
#             self.tf = nn.ModuleList([
#                 TinyTFBlock(d_model=d_tok, n_heads=tf_heads, ff_mult=tf_ff_mult, pdrop=pdrop)
#                 for _ in range(self.tf_layers)
#             ])

#         # Gated fusion
#         self.fuse_gate = nn.Linear(2 * d_tok, d_tok)

#         # Pooling
#         self.seeds = SeedAttentionPool(d_model=d_tok, K=seeds_K, n_heads=seeds_heads, pdrop=pdrop)
#         self.lse   = LSEPoolSeq(init_alpha=4.0)

#         # Head: [(K+3)*C + 6 stats] -> 3C -> 2C -> D_HID
#         pooled_dim = (seeds_K + 3) * d_tok
#         self.stats_norm = nn.LayerNorm(6)
#         self.head = nn.Sequential(
#             nn.Linear(pooled_dim + 6, 3 * d_tok),
#             nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(3 * d_tok, 2 * d_tok),
#             nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(2 * d_tok, D_HID),
#         )

#     def _fast_stats(self, x_tok: torch.Tensor) -> torch.Tensor:
#         """x_tok: [B, T, C]; returns [B, 6] over token-norm trajectory."""
#         traj = x_tok.norm(dim=-1)  # [B, T]
#         B, T = traj.shape
#         mean = traj.mean(dim=1, keepdim=True)
#         var  = traj.var(dim=1, unbiased=False, keepdim=True)
#         if T >= 2:
#             d = traj[:, 1:] - traj[:, :-1]
#             k = max(1, int(0.9 * (d.shape[1])))
#             p90d = d.abs().kthvalue(k, dim=1).values.unsqueeze(1)
#             tv   = d.abs().sum(dim=1, keepdim=True)
#         else:
#             p90d = traj.new_zeros(B, 1); tv = traj.new_zeros(B, 1)
#         w = max(1, T // 3)
#         el = (traj[:, :w].mean(dim=1, keepdim=True) - traj[:, -w:].mean(dim=1, keepdim=True))
#         last = traj[:, -1:].contiguous()
#         stats = torch.cat([mean, var, tv, p90d, el, last], dim=1)  # [B, 6]
#         return self.stats_norm(stats)

#     def forward(self, last_hidden: torch.Tensor) -> torch.Tensor:
#         # 1) project
#         x = self.proj(self.norm(last_hidden))                           # [B, S, C]

#         # 2) safe tail-preserving downsample (sequence-first)
#         x = tail_preserve_downsample_seq_safe(x, self.k_prefix, self.k_tail)  # [B, T', C]
#         B, T, C = x.shape

#         # 3) TCN trunk (channel-first)
#         xt = self.tcn(x.transpose(1, 2))                         # [B, C, T']
#         xt = self.tcn_drop(xt).transpose(1, 2).contiguous()      # [B, T', C]

#         # 4) Transformer trunk (no pos)
#         if self.tf_layers > 0:
#             xf = x
#             for blk in self.tf:
#                 xf = blk(xf)                                     # [B, T', C]
#         else:
#             xf = x

#         # 5) gated fusion
#         cat  = torch.cat([xf, xt], dim=-1)                       # [B, T', 2C]
#         gate = torch.sigmoid(self.fuse_gate(cat))                # [B, T', C]
#         fused = gate * xf + (1.0 - gate) * xt                    # [B, T', C]

#         # 6) pooling (seeds + LSE + last-window + last-token)
#         seeds = self.seeds(fused).reshape(B, -1)                 # [B, K*C]
#         lse   = self.lse(fused)                                  # [B, C]
#         m     = min(self.last_m, T)
#         last_avg = fused[:, -m:, :].mean(dim=1)                  # [B, C]
#         last_tok = fused[:, -1, :]                               # [B, C]

#         pooled = torch.cat([seeds, lse, last_avg, last_tok], dim=-1)  # [B, (K+3)*C]
#         stats  = self._fast_stats(fused)                         # [B, 6]
#         return self.head(torch.cat([pooled, stats], dim=-1))     # [B, D_HID]




def _logit_clamped(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log1p(-x)

class ConfFeatureExtractorLite(nn.Module):
    """
    Features from a 1D confidence sequence via multi-dilated convs + SAB + PMA.
    """
    def __init__(
        self, D_CONF: int = 384, d_tok: int = 128, k_conf: int = 192, base_c: int = 64,
        K: int = 3, sab_layers: int = 1, sab_heads: int = 4, pdrop: float = 0.10,
    ):
        super().__init__()
        self.k_conf, self.d_tok = int(k_conf), int(d_tok)
        self.stem = nn.Conv1d(3, base_c, kernel_size=5, padding=2)
        self.gn0 = nn.GroupNorm(_num_groups(base_c), base_c)

        g = _num_groups(base_c)
        def dw(dil):
            return nn.Sequential(
                nn.Conv1d(base_c, base_c, 5, padding=2*dil, dilation=dil, groups=g),
                nn.GroupNorm(_num_groups(base_c), base_c),
                nn.GELU()
            )
        self.dw1, self.dw2, self.dw3 = dw(1), dw(2), dw(4)
        self.mix_gate = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]), requires_grad=True)
        hid = max(8, base_c // 8)
        self.se = nn.Sequential(nn.Conv1d(base_c, hid, 1), nn.GELU(), nn.Conv1d(hid, base_c, 1), nn.Sigmoid())
        self.proj_tok = nn.Conv1d(base_c, d_tok, kernel_size=1)
        self.pos = nn.Parameter(torch.randn(1, self.k_conf, d_tok) / math.sqrt(d_tok))
        self.sab = SAB(d_model=d_tok, n_heads=sab_heads, pdrop=pdrop, num_layers=sab_layers)
        self.pma = PMA(d_model=d_tok, num_seeds=K, n_heads=sab_heads, pdrop=pdrop)
        self.out = nn.Sequential(
            nn.Linear(K * d_tok + 14, 2 * d_tok), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(2 * d_tok, D_CONF)
        )

    @torch.no_grad()
    def _rich_stats(self, x: torch.Tensor) -> torch.Tensor:
        B, k = x.shape
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, unbiased=False, keepdim=True)
        dx   = x[:, 1:] - x[:, :-1] if k >= 2 else torch.zeros(B, 0, device=x.device)
        tv   = dx.abs().sum(dim=-1, keepdim=True) if dx.numel() else torch.zeros(B, 1, device=x.device)
        p90d = percentile(dx.abs(), 0.90, dim=-1, keepdim=True) if dx.numel() else torch.zeros(B, 1, device=x.device)

        t = torch.arange(k, device=x.device, dtype=x.dtype).unsqueeze(0)
        t = t - t.mean()
        denom_t = (t**2).sum() + 1e-9
        slope = (t * (x - mean)).sum(dim=-1, keepdim=True) / denom_t

        varx  = var.clamp_min(1e-9)
        r2    = ((t * (x - mean)).sum(dim=-1, keepdim=True)**2 / (denom_t * varx * k)).clamp(0, 1)

        drawdown = (torch.cummax(x, dim=-1).values - x).amax(dim=-1, keepdim=True)
        peaks = ((dx[:, :-1] > 0.02) & (dx[:, 1:] < -0.02)).float().sum(dim=-1, keepdim=True) if dx.size(1) >= 2 else torch.zeros(B,1,device=x.device)

        p50, p70, p90 = (x > 0.5).float().mean(-1, keepdim=True), (x > 0.7).float().mean(-1, keepdim=True), (x > 0.9).float().mean(-1, keepdim=True)

        P  = torch.fft.rfft(x, dim=-1).abs()**2 + 1e-12
        M  = P.shape[-1]; q1, q2 = M//4, M//2
        Pl, Pm, PhPv = P[:, :q1].mean(-1, True), P[:, q1:q2].mean(-1, True), P[:, q2:].mean(-1, True)

        # 14 dims total
        return torch.cat([mean, var, tv, p90d, slope, r2, drawdown, peaks, p50, p70, p90, Pl, Pm, PhPv], dim=-1)

    def forward(self, conf: torch.Tensor, mask_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = conf.shape
        raw = conf
        d = F.pad(conf[:, 1:] - conf[:, :-1], (1,0)) if S >= 2 else torch.zeros_like(conf)
        lg = _logit_clamped(conf.to(torch.float32)).to(conf.dtype)
        x = torch.stack([raw, d, lg], dim=1)

        # Mask-free downsampling
        x = F.adaptive_avg_pool1d(x.to(torch.float32), self.k_conf)
        x = x.to(_module_param_dtype(self.stem))  # match conv dtype

        z = F.gelu(self.gn0(self.stem(x)))
        y1, y2, y3 = self.dw1(z), self.dw2(z), self.dw3(z)
        g = torch.softmax(self.mix_gate, dim=0)
        mix = g[0]*y1 + g[1]*y2 + g[2]*y3
        y   = mix * self.se(mix) + z

        tok = self.proj_tok(y).permute(0,2,1).contiguous() + self.pos.to(y.dtype)
        tok = self.sab(tok)
        pooled = self.pma(tok).flatten(1)
        stats = self._rich_stats(tok.mean(dim=-1).to(torch.float32)).to(pooled.dtype)
        vec = torch.cat([pooled, stats], dim=-1)
        return self.out(vec)



# conf_logits_dist_stats_mlp_v2_nolabel.py
import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Utilities for sequence pooling & summaries
# ----------------------------
def _pool_nonoverlap(x: torch.Tensor, k: int) -> torch.Tensor:
    # x: (B,S) -> (B, floor(S/k)) via avg_pool1d stride=kernel (dropping tail remainder)
    k = max(1, min(k, x.size(1)))
    return F.avg_pool1d(x.unsqueeze(1), kernel_size=k, stride=k).squeeze(1)

def _last_window(x: torch.Tensor, w: int) -> torch.Tensor:
    L = max(1, min(w, x.size(1)))
    return x[:, -L:]

@torch.no_grad()
def _seq_summary(x: torch.Tensor) -> torch.Tensor:
    # x: (B,S) -> (B, 10): mean,std,min,max,median,q10,q25,q75,q90,p95
    qs = torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=x.device, dtype=x.dtype)
    q = torch.quantile(x, qs, dim=-1).transpose(0, 1)  # (B,6)
    mean = x.mean(-1, True)
    std  = x.std(-1, unbiased=False, keepdim=True)
    mn   = x.amin(-1, True)
    mx   = x.amax(-1, True)
    return torch.cat([mean, std, mn, mx, q[:,2:3], q[:,0:1], q[:,1:2], q[:,3:4], q[:,4:5], q[:,5:6]], dim=-1)

@torch.no_grad()
def _window_block(x: torch.Tensor, w: int) -> torch.Tensor:
    """
    x (B,S) ->
      Across (non-overlap): mean/std/max/min of window means and window stds -> 8 dims
      Last-window: mean/std/min/max/range + slope + R^2 -> 7 dims
      Total: 15 dims per w
    """
    # across
    pooled = _pool_nonoverlap(x, w)                        # (B,Nw)
    x1 = x.unsqueeze(1)
    k = max(1, min(w, x.size(1)))
    m  = F.avg_pool1d(x1, kernel_size=k, stride=k)         # (B,1,Nw)
    m2 = F.avg_pool1d(x1 * x1, kernel_size=k, stride=k)
    std_win = (m2 - m * m).clamp_min(0).sqrt().squeeze(1)  # (B,Nw)
    mean_win = pooled
    def s4(t): return torch.stack([t.mean(-1), t.std(-1, False), t.max(-1).values, t.min(-1).values], dim=-1)
    across = torch.cat([s4(mean_win), s4(std_win)], dim=-1)  # 8

    # last-window
    xl = _last_window(x, w)
    mean_l = xl.mean(-1, True)
    std_l  = xl.std(-1, False, True)
    min_l  = xl.amin(-1, True)
    max_l  = xl.amax(-1, True)
    rng_l  = max_l - min_l
    B, L = xl.shape
    t = torch.arange(L, device=xl.device, dtype=xl.dtype).unsqueeze(0)
    t = t - t.mean()
    denom_t = (t.pow(2).sum() + 1e-9)
    xc = xl - xl.mean(dim=-1, keepdim=True)
    slope_l = (t * xc).sum(dim=-1, keepdim=True) / denom_t
    varx = (xc.pow(2).mean(dim=-1, keepdim=True)).clamp_min(1e-9)
    r2_l = ((t * xc).sum(dim=-1, keepdim=True).pow(2) / (denom_t * varx * L)).clamp(0, 1)
    last = torch.cat([mean_l, std_l, min_l, max_l, rng_l, slope_l, r2_l], dim=-1)  # 7
    return torch.cat([across, last], dim=-1)  # 15

@torch.no_grad()
def _lastK_block(x: torch.Tensor, K: int) -> torch.Tensor:
    # last-K emphasis -> (B,7): mean,std,min,max,p90,slope,R^2
    xl = _last_window(x, K)
    mean_l = xl.mean(-1, True)
    std_l  = xl.std(-1, False, True)
    min_l  = xl.amin(-1, True)
    max_l  = xl.amax(-1, True)
    p90_l  = torch.quantile(xl, 0.90, dim=-1, keepdim=True)
    B, L = xl.shape
    t = torch.arange(L, device=xl.device, dtype=xl.dtype).unsqueeze(0); t = t - t.mean()
    denom_t = (t.pow(2).sum() + 1e-9)
    xc = xl - xl.mean(dim=-1, keepdim=True)
    slope_l = (t * xc).sum(dim=-1, keepdim=True) / denom_t
    varx = (xc.pow(2).mean(dim=-1, keepdim=True)).clamp_min(1e-9)
    r2_l = ((t * xc).sum(dim=-1, keepdim=True).pow(2) / (denom_t * varx * L)).clamp(0, 1)
    return torch.cat([mean_l, std_l, min_l, max_l, p90_l, slope_l, r2_l], dim=-1)

@torch.no_grad()
def _run_length_mask_stats(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: (B,S) bool; returns (B,5): mean_run, max_run, n_runs, last_run_len, in_run_flag
    """
    B, S = mask.shape
    out = torch.zeros(B, 5, device=mask.device, dtype=torch.float32)
    for b in range(B):
        mb = mask[b]
        runs = []
        cnt = 0
        for i in range(S):
            if mb[i]:
                cnt += 1
            elif cnt > 0:
                runs.append(cnt); cnt = 0
        if cnt > 0: runs.append(cnt)
        if len(runs) == 0:
            mean_run = torch.tensor(0., device=mask.device)
            max_run  = torch.tensor(0., device=mask.device)
            n_runs   = torch.tensor(0., device=mask.device)
            last_len = torch.tensor(float(cnt) if (S>0 and mb[-1]) else 0.0, device=mask.device)
            in_flag  = torch.tensor(1.0 if (S>0 and mb[-1]) else 0.0, device=mask.device)
        else:
            r = torch.tensor(runs, device=mask.device, dtype=torch.float32)
            mean_run = r.mean()
            max_run  = r.max()
            n_runs   = torch.tensor(float(len(runs)), device=mask.device)
            last_len = torch.tensor(float(runs[-1]) if mb[-1] else 0.0, device=mask.device)
            in_flag  = torch.tensor(1.0 if mb[-1] else 0.0, device=mask.device)
        out[b] = torch.stack([mean_run, max_run, n_runs, last_len, in_flag])
    return out


# ----------------------------
# Memory-safe top-k over vocab (optional tiled)
# ----------------------------
@torch.no_grad()
def _topk_values_tiled(logits: torch.Tensor, k: int, tile: int = 8192) -> torch.Tensor:
    """
    Compute top-k over the vocab dimension in tiles to reduce peak memory.
    logits: (B,S,V), returns values (B,S,k)
    """
    B, S, V = logits.shape
    k = min(k, V)
    device = logits.device
    dtype  = torch.float32

    # initialize running top-k with very small values
    vals = torch.full((B, S, 0), -float("inf"), device=device, dtype=dtype)
    for start in range(0, V, tile):
        end = min(start + tile, V)
        chunk = logits[:, :, start:end].to(dtype)
        # merge with previous candidates and keep top-k
        cand = torch.cat([vals, chunk], dim=-1)            # (B,S,prev+k_tile)
        vals = torch.topk(cand, k=k, dim=-1).values        # (B,S,k)
    return vals


# ----------------------------
# Stronger head: Gated Residual MLP (lazy)
# ----------------------------
class _GatedResMLP(nn.Module):
    """
    Gated residual MLP with GEGLU blocks:
      x --LN--> fc_in(dim->2h) --chunk--> (a,g)
                    gelu(a)*g  --drop-->  fc_out(h->dim)
             residual + LN  (repeat 'blocks' times)
      final: Linear(dim->out_dim)
    """
    def __init__(self, out_dim: int, width_scale: float = 3.0, blocks: int = 2, pdrop: float = 0.1):
        super().__init__()
        self.out_dim = out_dim
        self.width_scale = width_scale
        self.blocks = blocks
        self.pdrop = pdrop
        self._built = False

    def _build(self, in_dim: int):
        h = max(128, int(round(in_dim * self.width_scale)))
        self.in_dim = in_dim
        self.h = h

        layers = []
        # initial LN before the stack
        self.ln0 = nn.LayerNorm(in_dim)

        # build GEGLU residual blocks
        self.fc_in  = nn.ModuleList([nn.Linear(in_dim, 2*h) for _ in range(self.blocks)])
        self.drop1  = nn.ModuleList([nn.Dropout(self.pdrop)   for _ in range(self.blocks)])
        self.fc_out = nn.ModuleList([nn.Linear(h, in_dim)     for _ in range(self.blocks)])  # <-- h -> dim (FIX)
        self.drop2  = nn.ModuleList([nn.Dropout(self.pdrop)   for _ in range(self.blocks)])
        self.ln     = nn.ModuleList([nn.LayerNorm(in_dim)     for _ in range(self.blocks)])

        # head
        self.head = nn.Linear(in_dim, self.out_dim)
        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._built:
            self._build(x.shape[-1])
            self.to(x.device, dtype=x.dtype)

        y = self.ln0(x)
        for i in range(self.blocks):
            u = self.fc_in[i](y)                 # (B, 2h)
            a, g = torch.chunk(u, 2, dim=-1)     # (B, h), (B, h)
            z = F.gelu(a) * g                    # (B, h)  <-- gated
            z = self.drop1[i](z)
            z = self.fc_out[i](z)                # (B, dim)
            z = self.drop2[i](z)
            y = self.ln[i](y + z)                # residual + norm
        return self.head(y)

# ----------------------------
# Main: Label-free logits extractor (V2)
# ----------------------------
class ConfLogitsDistStatsMLP_V2_NoLabel(nn.Module):
    """
    Label-free strong extractor:
      Inputs:  logits (B,S,V)
      Outputs: (B, D_CONF)
    Features:
      • top-k entropy, margin (z1-z2), top-1 prob, mass@k (k ∈ mass_ks)
      • global + per-window + last-K stats on {entropy, margin, p1}
      • run-length stats for low-margin & high-entropy regions
      • sequence length features + two PPL proxies: exp(mean(-log p1)), exp(mean(H_k))
      • optional tiled top-k to reduce memory
      • Gated Residual MLP head
    """
    def __init__(
        self,
        D_CONF: int = 256,
        window_sizes: Sequence[int] = (8, 32, 128, 512),
        lastK_sizes: Sequence[int] = (16, 64, 256),
        entropy_topk: int = 64,                 # compute entropy on top-k logits
        mass_ks: Sequence[int] = (1, 3, 5, 10, 50),
        margin_low_tau: float = 1.0,            # logit gap threshold for "low confidence"
        entropy_high_frac: float = 0.65,        # "high entropy" threshold as a fraction of log(k)
        use_tiled_topk: bool = False,           # enable tiled top-k across vocab for memory
        topk_tile: int = 8192,                  # tile size when tiled top-k is on
        pdrop_mlp: float = 0.10,
        res_blocks: int = 2,
        width_scale: float = 3.0,
    ):
        super().__init__()
        self.window_sizes = tuple(int(w) for w in window_sizes)
        self.lastK_sizes  = tuple(int(w) for w in lastK_sizes)
        self.entropy_topk = int(entropy_topk)
        self.mass_ks      = tuple(int(k) for k in mass_ks)
        self.margin_low_tau = float(margin_low_tau)
        self.entropy_high_frac = float(entropy_high_frac)
        self.use_tiled_topk = bool(use_tiled_topk)
        self.topk_tile = int(topk_tile)

        self.mlp = _GatedResMLP(out_dim=D_CONF, width_scale=width_scale, blocks=res_blocks, pdrop=pdrop_mlp)

    # ---------- Core per-token sequences (label-free) ----------
    @torch.no_grad()
    def _topk_values(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if self.use_tiled_topk:
            return _topk_values_tiled(logits, k=k, tile=self.topk_tile)    # (B,S,k)
        return torch.topk(logits.float(), k=min(k, logits.size(-1)), dim=-1).values  # (B,S,k)

    @torch.no_grad()
    def _entropy_topk(self, topk_vals: torch.Tensor) -> torch.Tensor:
        logpk = torch.log_softmax(topk_vals, dim=-1)
        pk = logpk.exp()
        return (-(pk * logpk).sum(dim=-1)).to(torch.float32)  # (B,S)

    @torch.no_grad()
    def _mass_topk(self, topk_vals: torch.Tensor, k_use: int) -> torch.Tensor:
        k_use = min(k_use, topk_vals.size(-1))
        logpk = torch.log_softmax(topk_vals, dim=-1)
        pk = logpk.exp()
        return pk[..., :k_use].sum(dim=-1).to(torch.float32)  # (B,S)

    @torch.no_grad()
    def _margin_seq(self, topk_vals: torch.Tensor) -> torch.Tensor:
        if topk_vals.size(-1) >= 2:
            return (topk_vals[..., 0] - topk_vals[..., 1]).to(torch.float32)
        return torch.zeros(topk_vals.shape[:2], device=topk_vals.device, dtype=torch.float32)

    @torch.no_grad()
    def _top1_prob(self, topk_vals: torch.Tensor) -> torch.Tensor:
        logpk = torch.log_softmax(topk_vals, dim=-1)
        pk = logpk.exp()
        return pk[..., 0].to(torch.float32)  # (B,S)

    # ---------- Feature packers ----------
    @torch.no_grad()
    def _pack_seq(self, seq: torch.Tensor) -> torch.Tensor:
        parts = [_seq_summary(seq)]
        for w in self.window_sizes:
            parts.append(_window_block(seq, w))
        for K in self.lastK_sizes:
            parts.append(_lastK_block(seq, K))
        return torch.cat(parts, dim=-1)

    @torch.no_grad()
    def _run_feats(self, margin_seq: torch.Tensor, entropy_seq: torch.Tensor, kmax: int) -> torch.Tensor:
        # Normalize entropy by log(kmax) to ~[0,1], then use high-entropy threshold
        H_norm = entropy_seq / max(1.0, math.log(kmax))
        high_h = (H_norm > self.entropy_high_frac)
        low_m  = (margin_seq < self.margin_low_tau)
        Rm = _run_length_mask_stats(low_m)   # (B,5)
        Rh = _run_length_mask_stats(high_h)  # (B,5)
        return torch.cat([Rm, Rh], dim=-1)   # (B,10), plus we’ll also return H_norm mean as a shape feature

    # ---------- Forward ----------
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,S,V) float
        returns: (B, D_CONF)
        """
        print(logits.shape)
        B, S, V = logits.shape
        kmax = min(self.entropy_topk, V)

        # 1) Top-k once, reuse downstream
        tk = self._topk_values(logits, k=kmax)                 # (B,S,kmax)

        # 2) Per-token sequences (label-free)
        Hk    = self._entropy_topk(tk)                         # (B,S)   top-k entropy
        margin= self._margin_seq(tk)                           # (B,S)   z1 - z2
        p1    = self._top1_prob(tk)                            # (B,S)   top-1 probability (top-k normalized)
        mass  = [self._mass_topk(tk, k) for k in self.mass_ks] # [(B,S)] list
        mass_cat = torch.stack(mass, dim=-1)                   # (B,S,len(mass_ks))

        # 3) Proxies for perplexity (label-free)
        #    - ppl_p1 = exp(mean(-log p1))  (p1 in (0,1])
        #    - ppl_Hk = exp(mean(Hk))       (entropy on normalized top-k)
        ppl_p1 = torch.exp((-p1.clamp_min(1e-12).log()).mean(dim=-1, keepdim=True))  # (B,1)
        ppl_Hk = torch.exp(Hk.mean(dim=-1, keepdim=True))                             # (B,1)

        # 4) Rich fixed-size stats for key sequences
        F_H   = self._pack_seq(Hk)
        F_M   = self._pack_seq(margin)
        F_P1  = self._pack_seq(p1)

        # 5) Mass@k summaries (global) + last-K emphasis based on p1
        mass_mean = mass_cat.mean(dim=1)  # (B, len(mass_ks))
        mass_std  = mass_cat.std(dim=1)   # (B, len(mass_ks))
        lastk_blocks = [_lastK_block(p1, K) for K in self.lastK_sizes]  # each (B,7)
        lastk_cat = torch.cat(lastk_blocks, dim=-1) if lastk_blocks else torch.zeros(B, 0, device=logits.device, dtype=logits.dtype)

        # 6) Run-length (low margin / high entropy) + length & shape features
        R = self._run_feats(margin, Hk, kmax)  # (B,10)
        S_feat = torch.full((B,1), float(S), device=logits.device, dtype=torch.float32)
        logS   = torch.full((B,1), math.log(max(1, S)), device=logits.device, dtype=torch.float32)
        H_norm_mean = (Hk / max(1.0, math.log(kmax))).mean(dim=-1, keepdim=True)  # (B,1)
        shape_feats = torch.cat([H_norm_mean, margin.mean(dim=-1, keepdim=True), p1.mean(dim=-1, keepdim=True)], dim=-1)  # (B,3)

        # 7) Concatenate everything and push through the head
        feats = torch.cat([
            F_H, F_M, F_P1,           # big blocks (global + windows + lastK)
            mass_mean, mass_std,      # (B, 2*len(mass_ks))
            lastk_cat,                # (B, 7*len(lastK))
            R,                        # (B,10)
            ppl_p1, ppl_Hk,           # (B,2)
            S_feat, logS,             # (B,2)
            shape_feats,              # (B,3)
        ], dim=-1)

        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).to(logits.dtype)
        return self.mlp(feats)







# ======================================================================================
# SECTION 5: FUSION HEAD
# ======================================================================================

class CorrectnessHeadLite(nn.Module):
    """
    Fuse (attention, confidence, hidden-state) feature vectors with learned gating.
    """
    def __init__(self, D_ATT: int, D_CONF: int, D_HID: int,
                 use_attn=True, use_conf=True, use_hid=True, pdrop: float = 0.10):
        super().__init__()
        self.use_attn, self.use_conf, self.use_hid = use_attn, use_conf, use_hid
        dims = []
        if use_attn: dims.append(D_ATT)
        if use_conf: dims.append(D_CONF)
        if use_hid:  dims.append(D_HID)
        D = sum(dims)
        if D == 0: raise ValueError("Enable at least one modality.")

        self.g_att = nn.Sequential(nn.LayerNorm(D_ATT), nn.Linear(D_ATT, 1)) if use_attn else None
        self.g_con = nn.Sequential(nn.LayerNorm(D_CONF), nn.Linear(D_CONF, 1)) if use_conf else None
        self.g_hid = nn.Sequential(nn.LayerNorm(D_HID),  nn.Linear(D_HID,  1)) if use_hid  else None
        self.ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 384), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(384, 128), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(128, 1),
        )

    def forward(self, z_att: Optional[torch.Tensor], z_conf: Optional[torch.Tensor], z_hid: Optional[torch.Tensor]):
        chunks, gates = [], []
        if self.use_attn: chunks.append(z_att);  gates.append(self.g_att(z_att))
        if self.use_conf: chunks.append(z_conf); gates.append(self.g_con(z_conf))
        if self.use_hid:  chunks.append(z_hid);  gates.append(self.g_hid(z_hid))

        g = torch.softmax(torch.cat(gates, dim=-1), dim=-1)
        out_slices = [ch * g[:, i:i+1] for i, ch in enumerate(chunks)]
        x = torch.cat(out_slices, dim=-1)
        return self.mlp(self.ln(x))








# class _Adapter(nn.Module):
#     """LN + Linear to project a modality vector to the shared fuse dim."""
#     def __init__(self, in_dim: int, out_dim: int, pdrop: float = 0.10):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#         )
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)

# class CorrectnessHead_V2(nn.Module):
#     """
#     Fusion head (attention, confidence, hidden) with:
#       - modality adapters -> shared D_FUSE
#       - tiny Transformer over [CLS] + modality tokens
#       - robust to missing modalities
#       - optional modality dropout
#       - optional temperature scaling (calibration)

#     Forward returns a scalar logit per batch item: [B, 1]
#     """
#     def __init__(
#         self,
#         D_ATT: int,
#         D_CONF: int,
#         D_HID: int,
#         use_attn: bool = True,
#         use_conf: bool = True,
#         use_hid:  bool = True,
#         *,
#         D_FUSE: int = 256,
#         n_heads: int = 4,
#         n_layers: int = 2,
#         pdrop: float = 0.10,
#         modality_drop_p: float = 0.15,   # randomly drop modalities at train time
#         use_temperature: bool = True,    # learn a global temperature for calibration
#     ):
#         super().__init__()
#         self.use_attn, self.use_conf, self.use_hid = use_attn, use_conf, use_hid
#         if not (use_attn or use_conf or use_hid):
#             raise ValueError("Enable at least one modality.")

#         # Adapters to shared space
#         self.adapters = nn.ModuleDict()
#         if use_attn: self.adapters["attn"] = _Adapter(D_ATT,  D_FUSE, pdrop)
#         if use_conf: self.adapters["conf"] = _Adapter(D_CONF, D_FUSE, pdrop)
#         if use_hid:  self.adapters["hid"]  = _Adapter(D_HID,  D_FUSE, pdrop)

#         # Learnable [CLS]/fusion token
#         self.cls = nn.Parameter(torch.randn(1, 1, D_FUSE) / (D_FUSE ** 0.5))

#         # Tiny Transformer for cross-modal interaction
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=D_FUSE, nhead=n_heads,
#             dim_feedforward=4*D_FUSE, dropout=pdrop,
#             activation="gelu", batch_first=True, norm_first=True
#         )
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

#         # Final readout from CLS
#         self.head = nn.Sequential(
#             nn.LayerNorm(D_FUSE),
#             nn.Linear(D_FUSE, 256),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(256, 1),
#         )

#         # Optional calibration temperature (initialized to 1.0)
#         self.use_temperature = use_temperature
#         if use_temperature:
#             self.log_temp = nn.Parameter(torch.zeros(()))  # temp = exp(log_temp)

#         self.modality_drop_p = float(modality_drop_p)

#     def _maybe_drop_modalities(self, names: List[str]) -> List[str]:
#         """
#         During training, randomly drop some modalities to make the head robust.
#         Always keep at least one.
#         """
#         if not self.training or self.modality_drop_p <= 0.0:
#             return names
#         kept = []
#         for n in names:
#             if torch.rand(()) > self.modality_drop_p:
#                 kept.append(n)
#         if len(kept) == 0:
#             kept.append(names[torch.randint(low=0, high=len(names), size=()).item()])
#         return kept

#     def forward(
#         self,
#         z_att: Optional[torch.Tensor],  # [B, D_ATT] or None
#         z_conf: Optional[torch.Tensor], # [B, D_CONF] or None
#         z_hid: Optional[torch.Tensor],  # [B, D_HID] or None
#         return_prob: bool = False
#     ):
#         # Gather available modalities
#         tokens = []
#         names  = []
#         if self.use_attn and z_att is not None: tokens.append(("attn", z_att))
#         if self.use_conf and z_conf is not None: tokens.append(("conf", z_conf))
#         if self.use_hid  and z_hid is not None: tokens.append(("hid",  z_hid))

#         if len(tokens) == 0:
#             raise ValueError("No modality provided to CorrectnessHeadV5.forward(...)")

#         # Optional stochastic modality dropping (train-time)
#         names_all = [n for n, _ in tokens]
#         names_kept = self._maybe_drop_modalities(names_all)
#         tokens = [(n, x) for (n, x) in tokens if n in names_kept]

#         # Adapt to shared space and stack as tokens
#         adapted = []
#         for n, x in tokens:
#             x_adapt = self.adapters[n](x)                 # [B, D_FUSE]
#             adapted.append(x_adapt.unsqueeze(1))          # [B, 1, D_FUSE]
#         X = torch.cat(adapted, dim=1)                     # [B, M, D_FUSE], M <= 3

#         # Prepend CLS token
#         B = X.size(0)
#         cls = self.cls.expand(B, 1, -1)                   # [B, 1, D_FUSE]
#         seq = torch.cat([cls, X], dim=1)                  # [B, 1+M, D_FUSE]

#         # Cross-modal fusion
#         fused = self.encoder(seq)                         # [B, 1+M, D_FUSE]
#         h_cls = fused[:, 0, :]                            # [B, D_FUSE]

#         # Final logit
#         logit = self.head(h_cls)                          # [B, 1]

#         if self.use_temperature:
#             temp = torch.exp(self.log_temp).clamp(min=1e-3, max=100.0)
#             logit = logit / temp

#         if return_prob:
#             prob = torch.sigmoid(logit)
#             return {"logit": logit, "prob": prob, "kept_modalities": names_kept}
#         return logit


# class CorrectnessHead_V3(nn.Module):
#     """
#     Content-adaptive fusion:
#       - Project each modality to D_FUSE
#       - Cross-attention from learnable queries -> modality tokens (per-input weights)
#       - Modality dropout (train-time) to avoid collapse
#       - Final MLP on fused representation
#     """
#     def __init__(
#         self,
#         D_ATT: int, D_CONF: int, D_HID: int,
#         use_attn: bool = True, use_conf: bool = True, use_hid: bool = True,
#         *, D_FUSE: int = 256, n_heads: int = 4, pdrop: float = 0.10,
#         n_queries: int = 2, modality_drop_p: float = 0.15,
#     ):
#         super().__init__()
#         if not (use_attn or use_conf or use_hid):
#             raise ValueError("Enable at least one modality.")

#         self.use_attn, self.use_conf, self.use_hid = use_attn, use_conf, use_hid
#         self.modality_drop_p = modality_drop_p

#         # 1) Adapters to shared space
#         self.adapters = nn.ModuleDict()
#         if use_attn: self.adapters["attn"] = _Adapter(D_ATT,  D_FUSE, pdrop)
#         if use_conf: self.adapters["conf"] = _Adapter(D_CONF, D_FUSE, pdrop)
#         if use_hid:  self.adapters["hid"]  = _Adapter(D_HID,  D_FUSE, pdrop)

#         # 2) Learnable query tokens for cross-attention
#         self.queries = nn.Parameter(torch.randn(1, n_queries, D_FUSE) / (D_FUSE ** 0.5))

#         # 3) Cross-attention (queries -> modality tokens)
#         self.mha = nn.MultiheadAttention(D_FUSE, n_heads, dropout=pdrop, batch_first=True)

#         # Light FFN on query outputs (Transformer-style head)
#         self.ffn = nn.Sequential(
#             nn.LayerNorm(D_FUSE),
#             nn.Linear(D_FUSE, 4 * D_FUSE),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(4 * D_FUSE, D_FUSE),
#         )

#         # 4) Readout
#         self.readout = nn.Sequential(
#             nn.LayerNorm(D_FUSE),
#             nn.Linear(D_FUSE, 256),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#             nn.Linear(256, 1),
#         )

#     def _maybe_drop(self, names: List[str]) -> List[str]:
#         if not self.training or self.modality_drop_p <= 0.0: return names
#         kept = [n for n in names if torch.rand(()) > self.modality_drop_p]
#         if not kept: kept = [names[torch.randint(len(names), (1,)).item()]]  # keep at least one
#         return kept

#     def forward(
#         self,
#         z_att: Optional[torch.Tensor],  # [B, D_ATT]
#         z_conf: Optional[torch.Tensor], # [B, D_CONF]
#         z_hid: Optional[torch.Tensor],  # [B, D_HID]
#         return_diag: bool = False,
#     ):
#         tokens = []
#         names = []
#         if self.use_attn and z_att is not None: tokens.append(("attn", z_att))
#         if self.use_conf and z_conf is not None: tokens.append(("conf", z_conf))
#         if self.use_hid  and z_hid is not None: tokens.append(("hid",  z_hid))
#         if not tokens:
#             raise ValueError("No modality provided.")

#         # Randomly drop some modalities at train time to prevent learned ignoring
#         names_all = [n for n, _ in tokens]
#         names_kept = self._maybe_drop(names_all)
#         tokens = [(n, x) for (n, x) in tokens if n in names_kept]

#         # Adapt each kept modality to shared space and stack as tokens [B, M, D_FUSE]
#         adapted = [self.adapters[n](x).unsqueeze(1) for n, x in tokens]
#         X = torch.cat(adapted, dim=1)  # [B, M, D_FUSE], M in [1..3]

#         # Queries attend to modality tokens -> per-input attention weights
#         B = X.size(0)
#         Q = self.queries.expand(B, -1, -1)  # [B, n_queries, D_FUSE]
#         out, attn = self.mha(Q, X, X, need_weights=True, average_attn_weights=False)
#         # out: [B, n_queries, D_FUSE], attn: [B, n_heads, n_queries, M]

#         # Residual + FFN on queries (tiny Transformer block)
#         out = out + self.ffn(out)  # [B, n_queries, D_FUSE]

#         # Pool queries (mean) -> fused vector
#         h = out.mean(dim=1)  # [B, D_FUSE]

#         logit = self.readout(h)  # [B, 1]

#         if return_diag:
#             # Average attention over heads for readability
#             attn_avg = attn.mean(dim=1)  # [B, n_queries, M]
#             return {"logit": logit, "attn_weights": attn_avg, "kept_modalities": names_kept}
#         return logit


# # ---------- shared pieces ----------
# class _Adapter(nn.Module):
#     """LN -> Linear -> GELU -> Dropout: maps modality vector to shared dim."""
#     def __init__(self, in_dim: int, out_dim: int, pdrop: float):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.Linear(in_dim, out_dim),
#             nn.GELU(),
#             nn.Dropout(pdrop),
#         )
#     def forward(self, x): return self.net(x)

# class _MLP(nn.Module):
#     def __init__(self, d_in: int, d_hidden: int, d_out: int, pdrop: float):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(d_in),
#             nn.Linear(d_in, d_hidden), nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(d_hidden, d_out)
#         )
#     def forward(self, x): return self.net(x)

# # =========================================================
# # B) Product-of-Experts (content-adaptive temperatures)
# # =========================================================
# class CorrectnessHead_V4(nn.Module):
#     """
#     - Per-modality logits via small MLPs
#     - A tiny gating MLP predicts a per-example temperature tau_m>0 for each modality
#     - Final logit = sum( logit_m / tau_m ), i.e., a weighted product-of-experts
#     - Ensures every modality has a direct gradient path; no fixed global ratio
#     """
#     def __init__(
#         self,
#         D_ATT=256, D_CONF=128, D_HID=256,
#         use_attn=True, use_conf=True, use_hid=True,
#         D_FUSE=256, pdrop=0.10,
#         min_tau=0.2, max_tau=5.0
#     ):
#         super().__init__()
#         assert use_attn or use_conf or use_hid, "Enable at least one modality"
#         self.use_attn, self.use_conf, self.use_hid = use_attn, use_conf, use_hid
#         self.min_tau, self.max_tau = float(min_tau), float(max_tau)

#         # Adapters
#         self.adapters = nn.ModuleDict()
#         if use_attn: self.adapters["attn"] = _Adapter(D_ATT,  D_FUSE, pdrop)
#         if use_conf: self.adapters["conf"] = _Adapter(D_CONF, D_FUSE, pdrop)
#         if use_hid:  self.adapters["hid"]  = _Adapter(D_HID,  D_FUSE, pdrop)

#         # Per-modality logit heads
#         self.logit_head = nn.ModuleDict()
#         if use_attn: self.logit_head["attn"] = _MLP(D_FUSE, D_FUSE, 1, pdrop)
#         if use_conf: self.logit_head["conf"] = _MLP(D_FUSE, D_FUSE, 1, pdrop)
#         if use_hid:  self.logit_head["hid"]  = _MLP(D_FUSE, D_FUSE, 1, pdrop)

#         # Temperature gate: takes concatenated adapted features -> tau for each present modality
#         in_dim = (int(use_attn) + int(use_conf) + int(use_hid)) * D_FUSE
#         self.tau_gate = nn.Sequential(
#             nn.LayerNorm(in_dim),
#             nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(pdrop),
#             nn.Linear(128, (int(use_attn) + int(use_conf) + int(use_hid)))
#         )

#     def forward(
#         self,
#         z_att: Optional[torch.Tensor],
#         z_conf: Optional[torch.Tensor],
#         z_hid: Optional[torch.Tensor],
#         return_diag: bool = False
#     ):
#         mods, vecs, logits = [], [], []

#         if self.use_attn and z_att is not None:
#             v = self.adapters["attn"](z_att); vecs.append(v); logits.append(self.logit_head["attn"](v)); mods.append("attn")
#         if self.use_conf and z_conf is not None:
#             v = self.adapters["conf"](z_conf); vecs.append(v); logits.append(self.logit_head["conf"](v)); mods.append("conf")
#         if self.use_hid and z_hid is not None:
#             v = self.adapters["hid"](z_hid);  vecs.append(v); logits.append(self.logit_head["hid"](v));  mods.append("hid")

#         if len(vecs) == 0:
#             raise ValueError("No modality provided.")

#         V = torch.cat(vecs, dim=-1)              # [B, M*D_FUSE]
#         taus_raw = self.tau_gate(V)              # [B, M]
#         taus = F.softplus(taus_raw) + self.min_tau
#         taus = torch.clamp(taus, max=self.max_tau)  # [B, M]

#         L = torch.cat(logits, dim=-1)            # [B, M]
#         logit = (L / taus).sum(dim=-1, keepdim=True)  # [B, 1]

#         if return_diag:
#             return {"logit": logit, "per_mod_logits": L, "taus": taus, "mods": mods}
#         return logit
