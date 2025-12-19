"""
PyTorch modules for extracting high-level features from LLM attention patterns,
hidden states, and confidence scores.  (Mask-free variant)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, Sequence, Tuple, List, Dict


# ======================================================================================
# SECTION 1: UTILITIES & HELPERS
# ======================================================================================
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




# # =============================================================================
# # Set Transformer bits (MHA/MAB/PMA)
# # =============================================================================
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

class SAB(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, pdrop: float = 0.1, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([MAB(d_model, n_heads, pdrop) for _ in range(num_layers)])
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for mab in self.layers:
            X = mab(X, X)
        return X

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

class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        m = max(8, c // r)
        self.fc = nn.Sequential(nn.Linear(c, m), nn.GELU(), nn.Linear(m, c), nn.Sigmoid())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        g = self.fc(s).unsqueeze(-1).unsqueeze(-1)
        return x * g


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



# -------------------------------------------------------
# === AttnFeatureExtractor
# -------------------------------------------------------
# class AttnFeatureExtractorLite_D3(nn.Module):
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
#         feature_mode: str = "cnn",                   # "cnn" | "spectral" | "both"
#         stats_groups: Sequence[str] = ("spec13",),
#         spec_radii: Tuple[float, float, float] = (0.15, 0.35, 0.60),
#         band_widths: Tuple[Optional[int], Optional[int]] = (None, None),
#         use_spectral: Optional[bool] = None,
#     ):
#         super().__init__()

#         if use_spectral is not None:
#             feature_mode = "both" if use_spectral else "cnn"
#         if feature_mode not in {"cnn", "spectral", "both"}:
#             raise ValueError(f"Invalid feature_mode: {feature_mode}")

#         self.feature_mode = feature_mode
#         self.stats_groups = tuple(stats_groups)
#         self.spec_radii   = spec_radii
#         self.band_widths  = band_widths
#         self.d_grid       = d_grid

#         self.has_cnn   = feature_mode in {"cnn", "both"}
#         self.has_stats = feature_mode in {"spectral", "both"}

#         # Conditional CNN creation
#         self.cnn_stem = self.cnn_body = self.se = None
#         cnn_out_dim = 0
#         if self.has_cnn:
#             in_c = 3
#             self.cnn_stem = nn.Sequential(
#                 nn.Conv2d(in_c, cnn_channels[0], 3, 1, 1, bias=False),
#                 nn.GroupNorm(max(1, cnn_channels[0] // 8), cnn_channels[0]),
#                 nn.GELU()
#             )
#             self.cnn_body = nn.Sequential(
#                 ResNetBlock(cnn_channels[0], cnn_channels[1], stride=2),
#                 ResNetBlock(cnn_channels[1], cnn_channels[2], stride=2),
#             )
#             self.se = SEBlock(cnn_channels[-1])
#             cnn_out_dim = cnn_channels[-1] * 2

#         stats_dim = _groups_dim(self.stats_groups) if self.has_stats else 0
#         in_dim = cnn_out_dim + stats_dim
#         if in_dim <= 0:
#             raise ValueError("No input features selected")

#         self.proj_per_map = nn.Linear(in_dim, d_grid)
#         self.layer_emb = nn.Embedding(max_layers, d_grid)
#         self.head_emb  = nn.Embedding(max_heads, d_grid)
#         nn.init.normal_(self.layer_emb.weight, std=0.02)
#         nn.init.normal_(self.head_emb.weight,  std=0.02)

#         axial = []
#         for _ in range(grid_conv_layers):
#             axial += [
#                 nn.Conv2d(d_grid, d_grid, (1,3), padding=(0,1), groups=d_grid, bias=False),
#                 nn.GELU(),
#                 nn.Conv2d(d_grid, d_grid, (3,1), padding=(1,0), groups=d_grid, bias=False),
#                 nn.GELU(),
#                 nn.Conv2d(d_grid, d_grid, 1, bias=False),
#                 nn.GroupNorm(max(1, d_grid // 8), d_grid),
#             ]
#         self.grid_processor = nn.Sequential(*axial)

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
#         B, L, H, k, _ = attn.shape
#         T = L * H
#         device = attn.device
#         maps = attn.reshape(B * T, 1, k, k)
#         per_chunks: List[torch.Tensor] = []

#         if self.has_cnn:
#             coords = self._coord(B * T, k, device=maps.device, dtype=maps.dtype)
#             x_maps = torch.cat([maps, coords], dim=1)
#             z = self.cnn_stem(x_maps)
#             z = self.cnn_body(z)
#             z = self.se(z)
#             gavg = F.adaptive_avg_pool2d(z, 1).flatten(1)
#             gmax = F.adaptive_max_pool2d(z, 1).flatten(1)
#             per_chunks.append(torch.cat([gavg, gmax], dim=-1))

#         if self.has_stats:
#             stats_vec = compute_attn_stats(
#                 maps.to(torch.float32),
#                 groups=self.stats_groups,
#                 spec_radii=self.spec_radii,
#                 band_widths=self.band_widths,
#             )
#             per_chunks.append(stats_vec.to(per_chunks[0].dtype if per_chunks else maps.dtype))

#         per_map = per_chunks[0] if len(per_chunks) == 1 else torch.cat(per_chunks, dim=-1)
#         feats = self.proj_per_map(per_map)

#         tok = feats.view(B, L, H, self.d_grid)
#         tok = tok + self.layer_emb(torch.arange(L, device=device)).view(1, L, 1, -1) \
#                   + self.head_emb(torch.arange(H, device=device)).view(1, 1, H, -1)
#         grid = tok.permute(0, 3, 1, 2).contiguous()
#         grid = grid + self.grid_processor(grid)

#         pma_in = grid.flatten(2).transpose(1, 2)
#         pooled = self.pma(pma_in)
#         return self.out(pooled.flatten(1))

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
            in_c = 1
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

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        B, L, H, k, _ = attn.shape
        T = L * H
        device = attn.device
        maps = attn.reshape(B * T, 1, k, k)
        per_chunks: List[torch.Tensor] = []

        if self.has_cnn:
            z = self.cnn_stem(maps)
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


# -------------------------------------------------------
# === HiddenFeatureExtractor
# -------------------------------------------------------
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




# -------------------------------------------------------
# === ConfFeatureExtractor
# -------------------------------------------------------
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


# -------------------------------------------------------
# === Head
# -------------------------------------------------------
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
        if D == 0:
            raise ValueError("Enable at least one modality.")

        self.g_att = nn.Sequential(nn.LayerNorm(D_ATT), nn.Linear(D_ATT, 1)) if use_attn else None
        self.g_con = nn.Sequential(nn.LayerNorm(D_CONF), nn.Linear(D_CONF, 1)) if use_conf else None
        self.g_hid = nn.Sequential(nn.LayerNorm(D_HID),  nn.Linear(D_HID,  1)) if use_hid  else None

        self.ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 384), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(384, 128), nn.GELU(), nn.Dropout(pdrop),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        z_att: Optional[torch.Tensor],
        z_conf: Optional[torch.Tensor],
        z_hid: Optional[torch.Tensor],
        return_penultimate: bool = False,
    ):
        chunks, gates = [], []
        if self.use_attn:
            chunks.append(z_att)
            gates.append(self.g_att(z_att))
        if self.use_conf:
            chunks.append(z_conf)
            gates.append(self.g_con(z_conf))
        if self.use_hid:
            chunks.append(z_hid)
            gates.append(self.g_hid(z_hid))

        g = torch.softmax(torch.cat(gates, dim=-1), dim=-1)
        out_slices = [ch * g[:, i:i+1] for i, ch in enumerate(chunks)]
        x = torch.cat(out_slices, dim=-1)

        x_norm = self.ln(x)

        # Penultimate representation: input to final Linear(128 -> 1)
        h_last = self.mlp[:-1](x_norm)      # shape [B, 128]

        # Final logit
        logits = self.mlp[-1](h_last)       # shape [B, 1]

        if return_penultimate:
            return logits, h_last
        return logits
