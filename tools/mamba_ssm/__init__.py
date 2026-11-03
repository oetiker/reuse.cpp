# Plain-PyTorch stand-in for mamba_ssm.Mamba (v1, mamba_simple slow path + selective_scan_ref).
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Mamba"]


def selective_scan_ref(u, delta, A, B, C, D, z, delta_bias):
    # u, delta, z: [b, d, l]; A: [d, n]; B, C: [b, n, l]; D: [d]
    delta = F.softplus(delta + delta_bias[None, :, None])
    dA = torch.exp(torch.einsum("bdl,dn->bldn", delta, A))          # [b, l, d, n]
    dBu = torch.einsum("bdl,bnl,bdl->bldn", delta, B, u)             # [b, l, d, n]
    Ct = C.transpose(1, 2)                                           # [b, l, n]
    bsz, d, L = u.shape
    h = u.new_zeros(bsz, d, A.shape[1])
    ys = []
    for i in range(L):
        h = dA[:, i] * h + dBu[:, i]
        ys.append((h * Ct[:, i, None, :]).sum(-1))
    y = torch.stack(ys, dim=2)                                       # [b, d, l]
    y = y + u * D[None, :, None]
    return y * F.silu(z)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.zeros(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, hidden_states):
        bsz, L, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)             # [b, 2d, l]
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :L])
        x_dbl = self.x_proj(x.transpose(1, 2))                       # [b, l, r+2n]
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ self.dt_proj.weight.t()).transpose(1, 2)          # [b, d, l], bias added in scan
        A = -torch.exp(self.A_log.float())
        y = selective_scan_ref(x, dt, A, Bm.transpose(1, 2), Cm.transpose(1, 2),
                               self.D.float(), z, self.dt_proj.bias.float())
        return self.out_proj(y.transpose(1, 2))
