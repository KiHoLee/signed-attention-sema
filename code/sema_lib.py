"""Shared library for the TSP manuscript
"Embedding Multiplexing Gain from Semantic Similarity via Signed User-Wise
Attention".

Everything the experiments need lives here: the shared-scene embedding
model, the EXACT signature-spread front end (the received signal is
simulated literally and despread, no expectation idealization), the
closed-form linear receivers (SR, LMMSE, OMA references), the learned
receivers (proposed signed attention SC, its softmax-constrained twin, the
projection-based attention receiver, the static trainable linear combiner),
the losses and metrics, and the closed-form MSE expressions of the paper.

Conventions (identical to the manuscript)
  * U users, d-dimensional unit-norm real embeddings e_u.
  * Shared-scene model: e_u = sqrt(1-a_u^2) p_u + a_u s with (s, p_1..p_U)
    orthonormal, so <e_u, e_v> = a_u a_v = beta_uv exactly; needs d >= U+1.
  * Signatures c_u in R^{Lc} (Lc = U chips) with Gram C^T C = B,
    C = chol(B)^T. Mask M_u = c_u (x) I_d in R^{Lc d x d}, so
    M_u^T M_v = beta_uv I_d EXACTLY. Transmit x_u = M_u e_u, ||x_u|| = 1.
  * Channel: r = sum_v h_v M_v e_v + n, n ~ N(0, sigma^2 I_{Lc d}),
    h_v = |CN(0,1)| after phase compensation, floored at H_FLOOR.
  * Per-user SNR rho = 1/(d sigma^2); per-coordinate SNR rho_c = rho/d.
  * Matched filter: t_u = M_u^T r / h_u = e_u + sum_v beta_uv (h_v/h_u) e_v
    + M_u^T n / h_u, with Cov of the despread noise sigma^2 B_uv/(h_u h_v) I.
  * OMA reference: the same chain with orthogonal signatures (C = I_U),
    i.e. each user alone on d of the Ud channel uses, same power.
Device: CUDA when available (AINS), else MPS/CPU; one flag for every script.
Seeds: every experiment seeds its own torch.Generator on DEVICE.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

import os
# CUDA when available (AINS); MPS lacks batched QR, so the Mac runs on CPU.
DEVICE = torch.device(os.environ.get("SEMA_DEVICE",
                      "cuda" if torch.cuda.is_available() else "cpu"))
DT = torch.float32

# ---------------------------------------------------------------- config
CFG = dict(
    U=4, d=64, tau=0.45,
    h_floor=0.2,                      # gains below this are declared outage
    snr_train=list(range(-8, 25, 2)),  # dB, training SNR set
    iters=20000, batch=256, lr=2e-3, lam=0.0,  # lam=0: plain cosine loss
    hidden=64, feat="norm", use_h=False, tail=0.0, kind="lin",
)


def gen(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEVICE)
    g.manual_seed(int(seed))
    return g


def snr_lin(snr_db: float) -> float:
    """Per-user SNR rho = 10^(snr/10) (unit embedding energy over block noise
    energy d sigma^2). The per-coordinate SNR equals rho as well."""
    return 10 ** (snr_db / 10.0)


def sigma2_of(snr_db: float, d: int) -> float:
    """Per-chip (= per-coordinate after despreading) noise variance 1/(d rho)."""
    return 1.0 / (d * snr_lin(snr_db))


def rho_of(snr_db: float, d: int) -> float:
    """Inverse noise variance d*rho (kept for the verification script)."""
    return d * snr_lin(snr_db)


# ------------------------------------------------------- embedding model
def sample_embeddings(batch: int, a: torch.Tensor, d: int, g: torch.Generator):
    """Shared-scene embeddings. a: (batch, U) share coefficients in [0,1].
    Returns e (batch, U, d) with <e_u,e_v> = a_u a_v exactly, and s (batch, d).
    A fresh orthonormal (U+1)-frame is drawn per block (batched QR)."""
    U = a.shape[1]
    assert d >= U + 1, "the shared-scene model needs d >= U+1"
    G = torch.randn(batch, d, U + 1, generator=g, device=DEVICE, dtype=DT)
    Q, R = torch.linalg.qr(G)                     # (batch, d, U+1)
    Q = Q * torch.sign(torch.diagonal(R, dim1=1, dim2=2)).unsqueeze(1)
    s = Q[:, :, 0]                                # (batch, d)
    P = Q[:, :, 1:].transpose(1, 2)               # (batch, U, d)
    e = torch.sqrt(torch.clamp(1 - a ** 2, min=0)).unsqueeze(-1) * P \
        + a.unsqueeze(-1) * s.unsqueeze(1)
    e = e / e.norm(dim=-1, keepdim=True)
    return e, s


def sample_two_cluster(batch: int, U: int, d: int, a_hi: float, g: torch.Generator):
    """Out-of-distribution structure: two independent scenes, users split
    into two clusters sharing different scene vectors (B block structured).
    Needs d >= U+2."""
    G = torch.randn(batch, d, U + 2, generator=g, device=DEVICE, dtype=DT)
    Q, R = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diagonal(R, dim1=1, dim2=2)).unsqueeze(1)
    s1, s2 = Q[:, :, 0], Q[:, :, 1]
    P = Q[:, :, 2:].transpose(1, 2)
    half = U // 2
    e = torch.empty(batch, U, d, device=DEVICE, dtype=DT)
    for u in range(U):
        s = s1 if u < half else s2
        e[:, u] = math.sqrt(1 - a_hi ** 2) * P[:, u] + a_hi * s
    return e / e.norm(dim=-1, keepdim=True)


def gram_of(e: torch.Tensor) -> torch.Tensor:
    """Exact per-block Gram (batch, U, U)."""
    return torch.einsum('bud,bvd->buv', e, e)


def affinity_matrix(a: torch.Tensor) -> torch.Tensor:
    """B = diag(1-a^2) + a a^T, i.e. B_uv = a_u a_v, B_uu = 1. a: (..., U)."""
    B = a.unsqueeze(-1) * a.unsqueeze(-2)
    U = a.shape[-1]
    eye = torch.eye(U, device=a.device, dtype=a.dtype)
    return B * (1 - eye) + eye


# ------------------------------------------------ training-profile sampler
def sample_profile(batch: int, U: int, g: torch.Generator, mode: str = "mixed"):
    """Share-coefficient profiles. 'mixed' draws, with equal probability,
    coherent (a_u = c), independent (a_u i.i.d.), and clustered (a random
    subset of 2..U-1 users shares a high coefficient) profiles.
    'mixed_signed' additionally flips the sign of each coefficient with
    probability 1/2 in half of the blocks (negative similarities)."""
    if mode == "mixed_signed":
        a = sample_profile(batch, U, g, "mixed")
        flip = (torch.rand(batch, U, generator=g, device=DEVICE) < 0.5) & (torch.rand(batch, 1, generator=g, device=DEVICE) < 0.5)
        return torch.where(flip, -a, a)
    a = torch.empty(batch, U, device=DEVICE, dtype=DT)
    if mode == "coherent":
        a[:] = torch.rand(batch, 1, generator=g, device=DEVICE) * 0.95
        return a
    if mode == "independent":
        return torch.rand(batch, U, generator=g, device=DEVICE) * 0.95
    which = torch.randint(0, 3, (batch,), generator=g, device=DEVICE)
    coh = torch.rand(batch, 1, generator=g, device=DEVICE) * 0.95
    ind = torch.rand(batch, U, generator=g, device=DEVICE) * 0.95
    lo = torch.rand(batch, U, generator=g, device=DEVICE) * 0.3
    hi = 0.6 + torch.rand(batch, U, generator=g, device=DEVICE) * 0.35
    # random subset of size k in {2,...,U-1} (U>=3); for U=2 clustered==coherent
    if U >= 3:
        k = torch.randint(2, U, (batch,), generator=g, device=DEVICE)
        ranks = torch.rand(batch, U, generator=g, device=DEVICE).argsort(dim=1).argsort(dim=1)
        mask = ranks < k.unsqueeze(1)
        clu = torch.where(mask, hi, lo)
    else:
        clu = coh.expand(batch, U)
    a = torch.where(which.unsqueeze(1) == 0, coh.expand(batch, U), ind)
    a = torch.where(which.unsqueeze(1) == 2, clu, a)
    return a


# --------------------------------------------------- signature front end
def signatures(B: torch.Tensor) -> torch.Tensor:
    """C (…, Lc=U, U): columns c_u with C^T C = B (Cholesky, B must be PD;
    a tiny ridge keeps singular Grams factorizable)."""
    U = B.shape[-1]
    eye = torch.eye(U, device=B.device, dtype=B.dtype)
    L = torch.linalg.cholesky(B + 1e-9 * eye)     # B = L L^T
    return L.transpose(-1, -2)                    # C = L^T, C^T C = B


def sample_gains(batch: int, U: int, g: torch.Generator, fading: bool = True,
                 h_floor: float = CFG["h_floor"], offsets_db=None):
    if fading:
        hr = torch.randn(batch, U, 2, generator=g, device=DEVICE, dtype=DT) / math.sqrt(2)
        h = hr.norm(dim=-1).clamp(min=h_floor)
    else:
        h = torch.ones(batch, U, device=DEVICE, dtype=DT)
    if offsets_db is not None:                     # near-far mean offsets
        off = torch.as_tensor(offsets_db, device=DEVICE, dtype=DT)
        h = h * (10 ** (off / 20.0)).unsqueeze(0)
    return h


def transmit_receive(e: torch.Tensor, C: torch.Tensor, h: torch.Tensor,
                     snr_db: float, g: torch.Generator):
    """Literal physical chain. e (batch,U,d), C (U,U) or (batch,U,U) with
    columns c_u, h (batch,U). Returns the despread outputs t (batch,U,d).
    r = sum_v h_v c_v e_v^T  (Lc x d chips), noise N(0, sigma^2) per chip,
    t_u = c_u^T r / h_u."""
    batch, U, d = e.shape
    if C.dim() == 2:
        C = C.unsqueeze(0).expand(batch, U, U)
    sigma = math.sqrt(sigma2_of(snr_db, d))
    # r[b, l, :] = sum_v h[b,v] C[b,l,v] e[b,v,:]
    r = torch.einsum('bv,blv,bvd->bld', h, C, e)
    r = r + sigma * torch.randn(r.shape, generator=g, device=DEVICE, dtype=DT)
    t = torch.einsum('blu,bld->bud', C, r) / h.unsqueeze(-1)
    return t


def transmit_receive_reduced(e, B, h, snr_db, g):
    """Equivalent reduced model t_u = e_u + sum beta (h_v/h_u) e_v + n~,
    with Cov(n~_u, n~_v) = sigma^2 B_uv/(h_u h_v) I. Used only to verify the
    physical chain (exp0_verify.py); the experiments use transmit_receive."""
    batch, U, d = e.shape
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, U, U)
    sigma = math.sqrt(sigma2_of(snr_db, d))
    Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))   # Gamma_uv = B_uv h_v/h_u
    t = torch.einsum('buv,bvd->bud', Gam, e)
    L = torch.linalg.cholesky(B + 1e-9 * torch.eye(U, device=DEVICE))
    xi = sigma * torch.randn(batch, U, d, generator=g, device=DEVICE, dtype=DT)
    t = t + torch.einsum('buv,bvd->bud', L, xi) / h.unsqueeze(-1)
    return t


# ----------------------------------------------------- linear receivers
def _nrm(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def rx_mf(t):
    """Single-user matched filter, normalized (identity combining)."""
    return _nrm(t)


def rx_sr(t, B, h):
    """Similarity-rejecting (zero-forcing) receiver: (Gamma^{-1} (x) I) t."""
    batch, U, d = t.shape
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, U, U)
    Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))
    out = torch.linalg.solve(Gam, t)              # (batch,U,d)
    return _nrm(out)


def lmmse_weights(B, h, snr_db, d, Bprior=None):
    """W = C Gamma^T (Gamma C Gamma^T + Cn)^{-1}, C = Bprior/d (Bprior=B by
    default), Cn = sigma^2 H^{-1} B H^{-1}. B (batch,U,U), h (batch,U)."""
    batch, U = h.shape
    sigma2 = sigma2_of(snr_db, d)
    Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))
    Cx = (B if Bprior is None else Bprior) / d
    Cn = sigma2 * B / (h.unsqueeze(1) * h.unsqueeze(2))
    S = Gam @ Cx @ Gam.transpose(1, 2) + Cn
    W = torch.linalg.solve(S.transpose(1, 2), (Cx @ Gam.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
    return W


def rx_lmmse(t, B, h, snr_db, Bprior=None):
    batch, U, d = t.shape
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, U, U)
    if Bprior is not None and Bprior.dim() == 2:
        Bprior = Bprior.unsqueeze(0).expand(batch, U, U)
    W = lmmse_weights(B, h, snr_db, d, Bprior)
    return _nrm(torch.einsum('buv,bvd->bud', W, t)), W


def rx_lmmse_mom(t, Bsig, h):
    """Moment-matched LMMSE without side information: the output Gram
    T = t t^T/d estimates S = Gam Cx Gam^T + Cn (Lemma 2). With the
    signature Gram and the gains known, Cx = Gam^{-1}(T - Cn)Gam^{-T}; the
    unit diagonal of the prior (d [Cx]_uu = 1) fixes sigma^2 by least
    squares, Cx is symmetrized, its diagonal is set to 1/d, its off-diagonal
    entries are clipped to [-1/d, 1/d], and negative eigenvalues are removed.
    Then the standard LMMSE filter is applied with the estimates."""
    batch, U, d = t.shape
    if Bsig.dim() == 2:
        Bsig = Bsig.unsqueeze(0).expand(batch, U, U)
    T = torch.einsum('bud,bvd->buv', t, t) / d
    Gam = Bsig * (h.unsqueeze(1) / h.unsqueeze(2))
    Gi = torch.linalg.inv(Gam)
    A = Gi @ T @ Gi.transpose(1, 2)
    Bi = torch.linalg.inv(Bsig)
    D = Bi / (h.unsqueeze(1) * h.unsqueeze(2))            # H^{-1} Bsig^{-1} H^{-1}
    dA = torch.diagonal(A, dim1=1, dim2=2); dD = torch.diagonal(D, dim1=1, dim2=2)
    sigma2 = ((dA - 1.0 / d) * dD).sum(1) / (dD * dD).sum(1)
    sigma2 = sigma2.clamp(min=1e-9)
    Cx = A - sigma2.view(-1, 1, 1) * D
    Cx = 0.5 * (Cx + Cx.transpose(1, 2))
    Bhat = (d * Cx).clamp(-1.0, 1.0)
    eye = torch.eye(U, device=t.device, dtype=t.dtype)
    Bhat = Bhat * (1 - eye) + eye
    # PSD projection of the tiny U x U estimates on the CPU (batched eigh on
    # CUDA reserves gigabytes of workspace for 25,000 4 x 4 matrices)
    ev, V = torch.linalg.eigh(Bhat.cpu())
    Bhat = (V @ torch.diag_embed(ev.clamp(min=1e-4)) @ V.transpose(1, 2)).to(t.device)
    Cn = sigma2.view(-1, 1, 1) * Bsig / (h.unsqueeze(1) * h.unsqueeze(2))
    Cxh = Bhat / d
    S = Gam @ Cxh @ Gam.transpose(1, 2) + Cn
    W = torch.linalg.solve(S.transpose(1, 2), (Cxh @ Gam.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
    return _nrm(torch.einsum('buv,bvd->bud', W, t)), W


def rx_lmmse_empirical(t, Cx_full, B, h, snr_db):
    """LMMSE with an empirical (Ud x Ud) joint prior (real data). Batched."""
    batch, U, d = t.shape
    sigma2 = sigma2_of(snr_db, d)
    Id = torch.eye(d, device=DEVICE, dtype=DT)
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, U, U)
    Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))
    Cn = sigma2 * B / (h.unsqueeze(1) * h.unsqueeze(2))
    G = torch.kron(Gam[0], Id).unsqueeze(0) if batch == 1 else \
        torch.stack([torch.kron(Gam[b], Id) for b in range(batch)])
    Cnf = torch.stack([torch.kron(Cn[b], Id) for b in range(batch)])
    S = G @ Cx_full @ G.transpose(1, 2) + Cnf
    rhs = (Cx_full @ G.transpose(1, 2)).transpose(1, 2)
    W = torch.linalg.solve(S.transpose(1, 2), rhs).transpose(1, 2)   # (batch,Ud,Ud)
    out = (W @ t.reshape(batch, U * d, 1)).reshape(batch, U, d)
    return _nrm(out)


# ----------------------------------------------------- learned receivers
class GramSC(nn.Module):
    """Proposed similarity-combining receiver (signed user-wise attention).
    T = t t^T (U x U, summed over coordinates);  W(T) = I + f_theta(T);
    e_hat_u = nrm(sum_i W_ui t_i).
    Input features of the score network:
      feat="raw"  : the U^2 entries of T
      feat="norm" : the U^2 entries of the correlation form T_uv/sqrt(T_uu T_vv)
                    and the U values log T_uu (scale-free similarity pattern
                    plus per-user output energies)
    use_h=True appends log h_u (the front-end gains the matched filter
    already uses). softmax=True gives the convex twin: W(T) = softmax_rows
    (f_theta), identical architecture, weights confined to the simplex."""
    def __init__(self, U: int, hidden: int = 64, softmax: bool = False,
                 feat: str = "raw", use_h: bool = False, use_rho: bool = False, K: int = 1000):
        super().__init__()
        self.U, self.softmax, self.feat, self.use_h = U, softmax, feat, use_h
        self.use_rho, self.K = use_rho, K    # SC-rho: one extra input, log of a long-term noise estimate
        n_in = U * U + (U if feat == "norm" else 0) + (U if use_h else 0) + (1 if use_rho else 0)
        self.score = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, U * U))
        with torch.no_grad():
            self.score[-1].weight.mul_(0.1)
            self.score[-1].bias.zero_()
            if softmax:   # start near the identity: large diagonal logits
                self.score[-1].bias.copy_(4.0 * torch.eye(U).reshape(-1))

    def features(self, t, h=None, rho_inv=None):
        """The score network reads the Gram T = t t^T (sum over the d
        coordinates, not the mean), whose entries are free of d:
        E[T] = Gam B Gam^T + rho^{-1} H^{-1} Bs H^{-1}.
        rho_inv (batch,) is the long-term estimate of 1/rho (SC-rho only)."""
        B_, U, d = t.shape
        T = torch.einsum('bud,bvd->buv', t, t)
        if self.feat == "norm":
            dg = torch.diagonal(T, dim1=1, dim2=2).clamp(min=1e-6)
            corr = T / torch.sqrt(dg.unsqueeze(2) * dg.unsqueeze(1))
            f = [corr.reshape(B_, U * U), torch.log(dg)]
        else:
            f = [T.reshape(B_, U * U)]
        if self.use_h:
            assert h is not None, "this receiver reads the front-end gains"
            f.append(torch.log(h))
        if self.use_rho:
            assert rho_inv is not None, "this receiver reads a long-term noise estimate"
            f.append(torch.log(rho_inv.clamp(min=1e-6)).reshape(B_, 1))
        return torch.cat(f, dim=1)

    def weights(self, t, h=None, rho_inv=None):
        B_, U, d = t.shape
        W = self.score(self.features(t, h, rho_inv)).reshape(B_, U, U)
        if self.softmax:
            return torch.softmax(W, dim=2)
        return W + torch.eye(U, device=t.device, dtype=t.dtype)

    def forward(self, t, h=None, rho_inv=None):
        W = self.weights(t, h, rho_inv)
        return _nrm(torch.einsum('bui,bid->bud', W, t))


def noise_estimate(t, B, h):
    """Per-block estimate of 1/rho from front-end knowledge only: the SR
    output Gamma^{-1} t_u has energy 1 + rho^{-1} [Bs^{-1}]_uu / h_u^2 in
    expectation (unit-norm embeddings), so h_u^2 (||.||^2 - 1) / [Bs^{-1}]_uu
    averaged over users (and, by the caller, over many blocks) estimates
    1/rho. Returns (batch,)."""
    batch, U, d = t.shape
    if B.dim() == 2:
        B = B.unsqueeze(0).expand(batch, U, U)
    Gam = B * (h.unsqueeze(1) / h.unsqueeze(2))
    s_sr = torch.linalg.solve(Gam, t)
    binv = torch.diagonal(torch.linalg.inv(B), dim1=1, dim2=2)
    return (((s_sr ** 2).sum(-1) - 1.0) * h ** 2 / binv).mean(dim=1)


class ProjAttention(nn.Module):
    """Projection-based user-wise cross-attention receiver (adapting the
    attention decoder principle of the shared-embedding scheme to the
    matched-filter outputs): per-user queries, shared key/value/output
    projections, softmax over users, residual to the own output."""
    def __init__(self, U: int, d: int, d_k: int = 32, heads: int = 4):
        super().__init__()
        self.U, self.d, self.dk, self.H = U, d, d_k, heads
        self.W_K = nn.Linear(d, d_k * heads, bias=False)
        self.W_V = nn.Linear(d, d_k * heads, bias=False)
        self.W_O = nn.Linear(d_k * heads, d, bias=False)
        self.q = nn.Parameter(torch.randn(U, d_k * heads) * 0.1)
        self.log_eta = nn.Parameter(torch.zeros(()))

    def forward(self, t):
        B_, U, d = t.shape
        K = self.W_K(t).view(B_, U, self.H, self.dk)
        V = self.W_V(t).view(B_, U, self.H, self.dk)
        q = self.q.view(U, self.H, self.dk)
        s = torch.einsum('uhk,bihk->buih', q, K) / math.sqrt(self.dk) * torch.exp(self.log_eta)
        al = torch.softmax(s, dim=2)
        ctx = torch.einsum('buih,bihk->buhk', al, V).reshape(B_, U, self.H * self.dk)
        return _nrm(self.W_O(ctx) + t)


class TLC(nn.Module):
    """Static trainable linear combiner: one U x U matrix for every block."""
    def __init__(self, U: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(U))

    def forward(self, t):
        return _nrm(torch.einsum('ui,bid->bud', self.W, t))


class TLCg(nn.Module):
    """Gain-scaled static combiner (TLC-g): one U x U matrix A applied to the
    gain-scaled outputs s_u = h_u t_u, result divided by h_u. A = B^{-1}
    realizes the SR receiver for every gain realization, so the gain-induced
    floor of the plain TLC cannot occur."""
    def __init__(self, U: int):
        super().__init__()
        self.A = nn.Parameter(torch.eye(U))

    def forward(self, t, h):
        s = t * h.unsqueeze(-1)
        return _nrm(torch.einsum('ui,bid->bud', self.A, s) / h.unsqueeze(-1))


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------- loss and metrics
def cosine_loss(e_hat, e, lam: float = 0.0, tail: float = 0.0,
                tail_tau: float = 0.55, tail_s: float = 0.05, kind: str = "lin"):
    """Mean cosine distortion 1 - <e_hat, e>; optional decorrelation
    penalty lam * mean_{u != v} |<e_hat_u, e_hat_v>| (ablation only) and an
    optional tail term tail * mean softplus((tail_tau - cos)/tail_s) that
    penalizes recoveries near the SER threshold."""
    cos = (e_hat * e).sum(-1)
    if kind == "log":      # log distortion: equal relative weight at every SNR
        loss = torch.log(1 - cos + 1e-3).mean()
    else:
        loss = (1 - cos).mean()
    if tail > 0:
        loss = loss + tail * torch.nn.functional.softplus((tail_tau - cos) / tail_s).mean() * tail_s
    if lam > 0:
        U = e_hat.shape[1]
        R = torch.einsum('bud,bvd->buv', e_hat, e_hat)
        off = R - torch.diag_embed(torch.diagonal(R, dim1=1, dim2=2))
        loss = loss + lam * off.abs().sum() / (e_hat.shape[0] * U * (U - 1))
    return loss


def metrics(e_hat, e, tau: float = CFG["tau"], s=None):
    """cos mean, SER at tau, NMSE of the unit-norm estimate, and (optional)
    private cosine (cosine after projecting out the scene direction s)."""
    cos = (e_hat * e).sum(-1)
    out = dict(cos=cos.mean().item(), ser=(cos < tau).float().mean().item(),
               nmse=((e_hat - e) ** 2).sum(-1).mean().item(),
               n=cos.numel())
    if s is not None:
        proj = lambda x: x - (x * s.unsqueeze(1)).sum(-1, keepdim=True) * s.unsqueeze(1)
        ph, pe = proj(e_hat), proj(e)
        out["pcos"] = ((ph * pe).sum(-1) / (ph.norm(dim=-1) * pe.norm(dim=-1) + 1e-12)).mean().item()
    return out


def ser_counts(e_hat, e, tau: float):
    cos = (e_hat * e).sum(-1)
    return int((cos < tau).sum().item()), int(cos.numel())


# --------------------------------------------------- closed-form theory
def eig_sym(U: int, beta: float):
    """Eigenvalues of B = (1-beta) I + beta 1 1^T."""
    return 1 + (U - 1) * beta, 1 - beta


def mse_sr_sym(U, beta_s, rho):
    """Per-user MSE of the SR receiver, (1/rho)[B_s^{-1}]_uu, symmetric
    signature Gram with correlation beta_s, h = 1, rho = per-user SNR."""
    m1, m2 = eig_sym(U, beta_s)
    return (1.0 / U) * (1 / m1 + (U - 1) / m2) / rho


def mse_lmmse_sym(U, beta, rho, orthogonal_sig=False, beta_s=None):
    """Per-user MSE of the LMMSE receiver (prior B/d), symmetric model, h=1.
    Embedding Gram eigenvalues lambda_k (correlation beta), signature Gram
    eigenvalues mu_k (correlation beta_s; beta_s = beta when None, mu_k = 1
    for orthogonal signatures). Per mode the relative MSE is
    1/(1 + mu_k lambda_k rho), so
        MSE = (1/U) sum_k m_k lambda_k / (1 + mu_k lambda_k rho)."""
    l1, l2 = eig_sym(U, beta)
    if orthogonal_sig:
        m1 = m2 = 1.0
    else:
        m1, m2 = eig_sym(U, beta if beta_s is None else beta_s)
    return (1.0 / U) * (l1 / (1 + m1 * l1 * rho) + (U - 1) * l2 / (1 + m2 * l2 * rho))


def beta_s_opt(U, beta, rho):
    """Optimal symmetric signature correlation (Prop. 3):
    beta_s* = beta / (lambda_1 lambda_2 rho), clipped to [0, 1]."""
    l1, l2 = eig_sym(U, beta)
    return min(1.0, max(0.0, beta / (l1 * l2 * rho)))


def mse_lmmse_general(B: torch.Tensor, h: torch.Tensor, rho: float, d: int, Bprior=None):
    """Per-user MSE d*[E]_uu of the LMMSE receiver for a general signature
    Gram B, gains h, embedding prior Bprior/d (Bprior = B when None):
    E = Cx - Cx Gam^T (Gam Cx Gam^T + Cn)^{-1} Gam Cx."""
    U = B.shape[-1]
    sigma2 = 1.0 / (rho * d)
    Gam = B * (h.unsqueeze(0) / h.unsqueeze(1))
    Cx = (B if Bprior is None else Bprior) / d
    Cn = sigma2 * B / (h.unsqueeze(0) * h.unsqueeze(1))
    S = Gam @ Cx @ Gam.T + Cn
    E = Cx - Cx @ Gam.T @ torch.linalg.solve(S, Gam @ Cx)
    return d * torch.diagonal(E)


def mse_sr_general(B, h, rho, d):
    U = B.shape[-1]
    sigma2 = 1.0 / (rho * d)
    Binv = torch.linalg.inv(B)
    return d * sigma2 * torch.diagonal(Binv) / h ** 2


def snr_boundary_sym(U, beta):
    """Combine-or-reject boundary (Prop. 2): with matched signatures the
    LMMSE off-diagonal weight is positive iff rho < 1/(lambda_1 lambda_2).
    Returns the boundary per-user SNR in dB."""
    l1, l2 = eig_sym(U, beta)
    return 10 * math.log10(1.0 / (l1 * l2))


# ------------------------------------------------------------- training
def train_receiver(model: nn.Module, U: int, d: int, iters: int, batch: int,
                   lr: float, seed: int, lam: float = 0.0, snr_set=None,
                   profile: str = "mixed", log=print, log_every=500,
                   data_fn=None, tail: float = 0.0, kind: str = "lin"):
    """Train any receiver on the synthetic model through the physical chain.
    data_fn(batch, g) -> (e, B, h, snr_db) overrides the sampler (real data,
    block-varying profiles)."""
    snr_set = snr_set or CFG["snr_train"]
    g = gen(seed)
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for it in range(iters):
        if data_fn is None:
            a = sample_profile(batch, U, g, profile)
            e, _ = sample_embeddings(batch, a, d, g)
            B = affinity_matrix(a)
            h = sample_gains(batch, U, g)
            snr = snr_set[torch.randint(0, len(snr_set), (1,), generator=g, device=DEVICE).item()]
        else:
            e, B, h, snr = data_fn(batch, g)
        t = transmit_receive(e, signatures(B), h, snr, g)
        if isinstance(model, GramSC) and model.use_rho:
            # long-term noise estimate from a separate draw of K blocks at this SNR
            a2 = sample_profile(model.K, U, g, profile) if data_fn is None else None
            if a2 is not None:
                e2, _ = sample_embeddings(model.K, a2, d, g); B2 = affinity_matrix(a2); h2 = sample_gains(model.K, U, g)
            else:
                e2, B2, h2, _ = data_fn(model.K, g)
            t2 = transmit_receive(e2, signatures(B2), h2, snr, g)
            rho_inv = noise_estimate(t2, B2, h2).mean().expand(e.shape[0])
            out = model(t, h, rho_inv)
        elif isinstance(model, (GramSC, TLCg)):
            out = model(t, h)
        else:
            out = model(t)
        loss = cosine_loss(out, e, lam, tail, kind=kind)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and (it % log_every == 0 or it == iters - 1):
            log(f"  it {it:5d}/{iters} loss {loss.item():.4f}")
    model.eval()
    return model
