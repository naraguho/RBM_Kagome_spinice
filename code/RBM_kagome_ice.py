#!/usr/bin/env python3
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import random

# ============================================================
# Reproducibility
# ============================================================
def seed_everything(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deterministic (can be slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)


# ============================================================
# RBM (binary v,h in {0,1}). Data processing {-1,1}->{0,1} expected.
# ============================================================
class RBM(nn.Module):
    def __init__(self, n_visible: int, n_hidden: int,
                 use_visible_bias=True, use_hidden_bias=True, device="cpu"):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.device = torch.device(device)

        self.W  = nn.Parameter(0.01 * torch.randn(n_visible, n_hidden, device=self.device))
        self.bv = nn.Parameter(torch.zeros(n_visible, device=self.device), requires_grad=use_visible_bias)
        self.bh = nn.Parameter(torch.zeros(n_hidden, device=self.device), requires_grad=use_hidden_bias)

    @staticmethod
    def _bernoulli_sample(probs: torch.Tensor) -> torch.Tensor:
        return torch.bernoulli(probs)

    def p_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(v @ self.W + self.bh)

    def p_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(h @ self.W.t() + self.bv)

    def gibbs_vhv(self, v0: torch.Tensor, k: int = 1):
        v = v0
        for _ in range(k):
            ph = self.p_h_given_v(v)
            h  = self._bernoulli_sample(ph)
            pv = self.p_v_given_h(h)
            v  = self._bernoulli_sample(pv)
        phk = self.p_h_given_v(v)
        return v, phk

    @torch.no_grad()
    def sample(self, n_samples: int, k_between: int = 1) -> torch.Tensor:
        """
        Draw samples from the model via Gibbs sampling starting from random init.
        No burn-in is used.
        """
        v = torch.bernoulli(0.5 * torch.ones(n_samples, self.n_visible, device=self.device))
        if k_between > 0:
            v, _ = self.gibbs_vhv(v, k=k_between)
        return v.clone()


def binary01_to_ising_pm1(v01: torch.Tensor) -> torch.Tensor:
    return (2.0 * v01 - 1.0)


# ============================================================
# Unit-cell spin-sum distribution (optional diagnostic)
# ============================================================
@torch.no_grad()
def unit_cell_sum_distribution_batch(spins_pm1: torch.Tensor, Lx: int, Ly: int):
    Nsamp = spins_pm1.shape[0]
    v = spins_pm1.view(Nsamp, Ly, Lx, 3)
    S = v.sum(dim=3)  # values in {-3,-1,1,3}
    counts = {}
    for val in (-3, -1, 1, 3):
        counts[val] = int((S == val).sum().item())
    return counts


def print_cell_distribution(counts, title="Unit-cell spin sum"):
    tot = sum(counts.values())
    print(f"\n[{title}] total cells counted = {tot}")
    for k in (-3, -1, 1, 3):
        c = counts.get(k, 0)
        print(f"  S={k:>2d}: {c}  (frac {c/tot:.6f})")


# ============================================================
# Save RBM
# ============================================================
def save_rbm(path_prefix: str, rbm: RBM, Lx: int, Ly: int, extra: dict | None = None):
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    torch.save(rbm.state_dict(), path_prefix + ".pt")

    meta = {
        "n_visible": rbm.n_visible,
        "n_hidden": rbm.n_hidden,
        "Lx": Lx,
        "Ly": Ly,
        "use_visible_bias": rbm.bv.requires_grad,
        "use_hidden_bias": rbm.bh.requires_grad,
        "dtype": str(rbm.W.dtype),
        "device": str(rbm.device),
    }
    if extra:
        meta.update(extra)

    with open(path_prefix + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved RBM to:\n  {path_prefix}.pt\n  {path_prefix}.json")


# ============================================================
# TRAIN: PCD_k + 2-stage LR + diagnostics + save checkpoint at chosen epoch
# ============================================================
def train_rbm_pcd_k(
    rbm: RBM,
    data_v: torch.Tensor,
    Lx: int,
    Ly: int,
    batch_size: int = 128,
    epochs_total: int = 2000,
    k: int = 1,

    lr_stage1: float = 1e-4,
    lr_stage2: float = 1e-5,
    lr_switch_epoch: int = 1000,  # switch at epoch lr_switch_epoch+1

    weight_decay: float = 0.0,

    diag_every: int = 100,
    diag_n_samples: int = 100,
    diag_k_between: int = 10,

    save_epoch: int | None = None,
    save_prefix_at_epoch: str | None = None,  # prefix without extension
):
    rbm.train()

    loader = DataLoader(
        TensorDataset(data_v),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,   # IMPORTANT for fixed-size persistent buffer
    )

    opt = torch.optim.Adam(rbm.parameters(), lr=lr_stage1, weight_decay=weight_decay)

    # persistent chains (PCD): one persistent visible batch
    v_persist = torch.bernoulli(0.5 * torch.ones(batch_size, rbm.n_visible, device=rbm.device))

    for ep in range(1, epochs_total + 1):

        # LR switch
        if ep == lr_switch_epoch + 1:
            for pg in opt.param_groups:
                pg["lr"] = lr_stage2
            print(f"\n[LR] switched to lr={lr_stage2:.2e} at epoch {ep}\n")

        # --- training over minibatches ---
        for (v0,) in loader:
            v0 = v0.to(rbm.device)

            # positive phase
            ph0 = rbm.p_h_given_v(v0)
            pos_W = v0.t() @ ph0

            # negative phase (PCD_k). The MLE in training is achieved in k --> infty limit.
            vk, phk = rbm.gibbs_vhv(v_persist, k=k)
            v_persist = vk.detach()

            neg_W = vk.t() @ phk

            # manual grads (PCD estimator)
            rbm.W.grad = -(pos_W - neg_W) / v0.shape[0]
            if rbm.bv.requires_grad:
                rbm.bv.grad = -(v0 - vk).mean(dim=0)
            if rbm.bh.requires_grad:
                rbm.bh.grad = -(ph0 - phk).mean(dim=0)

            opt.step()
            opt.zero_grad(set_to_none=True)

        # --- diagnostics and/or save ---
        do_diag = ((ep % diag_every) == 0)
        do_save = (save_epoch is not None and ep == save_epoch)

        if do_diag or do_save:
            rbm.eval()
            with torch.no_grad():
                v_samp = rbm.sample(
                    n_samples=diag_n_samples,
                    k_between=diag_k_between
                )
                sigma_samp = binary01_to_ising_pm1(v_samp)
                counts = unit_cell_sum_distribution_batch(sigma_samp, Lx=Lx, Ly=Ly)
                print_cell_distribution(counts, title=f"RBM samples @ epoch {ep} (N={diag_n_samples})")

                if do_save:
                    if save_prefix_at_epoch is None:
                        raise ValueError("save_prefix_at_epoch must be set when save_epoch is used.")
                    save_rbm(
                        path_prefix=save_prefix_at_epoch,
                        rbm=rbm,
                        Lx=Lx, Ly=Ly,
                        extra={
                            "saved_epoch": ep,
                            "epochs_total": epochs_total,
                            "pcd_k": k,
                            "lr_stage1": lr_stage1,
                            "lr_stage2": lr_stage2,
                            "lr_switch_epoch": lr_switch_epoch,
                            "batch_size": batch_size,
                            "weight_decay": weight_decay,
                            "diag_every": diag_every,
                            "diag_n_samples": diag_n_samples,
                            "diag_k_between": diag_k_between,
                        }
                    )
            rbm.train()

    return rbm


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    seed_everything(20260212)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    Lx, Ly = 32, 32
    train_path = "Your directory/train_ice1_32x32.npy"
    X = np.load(train_path)
    data_v = torch.tensor(X, dtype=torch.float32)

    Ndata, Nv = data_v.shape
    assert Nv == 3 * Lx * Ly, f"Nv={Nv} but expected 3*Lx*Ly={3*Lx*Ly}"

    rbm = RBM(
        n_visible=Nv,
        n_hidden=5000,
        use_visible_bias=False, #true for ice2
        use_hidden_bias=False, #true for ice2
        device=device
    )

    #
    rbm = train_rbm_pcd_k(
        rbm,
        data_v,
        Lx=Lx, Ly=Ly,
        batch_size=128,
        epochs_total=2000,
        k=10,

        lr_stage1=1e-4,
        lr_stage2=1e-5,
        lr_switch_epoch=1000,

        weight_decay=0.0,

        diag_every=50,
        diag_n_samples=100,
        diag_k_between=10,

        save_epoch=2000,
        save_prefix_at_epoch="/Your directory/machine",
    )

