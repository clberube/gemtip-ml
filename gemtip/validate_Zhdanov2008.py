#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from network import MLP
from emt import returnConducEff
from utilities import mape
from plotlib import restore_minor_ticks_log_plot


device = "cpu"
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

n_data = int(2)

model_name = "A"
fig_dir = "./figures"
wt_dir = "./weights"
data_dir = "./data"  # relatif à ce .py file


X_test = torch.ones(n_data, 4).float()
X_test[:, 0] = 1.0
X_test[:, 1] = 1.0

# Paramètres du réseau de neurones
model_params = {
    "input_dim": 4,  # nombre de variables d'intégration
    "hidden_dim": 128,  # nombre de neurones dans la couche cachée (à tuner)
    "output_dim": 6,  # nombre de dimensions de f
    "n_hidden": 4,  # nombre de couches cachées
    "activation": nn.SiLU(),  # fonction d'activation
}

# Instanciation du réseau de neurones
weights = torch.load(f"{wt_dir}/weights-best.pt", weights_only=True)
model = MLP(**model_params)
model.load_state_dict(weights)
model.eval()
model.to(device)
X_test = X_test.to(device)

y_hat = model(X_test).detach().cpu()

f = torch.logspace(-3, 4, 30)
w = 2 * np.pi * f

lambdal_hat = torch.diag_embed(y_hat[:, 3:])
gammal_hat = torch.diag_embed(y_hat[:, :3])

al = torch.tensor([0.1e-3, 0.2e-3])
fl = torch.tensor([0.20, 0.15])
sl = 1.0 / torch.tensor([0.1, 0.001])
Cl = torch.tensor([0.8, 0.6])
alphal = torch.tensor([1, 0.01])

s0 = torch.tensor(1) / 100

lambdal_true = torch.diag((2 / 3) * torch.ones(3)).repeat(2, 1, 1)
gammal_true = torch.diag((1 / 3) * torch.ones(3)).repeat(2, 1, 1)

Z_true = returnConducEff(
    lambdal_true,
    gammal_true,
    al,
    fl,
    sl,
    Cl,
    alphal,
    s0,
    w,
)

Z_hat = returnConducEff(
    lambdal_hat,
    gammal_hat,
    al,
    fl,
    sl,
    Cl,
    alphal,
    s0,
    w,
)


Zxx_hat = Z_hat[0]
Zyy_hat = Z_hat[1]
Zzz_hat = Z_hat[2]

Zxx_true = Z_true[0]
Zyy_true = Z_true[1]
Zzz_true = Z_true[2]


fig, axs = plt.subplots(2, 1, sharex=True)
ax = axs[0]
kwargs_zhd = dict(
    color="k",
    marker="o",
    ms=4,
    mfc="w",
    mew=0.5,
    ls="none",
    label=r"$\sigma$",
)
kwargs_mlp = dict(
    color="k",
    marker="x",
    ms=2,
    mew=0.5,
    ls="none",
    label=r"$\hat{\sigma}$",
)
ax.plot(f, 1000 * torch.real(Zxx_true), **kwargs_zhd)
ax.plot(f, 1000 * torch.real(Zxx_hat), **kwargs_mlp)
ax.set_ylabel(r"$\sigma'$ (mS/m)")
ax.legend(loc=0)

ax = axs[1]
ax.plot(
    f,
    1000 * torch.imag(Zxx_true),
    **kwargs_zhd,
)
ax.plot(
    f,
    1000 * torch.imag(Zxx_hat),
    **kwargs_mlp,
)

ax.set_ylabel(r"$\sigma''$ (mS/m)")

ax.set_xscale("log")
ax.set_xlabel(r"$f$ (Hz)")
restore_minor_ticks_log_plot(ax, axis="x")

plt.tight_layout()
plt.show()

print(
    "MAPE (%):",
    f"{mape(Zxx_hat, Zxx_true).mean().item():.3f}", 
    "+/-", 
    f"{mape(Zxx_hat, Zxx_true).std().item():.3f}",
)
