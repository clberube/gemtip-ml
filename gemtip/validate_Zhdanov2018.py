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
from scipy.spatial.transform import Rotation as R

from network import MLP
from emt import returnConducEff
from utilities import restore_minor_ticks_log_plot, mape
from integrals import integrandzhSurf


device = "cpu"
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

n_data = int(2)

model_name = "A"
fig_dir = "./figures"
wt_dir = "./weights"
data_dir = "./data"  # relatif à ce .py file

X_test = torch.ones(n_data, 4).float()
X_test[0, 0] = 1.0
X_test[0, 1] = 0.2
X_test[1, 0] = 0.2
X_test[1, 1] = 0.2

a_s = [1.0, 0.2]
b_s = [0.2, 1.0]

# Paramètres du réseau de neurones
model_params = {
    "input_dim": X_test.shape[-1],  # nombre de variables d'intégration
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

angles = np.array([[0, 0, 0], [0, 90, 0]])
rots = R.from_euler("xyz", angles, degrees=True)
rots = torch.from_numpy(rots.as_matrix()).float()
rots_T = torch.transpose(rots, 1, 2)
lambdal_hat = rots @ lambdal_hat @ rots_T
gammal_hat = rots @ gammal_hat @ rots_T


al = torch.tensor([1e-5, 1e-2])
fl = torch.tensor([0.15, 0.05])
sl = 1.0 / torch.tensor([0.1, 0.1])
Cl = torch.tensor([0.8, 0.8])
alphal = torch.tensor([0.5, 0.5])

s0 = torch.tensor(1) / 100

gammal_real = torch.empty(len(X_test), 3, 3)
lambdal_real = torch.empty(len(X_test), 3, 3)

for i in range(len(X_test)):
    a = a_s[i]
    b = b_s[i]

    if a < b:
        ecc = np.sqrt(1 - a**2 / b**2)
        gam = (1 - ecc**2) / ecc**3 * (np.arctanh(ecc) - ecc)
    elif b < a:
        ecc = np.sqrt(a**2 / b**2 - 1)
        gam = (1 + ecc**2) / ecc**3 * (ecc - np.arctan(ecc))
    gam_x = (1 - gam) / 2
    gam_y = (1 - gam) / 2
    gam_z = gam

    gammal = torch.tensor([gam_x, gam_y, gam_z]) / s0
    lambdal = integrandzhSurf(a, b, s0, N=int(1e6 - 1)) * min(a, b)
    lambdal = torch.from_numpy(lambdal)

    gammal_real[i] = torch.diag(s0 * gammal)
    lambdal_real[i] = torch.diag(s0 * lambdal)


Z_true = returnConducEff(
    lambdal_real,
    gammal_real,
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
labels_zhd = [
    rf"$\sigma_x$",
    rf"$\sigma_y$",
    rf"$\sigma_z$",
]
labels_mlp = [
    rf"$\hat{{\sigma}}_x$",
    rf"$\hat{{\sigma}}_y$",
    rf"$\hat{{\sigma}}_z$",
]
colors = ["0.0", "0.25", "0.7"]
kwargs_zhd = dict(
    marker="o",
    ms=4,
    mfc="none",
    mew=0.5,
    ls="none",
)
kwargs_mlp = dict(
    marker="x",
    ms=2,
    mew=0.5,
    ls="none",
)

ax = axs[0]
for i in [0, 2]:
    clr = colors[i]
    ax.plot(
        f,
        1000 * Z_true[i].real,
        label=labels_zhd[i],
        color=clr,
        **kwargs_zhd,
    )
    ax.plot(
        f,
        1000 * Z_hat[i].real,
        label=labels_mlp[i],
        color=clr,
        **kwargs_mlp,
    )
ax.set_ylabel(r"$\sigma'$ (mS/m)")
ax.legend(ncol=2, columnspacing=0.5, loc=0)

ax = axs[1]
for i in [0, 2]:
    clr = colors[i]
    ax.plot(
        f,
        1000 * Z_true[i].imag,
        label=labels_zhd[i],
        color=clr,
        **kwargs_zhd,
    )
    ax.plot(
        f,
        1000 * Z_hat[i].imag,
        label=labels_mlp[i],
        color=clr,
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
