#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

from network import MLP
from emt import returnConducRot
from plotlib import plot_conductivities


device = "cpu"
torch.set_default_device(device)
torch.set_default_dtype(torch.float32)

model_name = "D"
fig_dir = "./figures"
wt_dir = "./weights"
data_dir = "./data"  # relatif à ce .py file


df = pd.read_csv(
    "/Users/charles/Data/MLA-originals/Dataview Exports/55_Grain.csv",
    low_memory=False,
    skiprows=5,
)

minerals = ["Pyrite Ni", "Pyrrhotite Zn", "pentlandite", "Ti oxide", "Sphalerite"]
df = df[df["Mineral Name"].isin(minerals)]

df["Max Span Angle"] = df["Max Span Angle"] - 90 

kwargs = dict(bins=12, color="0.9")
fig, ax = plt.subplots(2, 2, figsize=(4.33333, 3.33333), sharey=True)
sns.histplot(df["Max Span Angle"], log_scale=False, ax=ax[1, 0], stat="count", **kwargs)
sns.histplot(1./df["Aspect Ratio"], log_scale=False, ax=ax[1, 1], stat="count", **kwargs)
sns.histplot(df["Length (MBR)"], log_scale=True, ax=ax[0, 0], stat="count", **kwargs)
sns.histplot(df["Breadth (MBR)"], log_scale=True, ax=ax[0, 1], stat="count", **kwargs)
ax[1, 0].set_xlabel(r"$\gamma$ ($^\circ$)")
ax[1, 1].set_xlabel(r"$b/a$")
ax[0, 0].set_xlabel(r"$a$ ($\mu$m)")
ax[0, 1].set_xlabel(r"$b$ ($\mu$m)")

plt.tight_layout()


n_data = len(df)

X_test = torch.ones(n_data, 4)
X_test[:, 0] = torch.from_numpy((df["Breadth (MBR)"] / df["Length (MBR)"]).values)
X_test[:, 1] = torch.empty(n_data).uniform_(X_test[:, 0].min(), 1.)
X_test[:, 2] = 3.511391e2 / 4.012058e2 

X_test = X_test.float()

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

f = torch.logspace(-2, 2, 50)
w = 2 * np.pi * f

lambdal_hat = torch.diag_embed(y_hat[:, 3:])
gammal_hat = torch.diag_embed(y_hat[:, :3])

alpha = torch.zeros(n_data)  # about the x axis
beta = torch.zeros(n_data)  # about the y axis
gamma = torch.from_numpy(df["Max Span Angle"].values).float()  # about the z axis

angles = torch.stack((alpha, beta, gamma), -1)
rots = R.from_euler("xyz", angles, degrees=True)
rots = torch.from_numpy(rots.as_matrix()).float()
rots_T = torch.transpose(rots, 1, 2)
lambdal_hat = rots @ lambdal_hat @ rots_T
gammal_hat = rots @ gammal_hat @ rots_T

al = 1e-6 * torch.from_numpy(df["Length (MBR)"].values).float()
min_abc = al * X_test[:, :2].min(1).values.cpu()

fl = 1.0*torch.from_numpy(df["Area%"].values / 100).float()
sl = torch.tensor([1e4, 1e4, 1e4]) * torch.eye(3).repeat(n_data, 1, 1)

Cl = torch.ones(n_data)
alphal = torch.ones(n_data).log_normal_(2, 2)

s0 = torch.tensor([1 / 2760, 1 / 3153.53, 1 / 2760]) * torch.eye(3)
min_xyz = torch.tensor(3.511391e2 / 4.012058e2)


Z = returnConducRot(
    lambdal_hat,
    gammal_hat,
    min_abc.unsqueeze(-1).unsqueeze(-1),
    min_xyz,
    fl,
    sl,
    Cl,
    alphal,
    s0,
    w,
)


sip_path = "/Users/charles/Data/SIP-Data-Berube2019/AVG_SIP-MLA12-K389055_stable.dat"
sip = pd.read_csv(sip_path)
sip = sip[sip["Freq (Hz)"] < f.max().item()]
geo_fact = 0.016225

sip[" Res (Ohm-m)"] *= geo_fact 
sip[" dRes (Ohm-m)"] *= geo_fact 
sip[" Phase (mrad)"] /= 1000
sip[" dPhase (mrad)"] /= 1000

kwargs = dict(color="k", marker="o", linestyle="")
fig, ax = plot_conductivities(f, Z, color='0.5')

sip["complex"] = (
    sip[" Res (Ohm-m)"] * np.exp(1j * sip[" Phase (mrad)"])
)

EI = np.sqrt(
    (
        (
            sip[" Res (Ohm-m)"]
            * np.cos(sip[" Phase (mrad)"])
            * sip[" dPhase (mrad)"]
        )
        ** 2
    )
    + (np.sin(sip[" Phase (mrad)"]) * sip[" dRes (Ohm-m)"]) ** 2
)
ER = np.sqrt(
    (
        (
            sip[" Res (Ohm-m)"]
            * np.sin(sip[" Phase (mrad)"])
            * sip[" dPhase (mrad)"]
        )
        ** 2
    )
    + (np.cos(sip[" Phase (mrad)"]) * sip[" dRes (Ohm-m)"]) ** 2
)

PEI = np.abs(EI / sip["complex"].values.imag)
PER = np.abs(ER / sip["complex"].values.real)

sip["complex"] = 1.0 / (sip["complex"])


ax[0].errorbar(
    sip["Freq (Hz)"].values,
    1000 * sip["complex"].values.real,
    1000 * sip["complex"].values.real * PER,
    **kwargs,
)
ax[1].errorbar(
    sip["Freq (Hz)"].values,
    1000 * sip["complex"].values.imag,
    1000 * sip["complex"].values.imag * PEI,
    label="K389055 ($\sigma_{x}$)",
    **kwargs,
)
ax[1].legend(loc=2)
