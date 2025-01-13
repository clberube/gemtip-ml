# Authors : Charles L. Bérubé & J.-L. Gagnon
# Created on: Fri Jun 02 2023
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon

from emt import*
import matplotlib.pyplot as plt
from plotlib import*

#Device used 

device = "cuda" if torch.cuda.is_available() else "cpu"

####
## Depolarisation tensors test (Figure 3) ##
####

#Here we reproduce figure 3 from our paper 

#Create the GEMTIP instance 

instance = GEMTIP_EMT(device=device, wt_dir = "gemtip/weights")

#Estimates used to generate the results (volumic fractions, semi-axis lenghts, etc.)
al = torch.tensor([0.1e-3, 0.2e-3],device=device)
bl=al
cl=al

s0 = torch.tensor([[1/100,0,0],[0,1/100,0],[0,0,1/100]],device=device)

fl = torch.tensor([0.20, 0.15],device=device)
sl = 1.0 / torch.tensor([0.1, 0.001],device=device)
c_exp_l = torch.tensor([0.8, 0.6],device=device)
alphal = torch.tensor([1, 0.01],device=device)


#Considered frequencies 
f = torch.logspace(-3, 4, 30,device=device)
w = 2 * torch.pi * f

#Get the depolarisation tensors 
depols = instance.eval_depol_tensor(al,bl,cl,s0)

#Get the effective medium conductivity 
EM_conduc = instance.return_conduc_rot(depols[0],depols[1],fl,sl,c_exp_l,alphal,s0,w).to("cpu")
f_cpu = f.to("cpu")

#Plot conductivity
figure, ax = plot_conductivities(f_cpu,EM_conduc)

figure.savefig(r"assets/figures/figure_3.pdf")

####
## Anisotropic conductivity curve ##
####

#Here we create anisotropic data

s0 = torch.tensor([[1/100,0,0],[0,1/150,0],[0,0,1/125]],device=device)
bl=al/2.5
cl=al/2

depols_anis = instance.eval_depol_tensor(al,bl,cl,s0)

EM_conduc_anis = instance.return_conduc_rot(depols_anis[0],depols_anis[1],fl,sl,c_exp_l,alphal,s0,w).to("cpu")

figure2, ax = plot_conductivities(f_cpu,EM_conduc_anis)

figure2.savefig(r"assets/figures/figure_anisotropic.pdf")