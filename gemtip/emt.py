# Authors : Charles L. Bérubé & Jean-Luc Gagnon
# Created on: Fri Jun 02 2023
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon

from network import MLP
from torch import nn
import torch

class GEMTIP_EMT:

    """Class to instance the device and neural network and use the function to obtain an anisotropic GEMTIP estimate.
    wt_dir : Directory where the weights are stored
    device : Device used (cuda or cpu)
    """

    def __init__(self, device = "cpu", dtype = torch.float32, wt_dir = "./weights") -> None:

        self.device = device
        self.dtype = dtype

        model_params = {
            "input_dim": 4,  # Number of integration variables
            "hidden_dim": 128,  # Dimension of the hidden layers
            "output_dim": 6,  # Output dimensions
            "n_hidden": 4,  # number of hidden layers
            "activation": nn.SiLU(),  # activation function
        }

        # Neural network instance
        weights = torch.load(f"{wt_dir}/weights-best.pt", weights_only=True, map_location=torch.device('cpu'))
        model = MLP(**model_params)
        model.load_state_dict(weights)
        model.eval()
        model.to(device = self.device, dtype=self.dtype)


        self.model = model



    @torch.no_grad
    def return_conduc_rot(self,gammal : torch.Tensor,lambdal : torch.Tensor, fl : torch.Tensor, sl : torch.Tensor, cl : torch.Tensor, 
                        alphal : torch.Tensor, s0 : torch.Tensor, w : torch.Tensor) -> torch.Tensor:
        """Returns the general effective medium conductivity. The tensors can be non-diagonal. 
        This is for a medium with N ellipsoidal inclusions with W frequencies considered.

        Input :  
            lambdal : Surface depolarization tensors (N x 3 x 3)
            gammal  : Volume depolarization tensors (N x 3 x 3)
            fl      : Volume fractions (N x 1)
            sl      : Inclusion conductivities (N x 1)
            cl      : Inclusion relaxation factors (N x 1)
            alphal  : Surface polarisability factors
            s0      : Bulk conductivity (3 x 3)
            w       : Frequencies (1 x W)

        Output : 
            sigma_tot : Effective medium conductivity (3 x W)
        """
        N = len(fl)
        sigma_tot = torch.empty(len(w), 3, 3, dtype=torch.complex64, device = self.device)

        ii = torch.eye(3, device = self.device, dtype = self.dtype)

        ii = ii.unsqueeze(0)
        ii = ii.repeat(N,1,1)
        s0 = s0.type(torch.complex64)
        sl = sl.type(torch.complex64)
        lambdal = lambdal.type(torch.complex64)
        gammal = gammal.type(torch.complex64)

    

        s0 = s0.unsqueeze(0)
        s0 = s0.repeat(N,1,1)

        sl = sl.unsqueeze(-1).unsqueeze(-1)
        fl = fl.unsqueeze(-1).unsqueeze(-1)
        alphal = alphal.unsqueeze(-1).unsqueeze(-1)
        cl = cl.unsqueeze(-1).unsqueeze(-1)

        deltasig = sl * ii - s0
        chi = (s0 * sl) @ torch.linalg.inv(deltasig)

        for i,_ in enumerate(w):

            kl = alphal * (1j * w[i]) ** (-cl)


            khii = kl * chi
            pl = khii @ lambdal @ torch.linalg.inv(gammal)
            sum_n = (
                (
                    torch.linalg.inv(ii + pl)
                    @ torch.linalg.inv(ii - deltasig * ((ii + pl) @ -gammal))
                    @ (ii + pl)
                )
                * fl
                * deltasig
            )
            sigma_tot[i] = s0[0] * ii[0] + sum_n.sum(0)

        return sigma_tot
    
    @torch.no_grad
    def eval_depol_tensor(self, a_l : torch.Tensor, b_l : torch.Tensor, c_l : torch.Tensor, s0 : torch.Tensor) -> tuple[torch.Tensor]:
        """
        Function to evaluate the depolarization tensors. 
        a_l >= b_l , a_l >= c_l and s0_x >= s0_y, s0_x >= s0_z. 

        Input : 
            a_l : length of the ellipsoid semi-axis in the x direction 
            b_l : length of the ellipsoid semi-axis in the y direction 
            c_l : length of the ellipsoid semi-axis in the z direction 
            s0  : Bulk conductivity tensor

        Output
            dpolt : Surface and volume depolarization tensors
        """


        A = b_l * a_l ** (-1) 
        B = c_l * a_l ** (-1)

        #Fit the dimensions
        N = len(A)

        C = (s0[1,1]/s0[0,0]).repeat(N)
        D = (s0[2,2]/s0[0,0]).repeat(N)

        ABCD = torch.cat((A,B,C,D))

        assert (ABCD.dtype == torch.float32 or ABCD.dtype == torch.float64)
        assert (torch.all(ABCD) <= 1), "Not all ratios (A,B,C,D) are strictly smaller than one"

        depolt_p = self.model.forward(torch.stack((A,B,C,D)).T)

        #Create the normalising matrix 
        norm_Lamb1 = (a_l.unsqueeze(1).repeat(1,3)) ** (-1)
        norm_Gamm = torch.ones_like(norm_Lamb1) 

        norm_Lamb_p = torch.ones_like(norm_Lamb1)

        norm_Lamb = torch.multiply(norm_Lamb1,norm_Lamb_p)
        
        norm_fact = torch.cat((norm_Gamm,norm_Lamb),1)

        #Get the normalized pre-tensors
        depolt = torch.multiply(depolt_p,norm_fact)

        depolt = depolt.split(depolt.size(1)//2, 1)
        depolt = [torch.diag_embed(split_tensor) for split_tensor in depolt]

        #Get the final tensors
        depol_gamma = depolt[0] * s0.inverse()
        depol_lambda = depolt[1] * s0.inverse()

        return depol_gamma,depol_lambda





        




