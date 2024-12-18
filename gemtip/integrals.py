#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#

import numpy as np
from scipy.special import ellipkinc, ellipeinc
from torchquad import Simpson
import torch


###
##MISC FUNCTIONS
###


def rhoIso(tp, A, B):
    rhoAB = torch.sqrt(
        (torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (A * torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (B * torch.cos(tp[0])) ** 2
    )
    return rhoAB


def etaIso(tp, A, B):
    etaAB = torch.sqrt(
        (torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (A ** (-1) * torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (B ** (-1) * torch.cos(tp[0])) ** 2
    )
    return etaAB


def rhonIso11(tp, Ap, Bp):
    rhoAB = torch.sqrt(
        (torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (Ap * torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (Bp * torch.cos(tp[0])) ** 2
    )
    return rhoAB


def rhonIso22(tp, Ap, Bp):
    rhoAB = torch.sqrt(
        ((1 / Ap) * torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + ((Bp / Ap) * torch.cos(tp[0])) ** 2
    )
    return rhoAB


def rhonIso33(tp, Ap, Bp):
    rhoAB = torch.sqrt(
        ((1 / Bp) * torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + ((Ap / Bp) * torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (torch.cos(tp[0])) ** 2
    )
    return rhoAB


def etanIso11(tp, Cp, Dp):
    rhoAB = torch.sqrt(
        (torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (torch.sqrt(1 / Cp) * torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (torch.sqrt(1 / Dp) * torch.cos(tp[0])) ** 2
    )
    return rhoAB


def etanIso22(tp, Cp, Dp):
    rhoAB = torch.sqrt(
        (torch.sqrt(Cp) * torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (torch.sqrt(Cp / Dp) * torch.cos(tp[0])) ** 2
    )
    return rhoAB


def etanIso33(tp, Cp, Dp):
    rhoAB = torch.sqrt(
        (torch.sqrt(Dp) * torch.sin(tp[0]) * torch.cos(tp[1])) ** 2
        + (torch.sqrt(Dp / Cp) * torch.sin(tp[0]) * torch.sin(tp[1])) ** 2
        + (torch.cos(tp[0])) ** 2
    )
    return rhoAB


###
##INTEGRANDS TORCHQUAD
###

# Volume isotropic


class VolumeIntegrandIso:
    def __init__(self, A, B):
        # self.dtype = dtype
        # self.device = device

        # torch.set_default_dtype(dtype)
        # torch.set_default_device(device)

        self.A = A
        self.B = B

    def Integrand11V(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhoIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**3)

        ele11 = factor * (torch.sin(tp[0])) ** (2) * (torch.cos(tp[1])) ** (2)

        return ele11

    def Integrand22V(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhoIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**3)

        ele22 = factor * (torch.sin(tp[0])) ** (2) * (torch.sin(tp[1])) ** (2)

        return ele22

    def Integrand33V(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhoIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**3)

        ele33 = factor * (torch.cos(tp[0])) ** (2)

        return ele33


# Surface isotropic


class SurfaceIntegrandIso:
    def __init__(self, A, B):
        # self.dtype = dtype
        # self.device = device

        # torch.set_default_dtype(dtype)
        # torch.set_default_device(device)

        self.A = A
        self.B = B

    def Integrand11S(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhoIso(tp, self.A, self.B)
        etaAB = etaIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaAB)

        ele11 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.cos(tp[1])) ** (2)
            * (-3 + rhoAB**2)
        )

        return ele11

    def Integrand22S(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhoIso(tp, self.A, self.B)
        etaAB = etaIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaAB)

        ele22 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.sin(tp[1])) ** (2)
            * (-3 + (rhoAB / self.A) ** 2)
        )

        return ele22

    def Integrand33S(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhoIso(tp, self.A, self.B)
        etaAB = etaIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaAB)

        ele33 = factor * (torch.cos(tp[0])) ** (2) * (-3 + (rhoAB / self.B) ** 2)

        return ele33


# Volume non-isotropic


class VolumeIntegrandNonIso:
    def __init__(self, Ap, Bp):

        self.Ap = Ap
        self.Bp = Bp

    def Integrand11V(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhonIso11(tp, self.Ap, self.Bp)

        factor = (torch.sin(tp[0])) / (rhoAB**3)

        ele11 = factor * (torch.sin(tp[0])) ** (2) * (torch.cos(tp[1])) ** (2)

        return ele11

    def Integrand22V(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhonIso11(tp, self.Ap, self.Bp)

        factor = (torch.sin(tp[0])) / (rhoAB**3)

        ele22 = factor * (torch.sin(tp[0])) ** (2) * (torch.sin(tp[1])) ** (2)

        return ele22

    def Integrand33V(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhonIso11(tp, self.Ap, self.Bp)

        factor = (torch.sin(tp[0])) / (rhoAB**3)

        ele33 = factor * (torch.cos(tp[0])) ** (2)

        return ele33


# Surface non-isotropic


class SurfaceIntegrandNonIso:
    def __init__(self, A, B, Ap, Bp, Cp, Dp):

        self.Ap = Ap
        self.Bp = Bp
        self.Cp = Cp
        self.Dp = Dp
        self.A = A
        self.B = B

    def Integrand11S(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhonIso11(tp, self.Ap, self.Bp)
        etaAB = etanIso11(tp, self.Cp, self.Dp)
        etaRAB = etaIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaRAB)

        ele11 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.cos(tp[1])) ** (2)
            * (-3 * etaAB + rhoAB**2)
        )

        return ele11

    def Integrand22S(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhonIso22(tp, self.Ap, self.Bp)
        etaAB = etanIso22(tp, self.Cp, self.Dp)
        rhoABn = rhonIso11(tp, self.Ap, self.Bp)
        etaRAB = etaIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoABn**5 * etaRAB)

        ele22 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.sin(tp[1])) ** (2)
            * (-3 * etaAB + rhoAB**2)
        )

        return ele22

    def Integrand33S(self, tp):
        tp = torch.transpose(tp, 0, 1)  # theta and phi parameters combined

        rhoAB = rhonIso33(tp, self.Ap, self.Bp)
        etaAB = etanIso33(tp, self.Cp, self.Dp)
        rhoABn = rhonIso11(tp, self.Ap, self.Bp)
        etaRAB = etaIso(tp, self.A, self.B)

        factor = (torch.sin(tp[0])) / (rhoABn**5 * etaRAB)

        ele33 = factor * (torch.cos(tp[0])) ** (2) * (-3 * etaAB + rhoAB**2)

        return ele33


###
# INTEGRAND MACHINE LEARNING
###


# Volume isotropic
def fullIntegrandIsoV(A, B, t, p, Tensii=None):
    tp = [t, p]
    rhoAB = rhoIso(tp, A, B)

    factor = (torch.sin(tp[0])) / (rhoAB**3)

    def IntegrandIso11V(factor, A, B, t, p):
        ele11 = factor * (torch.sin(tp[0])) ** (2) * (torch.cos(tp[1])) ** (2)

        return ele11

    def IntegrandIso22V(factor, A, B, tp):
        ele22 = factor * (torch.sin(tp[0])) ** (2) * (torch.sin(tp[1])) ** (2)

        return ele22

    def IntegrandIso33V(factor, A, B, tp):
        ele33 = factor * (torch.cos(tp[0])) ** (2)

        return ele33

    if Tensii == None:
        return (
            IntegrandIso11V(factor, A, B, tp),
            IntegrandIso22V(factor, A, B, tp),
            IntegrandIso33V(factor, A, B, tp),
        )
    elif Tensii == 11:
        return IntegrandIso11V(factor, A, B, tp)
    elif Tensii == 22:
        return IntegrandIso22V(factor, A, B, tp)
    elif Tensii == 33:
        return IntegrandIso33V(factor, A, B, tp)
    else:
        print(
            f"Error : please chose between 11,22,33 or no parameters for the last parameter... \n Defaulting to 11"
        )
        return IntegrandIso11V(factor, A, B, tp)


# Surface isotropic


def fullIntegrandIsoS(A, B, t, p, Tensii=None):
    tp = [t, p]
    rhoAB = rhoIso(tp, A, B)
    etaAB = etaIso(tp, A, B)

    factor = (torch.sin(tp[0])) / (rhoAB**5 * etaAB)

    def IntegrandIso11S(factor, A, B, tp):
        ele11 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.cos(tp[1])) ** (2)
            * (-3 + rhoAB**2)
        )

        return ele11

    def IntegrandIso22S(factor, A, B, tp):
        ele22 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.sin(tp[1])) ** (2)
            * (-3 + (rhoAB / A) ** 2)
        )

        return ele22

    def IntegrandIso33S(factor, A, B, tp):
        ele33 = factor * (torch.cos(tp[0])) ** (2) * (-3 + (rhoAB / B) ** 2)

        return ele33

    if Tensii == None:
        return (
            IntegrandIso11S(factor, A, B, tp),
            IntegrandIso22S(factor, A, B, tp),
            IntegrandIso33S(factor, A, B, tp),
        )
    elif Tensii == 11:
        return IntegrandIso11S(factor, A, B, tp)
    elif Tensii == 22:
        return IntegrandIso22S(factor, A, B, tp)
    elif Tensii == 33:
        return IntegrandIso33S(factor, A, B, tp)
    else:
        print(
            f"Error : please chose between 11,22,33 or no parameters for the last parameter... \n Defaulting to 11"
        )
        return IntegrandIso11S(factor, A, B, tp)


# Volume non-isotropic


def fullIntegrandNIsoV(Ap, Bp, t, p, Tensii=None):
    tp = [t, p]
    rhoAB = rhonIso11(tp, Ap, Bp)

    factor = (torch.sin(tp[0])) / (rhoAB**3)

    def IntegrandNIso11V(factor, Ap, Bp, tp):
        ele11 = factor * (torch.sin(tp[0])) ** (2) * (torch.cos(tp[1])) ** (2)

        return ele11

    def IntegrandNIso22V(factor, Ap, Bp, tp):
        ele22 = factor * (torch.sin(tp[0])) ** (2) * (torch.sin(tp[1])) ** (2)

        return ele22

    def IntegrandNIso33V(factor, Ap, Bp, tp):
        ele33 = factor * (torch.cos(tp[0])) ** (2)

        return ele33

    if Tensii == None:
        return (
            IntegrandNIso11V(factor, Ap, Bp, tp),
            IntegrandNIso22V(factor, Ap, Bp, tp),
            IntegrandNIso33V(factor, Ap, Bp, tp),
        )
    elif Tensii == 11:
        return IntegrandNIso11V(factor, Ap, Bp, tp)
    elif Tensii == 22:
        return IntegrandNIso22V(factor, Ap, Bp, tp)
    elif Tensii == 33:
        return IntegrandNIso33V(factor, Ap, Bp, tp)
    else:
        print(
            f"Error : please chose between 11,22,33 or no parameters for the last parameter... \n Defaulting to 11"
        )
        return IntegrandNIso11V(factor, Ap, Bp, tp)


# Surface non-isotropic
def fullIntegrandNIsoV(A, B, Ap, Bp, Cp, Dp, t, p, Tensii=None):
    tp = [t, p]

    def IntegrandNIso11S(tp, A, B, Ap, Bp, Cp, Dp):
        rhoAB = rhonIso11(tp, Ap, Bp)
        etaAB = etanIso11(tp, Cp, Dp)
        etaRAB = etaIso(tp, A, B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaRAB)

        ele11 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.cos(tp[1])) ** (2)
            * (-3 * etaAB + rhoAB**2)
        )

        return ele11

    def IntegrandNIso22S(tp, A, B, Ap, Bp, Cp, Dp):
        rhoAB = rhonIso22(tp, Ap, Bp)
        etaAB = etanIso22(tp, Cp, Dp)
        etaRAB = etaIso(tp, A, B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaRAB)

        ele22 = (
            factor
            * (torch.sin(tp[0])) ** (2)
            * (torch.sin(tp[1])) ** (2)
            * (-3 * etaAB + rhoAB**2)
        )

        return ele22

    def IntegrandNIso33S(tp, A, B, Ap, Bp, Cp, Dp):
        rhoAB = rhonIso33(tp, Ap, Bp)
        etaAB = etanIso33(tp, Cp, Dp)
        etaRAB = etaIso(tp, A, B)

        factor = (torch.sin(tp[0])) / (rhoAB**5 * etaRAB)

        ele33 = factor * (torch.cos(tp[0])) ** (2) * (-3 * etaAB + rhoAB**2)

        return ele33

    if Tensii == None:
        return (
            IntegrandNIso11S(tp, Ap, Bp, Cp, Dp),
            IntegrandNIso22S(tp, Ap, Bp, Cp, Dp),
            IntegrandNIso33S(tp, Ap, Bp, Cp, Dp),
        )
    elif Tensii == 11:
        return IntegrandNIso11S(tp, Ap, Bp, Cp, Dp)
    elif Tensii == 22:
        return IntegrandNIso22S(tp, Ap, Bp, Cp, Dp)
    elif Tensii == 33:
        return IntegrandNIso33S(tp, Ap, Bp, Cp, Dp)
    else:
        print(
            f"Error : please chose between 11,22,33 or no parameters for the last parameter... \n Defaulting to 11"
        )
        return


def NxNyNz_elliptic_functions(a, b, c):
    if a == b and b == c:
        return np.array([[1 / 3, 0, 0], [0, 1 / 3, 0], [0, 0, 1 / 3]])

    else:  # Dans ce cas, a > b > c
        k = (a**2 - b**2) / (a**2 - c**2)

        phi = np.arcsin(np.sqrt(1 - c**2 / a**2))
        if phi >= np.pi / 2:
            phi = np.pi - phi

        F = ellipkinc(phi, k)
        E = ellipeinc(phi, k)

        gamma_11 = (a * b * c * (F - E)) / ((a**2 - b**2) * np.sqrt(a**2 - c**2))
        gamma_22 = (
            (a * b * c * (E - F)) / ((a**2 - b**2) * np.sqrt(a**2 - c**2))
            + (a * b * c * E) / ((b**2 - c**2) * np.sqrt(a**2 - c**2))
            - (c**2 / (b**2 - c**2))
        )

        gamma_33 = (-a * b * c * E) / ((b**2 - c**2) * np.sqrt(a**2 - c**2)) + (
            b**2 / (b**2 - c**2)
        )

    return np.array(
        [[gamma_11.item(), 0, 0], [0, gamma_22.item(), 0], [0, 0, gamma_33.item()]]
    )


###
##ZHADANOV INTEGRAL FOR SURFACE
###


class IntegrandZH:
    def __init__(self, a, b):

        self.a = a
        self.b = b

    def ele33(self, z):
        numerator = (
            3 * self.b**4 - self.b**2 * self.a**2 - (self.b**2 - self.a**2) * z**2
        ) * z**2
        denominator = (
            self.a**2 * self.b**2 + (self.b**2 - self.a**2) * z**2
        ) ** (5 / 2) * torch.sqrt(self.b**4 - (self.b**2 - self.a**2) * z**2)
        return (numerator / denominator) * self.b * self.a**3

    def ele1122(self, z):
        numerator = (
            2 * self.a**2 * self.b**2 - (self.b**2 - self.a**2) * z**2
        ) * (self.b**2 - z**2)
        denominator = (
            self.a**2 * self.b**2 + (self.b**2 - self.a**2) * z**2
        ) ** (5 / 2) * torch.sqrt(self.b**4 - (self.b**2 - self.a**2) * z**2)
        return (numerator / denominator) * ((self.a * self.b**3) / 2)


###
##INTEGRATION SCHEMES
###


def torchIntegralPVol(a, b, c, sigma, matrix, N, dtype, device, Integrator):
    # Definition of the factors
    A = b / a
    B = c / a
    if Integrator is None:
        Integrator = Simpson()

    Domain = [[0, torch.pi], [0, 2 * torch.pi]]  # Domain of integration
    if matrix:
        S = sigma  # The matrix is diagonal
        sig11 = S[0][0]
        sig22 = S[1][1]
        sig33 = S[2][2]

        # Calculation of the new factors
        ap = a / torch.sqrt(sig11)
        bp = b / torch.sqrt(sig22)
        cp = c / torch.sqrt(sig33)
        Ap = bp / ap
        Bp = cp / ap
        sigs = torch.sqrt(sig11 * sig22 * sig33)

        # Integration scheme (volume)

        IntegrandV = VolumeIntegrandNonIso(Ap, Bp, dtype, device)
        factorV = -(a * b * c) / (4 * torch.pi * sigs * ap**3 * sig33)
        Gamma33 = factorV * Integrator.integrate(
            IntegrandV.Integrand33V, 2, N, Domain
        )  # Integration using torchquad

        return Gamma33

    else:
        sig = sigma

        # Integration scheme (volume)

        IntegrandV = VolumeIntegrandIso(A, B)
        factorV = -(b * c) / (4 * torch.pi * sig * a**2)
        Gamma33 = factorV * Integrator.integrate(
            IntegrandV.Integrand33V, 2, N, Domain
        )  # Integration using torchquad

        return Gamma33


def torchIntegralPSurf(a, b, c, sigma, matrix, N, dtype, device, Integrator):
    # Definition of the factors
    A = b / a
    B = c / a
    if Integrator is None:
        Integrator = Simpson()

    Domain = [[0, torch.pi], [0, 2 * torch.pi]]  # Domain of integration
    if matrix:
        S = sigma  # The matrix is diagonal
        sig11 = S[0][0]
        sig22 = S[1][1]
        sig33 = S[2][2]

        # Calculation of the new factors
        ap = a / torch.sqrt(sig11)
        bp = b / torch.sqrt(sig22)
        cp = c / torch.sqrt(sig33)
        A = b / a
        B = c / a
        Ap = bp / ap
        Bp = cp / ap
        Cp = sig22 / sig11
        Dp = sig33 / sig11

        sigs = torch.sqrt(sig11 * sig22 * sig33)

        # Integration scheme (surface)

        IntegrandS = SurfaceIntegrandNonIso(A, B, Ap, Bp, Cp, Dp, dtype, device)
        factorS = (a**2 * b * c) / (4 * torch.pi * sigs * ap**5 * sig33**2)
        Lambda33 = factorS * Integrator.integrate(
            IntegrandS.Integrand33S, 2, N, Domain
        )  # Integration using torchquad

        return Lambda33

    else:
        sig = sigma
        # Integration scheme (surface)
        IntegrandS = SurfaceIntegrandIso(A, B, dtype, device)
        factorS = (b * c) / (4 * torch.pi * sig * a**3)
        Lambda33 = factorS * Integrator.integrate(
            IntegrandS.Integrand33S, 2, N, Domain
        )  # Integration using torchquad
        return Lambda33


# Permutation in order to get the other parameters
def evalfullTensVol(a, b, c, sigma, matrix, N, dtype, device, Integrator=None):
    diagTensS = []

    for i in range(3):
        if matrix == True:
            sigma11 = sigma[0, 0]
            sigma22 = sigma[1, 1]
            sigma33 = sigma[2, 2]
            if i == 0:
                sigs = [sigma11, sigma22, sigma33]
                abcs = [a, b, c]
            elif i == 1:
                sigs = [sigma33, sigma22, sigma11]
                abcs = [c, b, a]
            elif i == 2:
                sigs = [sigma11, sigma33, sigma22]
                abcs = [a, c, b]
            sigma = torch.tensor([[sigs[0], 0, 0], [0, sigs[1], 0], [0, 0, sigs[2]]])
            total = torchIntegralPVol(
                abcs[0], abcs[1], abcs[2], sigma, matrix, N, dtype, device, Integrator
            )

        else:
            if i == 0:
                abcs = [a, b, c]
            elif i == 1:
                abcs = [c, b, a]
            elif i == 2:
                abcs = [a, c, b]

            total = torchIntegralPVol(
                abcs[0], abcs[1], abcs[2], sigma, matrix, N, dtype, device, Integrator
            )

        elementSii = total.cpu().numpy()
        diagTensS.append(elementSii)

    diagTensS = np.array([diagTensS[1], diagTensS[2], diagTensS[0]])

    return diagTensS


def evalfullTensSurf(a, b, c, sigma, matrix, N, dtype, device, Integrator=None):
    diagTensS = []

    for i in range(3):
        if matrix == True:
            sigma11 = sigma[0, 0]
            sigma22 = sigma[1, 1]
            sigma33 = sigma[2, 2]

            if i == 0:
                sigs = [sigma11, sigma22, sigma33]
                abcs = [a, b, c]
            elif i == 1:
                sigs = [sigma33, sigma22, sigma11]
                abcs = [c, b, a]
            elif i == 2:
                sigs = [sigma11, sigma33, sigma22]
                abcs = [a, c, b]

            sigma = torch.tensor([[sigs[0], 0, 0], [0, sigs[1], 0], [0, 0, sigs[2]]])
            total = torchIntegralPSurf(
                abcs[0], abcs[1], abcs[2], sigma, matrix, N, dtype, device, Integrator
            )

        else:
            if i == 0:
                abcs = [a, b, c]
            if i == 1:
                abcs = [c, b, a]
            elif i == 2:
                abcs = [a, c, b]

            total = torchIntegralPSurf(
                abcs[0], abcs[1], abcs[2], sigma, matrix, N, dtype, device, Integrator
            )

        elementSii = total.cpu()
        diagTensS.append(elementSii)

    diagTensS = [diagTensS[1], diagTensS[2], diagTensS[0]]

    return diagTensS


def integrandzhSurf(a, b, sig, N):
    dom = [[0, b]]
    Integrator = Simpson()
    fct112233 = IntegrandZH(a, b)
    Lambda1122 = (1 / sig) * Integrator.integrate(
        fct112233.ele1122, 1, N, dom
    ).cpu().numpy()
    Lambda33 = (1 / sig) * Integrator.integrate(
        fct112233.ele33, 1, N, dom
    ).cpu().numpy()
    diagtens = np.array([Lambda1122, Lambda1122, Lambda33])
    return diagtens
