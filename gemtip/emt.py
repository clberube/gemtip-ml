#
# Author: Charles L. Bérubé
# Created on: Fri Jun 02 2023
#
# Copyright (c) 2023 C.L. Bérubé & J.-L. Gagnon
#

import numpy as np
import torch
from scipy.stats import special_ortho_group


def rotationAxe(M, alpha=None, beta=None, gamma=None, random=False):
    """Retourne une matrice tournée par rapport aux axes x,y,z
    M     : Tenseur de dépolarisation à tourner
    alpha : Angle par rapport à x
    beta  : Angle par rapport à y
    gamma : Angle par rapport à z
    """

    if random:
        Mrot = special_ortho_group.rvs(3)

    else:
        Mrot = np.array(
            [
                [
                    np.cos(alpha) * np.cos(beta),
                    np.cos(alpha) * np.sin(beta) * np.sin(gamma)
                    - np.sin(alpha) * np.cos(gamma),
                    np.cos(alpha) * np.sin(beta) * np.cos(gamma)
                    + np.sin(alpha) * np.sin(gamma),
                ],
                [
                    np.sin(alpha) * np.cos(beta),
                    np.sin(alpha) * np.sin(beta) * np.sin(gamma)
                    + np.cos(alpha) * np.cos(gamma),
                    np.sin(alpha) * np.sin(beta) * np.cos(gamma)
                    - np.cos(alpha) * np.sin(gamma),
                ],
                [
                    -np.sin(beta),
                    np.cos(beta) * np.sin(gamma),
                    np.cos(beta) * np.cos(gamma),
                ],
            ]
        )
    Mrot = torch.from_numpy(Mrot).float()
    Mrotm1 = torch.transpose(Mrot, 0, 1)
    Mrotated = Mrot @ M @ Mrotm1

    return Mrotated


# Fonction générale, à utiliser seulement quand il y a une rotation et donc des termes non-diagonaux, sinon c'est plus lourd en terme de calcul :
def returnConducRot(Lambdal, Gammal, min_abc, min_xyz, fl, sl, Cl, alphal, s0, w):
    """Retourne la conductivité effective de grandeur (3 x 3) x w.size(), dans le cas générique, dans la base orthogonale (x,y,z)
    Lambdal : Tenseur de dépolarisation surfacique
    Gammal  : Tenseur de dépolarisation volumique
    N       : Nombre d'inclusions différentes
    fl      : Fraction volumique de la lième inclusion (1 x N)
    sl      : Conductivité de la lième inclusion (1 x N)
    Cl      : Coefficient de relaxation de la lième inclusion (1 x N)
    s0      : Conductivité du matériau hôte = millieu infini (1 x N)
    w       : Fréquences (1 x len(w))
    """

    sigmaTot = torch.empty(len(w), 3, 3, dtype=torch.complex64)

    I = torch.eye(3)

    s0 = s0.type(torch.complex64)
    sl = sl.type(torch.complex64)
    Lambdal = Lambdal.type(torch.complex64)
    Gammal = Gammal.type(torch.complex64)

    s0 = s0.unsqueeze(0)

    fl = fl.unsqueeze(-1).unsqueeze(-1)
    Cl = Cl.unsqueeze(-1).unsqueeze(-1)
    alphal = alphal.unsqueeze(-1).unsqueeze(-1)

    Lambdal = Lambdal @ torch.linalg.inv(s0) / min_xyz / min_abc
    Gammal = Gammal @ torch.linalg.inv(s0) / min_xyz

    for i in range(len(w)):
        kl = alphal * (1j * w[i]) ** (-Cl)
        deltasig = sl * I - s0
        khii = kl * (s0 * sl) @ torch.linalg.inv(deltasig)
        pl = khii @ Lambdal @ torch.linalg.inv(Gammal)
        sumN = (
            (
                torch.linalg.inv(I + pl)
                @ torch.linalg.inv(I - deltasig * ((I + pl) @ -Gammal))
                @ (I + pl)
            )
            * fl
            * deltasig
        )
        sigmaTot[i] = s0 * I + sumN.sum(0)

    return sigmaTot


def returnConducEff(Lambdal, Gammal, al, fl, sl, Cl, alphal, s0, w):
    """Retourne la conductivité effective de grandeur 3 x w.size(), donc en x,y,z dans le cas spécifique où les deux tenseurs sont diagonaux
    Lambdal : Tenseur de dépolarisation surfacique
    Gammal  : Tenseur de dépolarisation volumique
    N       : Nombre d'inclusions différentes
    fl      : Fraction volumique de la lième inclusion (1 x N)
    sl      : Conductivité de la lième inclusion (1 x N)
    Cl      : Coefficient de relaxation de la lième inclusion (1 x N)
    s0      : Conductivité du matériau hôte = millieu infini (1 x N)
    w       : Fréquences (1 x len(w))
    """
    sumN11 = 0
    sumN22 = 0
    sumN33 = 0
    Lambdal = Lambdal / (s0 * al).unsqueeze(-1).unsqueeze(-1)
    Gammal = Gammal / s0
    for i in range(len(Lambdal)):
        kl = alphal[i] * (1j * w) ** (-Cl[i])
        deltasig = sl[i] - s0
        khii = kl * (s0 * sl[i]) / (deltasig)
        sumi11 = fl[i] / (
            np.ones(len(w)) / deltasig
            + (Lambdal[i][0, 0] * khii + Gammal[i][0, 0] * np.ones(len(w)))
        )
        sumi22 = fl[i] / (
            np.ones(len(w)) / deltasig
            + (Lambdal[i][1, 1] * khii + Gammal[i][1, 1] * np.ones(len(w)))
        )
        sumi33 = fl[i] / (
            np.ones(len(w)) / deltasig
            + (Lambdal[i][2, 2] * khii + Gammal[i][2, 2] * np.ones(len(w)))
        )
        sumN11 = sumi11 + sumN11
        sumN22 = sumi22 + sumN22
        sumN33 = sumi33 + sumN33
    sigmaEff = (
        s0 * np.ones(len(w)) + sumN11,
        s0 * np.ones(len(w)) + sumN22,
        s0 * np.ones(len(w)) + sumN33,
    )
    return sigmaEff


def Nrandomparam(minmax, Nobjec):
    """Fonction retournant selon une liste d'extremum N paramètres
    minmax : Liste de la forme [min,max]
    Nobjec : Nombre de paramètre à générer
    """
    s = np.random.uniform(minmax[0], minmax[1], Nobjec)
    return s


def Nrandomabc(minmax, Nobjec):
    """Fonction retournant selon une liste d'extremum 3N paramètres qui respecte la condition s[0] > s[1] > s[2]
    minmax : Liste de la forme [min,max]
    Nobjec : Nombre de paramètre à générer
    """
    a = np.random.uniform(minmax[0], minmax[1], Nobjec)
    b = np.random.uniform(minmax[0], minmax[1], Nobjec)
    c = np.random.uniform(minmax[0], minmax[1], Nobjec)
    s = np.transpose(np.sort([a, b, c], axis=0))
    s = np.flip(s, axis=1)
    return s


def returnGEMTIPalphaBorn(s0, sl, taul, Cl, amoy):
    """Fonction retournant selon des listes d'extremum les bornes sur le paramètre semi-empirique alpha
    sl   : Conductivité de la lième inclusion (1 x N)
    s0   : Conductivité du matériau hôte = millieu infini (1 x N)
    fl   : Fraction volumique de la lième inclusion (1 x N)
    sl   : Conductivité de la lième inclusion (1 x N)
    Cl   : Coefficient de relaxation de la lième inclusion (1 x N)
    amoy : Rayon moyen (a + b + c)/3
    """
    pl = np.reciprocal(sl)
    p0 = np.reciprocal(s0)

    alphamin = (amoy[0] * (2 * pl[0] + p0[0])) / (taul[1]) ** (Cl[0])
    alphamax = (amoy[1] * (2 * pl[1] + p0[1])) / (taul[0]) ** (Cl[1])
    return np.array([alphamin, alphamax])


def dirichletSum(maxFrac, N):
    """Fonction retournant au maximum une liste contenant les fractions volumiques sommant à maxFrac
    maxFrac : Valeur maximale de la somme de la fraction totale d'inclusion
    N       : Nombre d'inclusion différentes
    """
    fl = maxFrac * np.random.rand() * np.random.dirichlet(np.ones(N))
    return fl
