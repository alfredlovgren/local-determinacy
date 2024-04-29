import numpy as np
from sequence_jacobian import *
import math

def onatskiMatrix(x, C):
    dM = np.empty((0,len(C[0])), int)
    for i in range(len(C)):
        tmp = np.empty(0, int)
        for j in range(len(C[i])):
            tmp = np.append(tmp, C[i][j][0,x])
        dM = np.append(dM, [tmp], axis=0)
    return dM

def onatski(target, exogenous, scale, T, ss0, G):
    DPi=np.array(G[target][exogenous])
    dpi = DPi /ss0[scale]

    C= np.array([[dpi]])
    
    Dr= np.array(G[target][exogenous])

    onatskiLag  = ss0["phi"]*Dr[0,0]/ss0["A"]

    lambdas = np.linspace(0, 2*np.pi, 10000)
    valuesF = np.empty(10000, complex)
    for i in range(10000):
        valuesF[i] = sum((onatskiMatrix(x, C))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0, T-1)) + onatskiLag*math.e**(np.sqrt(-1+0j)*lambdas[i])
    return valuesF

def windingNumberClockwise(F):
    return sum((-1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real > F[i-1].real)  else 0) for i in range(len(F))) 

def windingNumberCounterClockwise(F):
    return sum((1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real < F[i-1].real)  else 0) for i in range(len(F)))

def onatskiWindingNumber(F):
    return windingNumberClockwise(F) + windingNumberCounterClockwise(F)
