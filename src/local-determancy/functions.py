import numpy as np
import sequence_jacobian 
import math
import matplotlib.pyplot as plt

def onatskiMatrix(x, C):
    dM = np.empty((len(C[0]),len(C[0])), float)
    for i in range(len(C)):
        for j in range(len(C[i])):
            dM[i, j] = C[i, j][0, x]
    return dM

def onatski(targets, endogenous, scale, T, ss0, H_U):
    if (len(targets) != len(endogenous)):
        raise Exception("Number of targets and unknowns must be the same!")

    dU = np.empty((len(targets), len(endogenous)), dtype=np.ndarray)

    for i, target in enumerate(targets):
        for j, unknown in enumerate(endogenous):
            if unknown in H_U[target]:
                if type(H_U[target][unknown]) is sequence_jacobian.classes.sparse_jacobians.SimpleSparse:
                    dU[i, j] = np.array(np.squeeze(np.asarray(H_U[target][unknown].matrix(T)/ss0[scale])))
                else:
                    dU[i, j] = np.array(H_U[target][unknown]/ss0[scale])
            else:
                dU[i, j] = np.zeros((T,T))

    lambdas = np.linspace(0, 2*np.pi, 1000)
    valuesF = np.empty(1000, complex)
    for i in range(1000):
        if(len(targets) == 1):
            valuesF[i] =sum((dU[0,0][0,x])*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0,T-1))
        else:
            # valuesF_real[i] =complex(np.linalg.det(sum((onatskiMatrix(x, dU))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0,T-1))))
            # valuesF_imag[i] =complex(np.linalg.det(sum((onatskiMatrix(x, dU))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0,T-1))))
            #valuesF[i] = np.linalg.det(sum((onatskiMatrix(x, dU))*(np.cos((x-1)*lambdas[i])+ (-np.sqrt(-1+0j))* np.sin((x-1)*lambdas[i])) for x in range(0,T-1)))
            valuesF[i] = np.linalg.det(sum((onatskiMatrix(x, dU))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0,T-1)))
    
    return valuesF


def windingNumberClockwise(F):
    return sum((-1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real > F[i-1].real)  else 0) for i in range(len(F))) 

def windingNumberCounterClockwise(F):
    return sum((1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real < F[i-1].real)  else 0) for i in range(len(F)))

def onatskiWindingNumber(F):
    return windingNumberClockwise(F) + windingNumberCounterClockwi