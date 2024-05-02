import numpy as np
import sequence_jacobian 
import math
import matplotlib.pyplot as plt

def onatskiMatrix(x, C):
    dM = np.empty((0,len(C[0])), int)
    for i in range(len(C)):
        tmp = np.empty(0, int)
        for j in range(len(C[i])):
            tmp = np.append(tmp, C[i][j][0,x])
        dM = np.append(dM, [tmp], axis=0)
    return dM

def onatski(targets, endogenous, scale, T, ss0, H_U):
    if (len(targets) != len(endogenous)):
        raise Exception("Number of targets and unknowns must be the same!")

    dpi = np.empty((len(targets), len(endogenous)), dtype=np.ndarray)

    for i, target in enumerate(targets):
        for j, unknown in enumerate(endogenous):
            if unknown in H_U[target]:
                if type(H_U[target][unknown]) is sequence_jacobian.classes.sparse_jacobians.SimpleSparse:
                    dpi[i, j] = np.array(np.squeeze(np.asarray(H_U[target][unknown].matrix(T)/ss0[scale])))
                else:
                    dpi[i, j] = np.array(H_U[target][unknown]/ss0[scale])
            else:
                dpi[i, j] = np.zeros((T,T))

    lambdas = np.linspace(0, 2*np.pi, 1000)
    valuesF = np.empty(1000, complex)
    for i in range(1000):
        valuesF[i] =complex(np.linalg.det(sum((onatskiMatrix(x, dpi))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0,T-1))))
    return valuesF

def windingNumberClockwise(F):
    return sum((-1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real > F[i-1].real)  else 0) for i in range(len(F))) 

def windingNumberCounterClockwise(F):
    return sum((1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real < F[i-1].real)  else 0) for i in range(len(F)))

def onatskiWindingNumber(F):
    return windingNumberClockwise(F) + windingNumberCounterClockwise(F)

def plot(F):
    plt.plot(F.real, F.imag, color='blue',linewidth=3)

    plt.xlabel('Real - axis', fontsize=18)

    plt.ylabel('Imaginary - axis', fontsize=18)

    plt.title('Onatski graph')

    angle = np.deg2rad(45)

    cross_length = max(max(F.real)-min(F.real), max(F.imag)-min(F.imag)) * 0.065

    plt.plot([-cross_length * np.cos(angle), cross_length * np.cos(angle)],
            [-cross_length * np.sin(angle), cross_length * np.sin(angle)],
            color='red', linestyle='-', linewidth=3)  # Diagonal line /

    plt.plot([-cross_length * np.cos(angle), cross_length * np.cos(angle)],
            [cross_length * np.sin(angle), -cross_length * np.sin(angle)],
            color='red', linestyle='-', linewidth=3)  # Diagonal line \

    plt.show()