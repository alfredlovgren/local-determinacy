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

def onatski(targets: list, endogenous: list, predetermined: list, lags: list,
            T: int, ss0: sequence_jacobian.classes.steady_state_dict.SteadyStateDict, 
            H_U: sequence_jacobian.classes.jacobian_dict.JacobianDict, 
            exogenous: str = None, H_Z: sequence_jacobian.classes.jacobian_dict.JacobianDict = None) -> np.ndarray:

    dReal = np.zeros((1, 1))
    ss0phiB = 0

    if ((lags == None) | (predetermined == None)):
        raise Exception("Both number of lags and predeterminedness need to be specified!")

    if(((len(lags) != len(endogenous)) | (len(predetermined) != len(endogenous)) | (len(targets) != len(endogenous)))):
        raise Exception("Number of targets, unknowns, predetermined and lags must be the same!")

    if ((exogenous == None) | (H_Z == None)):
            raise Exception("Requires an exogenous parameter and a jacobian!")

    else:
         DReal = np.array(H_Z['asset_mkt'][exogenous])
         dReal = DReal*(1+ss0['rstar'])

    dU = np.empty((len(targets), len(endogenous)), dtype=np.ndarray)
    M = max(np.add(lags, predetermined))
    valueVBond = np.zeros((1, len(endogenous)))
    valueVCap = np.zeros((1, len(endogenous)))

    for i, target in enumerate(targets):
        for j, unknown in enumerate(endogenous):
            if unknown in H_U[target]:
                scale = 1 # Scale = 1?
                m = M - lags[j] - predetermined[j] 

                if(ss0[unknown] != 0):
                    scale = ss0[unknown]

                if((lags[j] > 0) | (predetermined[j] > 0)):
                    try:
                         ValueM =  np.array(H_U['bonds'][unknown].matrix(T))
                         valueBond =  ValueM[0][0]
                    except:   
                         valueBond = 0
                else:
                         valueBond = 0

                valueVBond[0,j] = valueBond

                if((lags[j] > 0) | (predetermined[j] > 0)):
                    try:
                         ValueC =  np.array(H_U['capital']['unknown'].matrix(T))
                         valueCap[0,j] =  ValueC[0][0]
                    except:   
                         valueCap = 0
                else:
                         valueCap = 0

                valueVCap[0,j] = valueCap

                if type(H_U[target][unknown]) is sequence_jacobian.classes.sparse_jacobians.SimpleSparse:
                        dU[i, j] = np.array(np.squeeze(np.asarray(H_U[target][unknown].matrix(T)*scale)))

                        if target =='asset_mkt':
                            dU[i, j][0,0] += valueVBond[0,j]*(1-ss0phiB)*dReal[0,0]
                else:
                    if m >0:
                        tmp1 = np.hstack(np.hsplit(np.zeros((T,T)), T)[0:m])
                        tmp2 = np.hstack(np.hsplit(H_U[target][unknown]*scale, T))[0:T-m+1]
                        dU[i, j] = np.hstack((tmp1,tmp2))
                    if m==0:
                        dU[i, j] = H_U[target][unknown]*scale
                    if target =='asset_mkt':
                            dU[i, j][0,0] += valueVBond[0,j]*(1-ss0phiB)*dReal[0,0]
            else:
                dU[i, j] = np.zeros((T,T))

    lambdas = np.linspace(0, 2*np.pi, 1000)
    valuesF = np.empty(1000, complex)

    for i in range(1000):
        if(len(targets) == 1):
            valuesF[i] =sum((dU[0,0][0,x])*math.e**(-np.sqrt(-1+0j)*(x-M)*lambdas[i]) for x in range(0,T-1)) #+ valueVBond[0,j]*(1-ss0phiB)*dReal[0,0]*math.e**(np.sqrt(-1+0j)*lambdas[i])
        else:
            valuesF[i] = np.linalg.det(sum((onatskiMatrix(x, dU))*math.e**(-np.sqrt(-1+0j)*(x-M)*lambdas[i]) for x in range(0,T-1)))

    return valuesF, dU

def windingNumberClockwise(F):
    return sum((-1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real > F[i-1].real)  else 0) for i in range(len(F))) 

def windingNumberCounterClockwise(F):
    return sum((1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real < F[i-1].real)  else 0) for i in range(len(F)))

def onatskiWindingNumber(F):
    return windingNumberClockwise(F) + windingNumberCounterClockwise(F)

def checkSolutions(F):
    #### Interpretation of winding number:
    #### Winding number CW (Multiple Solution)
    #### Winding number CCW (No Solution)

    winding_out = "Winding number: " + str(F)
    if F == 0:
        return(winding_out + "\nThe economy is DETERMINATE")
    elif F > 0:
        return(winding_out + "\nThe economy is INDETERMINATE (NO SOLUTION)")    
    elif F < 0:
        return(winding_out + "\nThe economy is INDETERMINATE (MULTIPLE SOLUTIONS)") 

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