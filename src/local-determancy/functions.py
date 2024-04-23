import numpy as np
import sequence_jacobian as sj
import math

# def onatski(self):
#         T = 300
#         exogenous = ['rstar','rshock','taushock','Realshock']
#         unknowns = ['piL']
#         targets = ['asset_mkt','goods_mkt','test_mkt']

#         H_U = self.hank.jacobian(self.ss, unknowns, targets, T=T)
#         H_Z = self.hank.jacobian(self.ss, exogenous, targets, T=T)

#         DReal=np.array(H_Z['asset_mkt']['Realshock'])
#         Dtau=np.array(H_Z['asset_mkt']['taushock'])
#         Dr= np.array(H_Z['asset_mkt']['rshock'])
#         DPi=np.array(H_U['asset_mkt']['piL'])
#         Dmpc=np.array(H_Z['goods_mkt']['taushock'])

#         #dr = Dr * (1+0.005)/ss0["A
#         dpi = DPi /self.ss0["A"]
        
#         dtau = Dtau 

#         dReal = DReal*(1+self.ss0["rstar"]) /self.ss0["A"]

#         dmpc = Dmpc
        
        #Onatskicoeff = np.empty(T)

        #for x in range(0,T-1):
            #Onatskicoeff[x] =  -dtau[0,x]*(1+self.ss0["rstar"])-dReal[0,x]+ ((1+self.ss0["rstar"])*dtau[0,x+1]+dReal[0,x+1])*self.ss0["phi"]      

        #onatskidummy =  (dtau[0,0]*(1+self.ss0["rstar"])+dReal[0,0])*self.ss0["phi"]

def onatskiMatrix(x, C):
    dM = np.empty((0,len(C[0])), int)
    for i in range(len(C)):
        tmp = np.empty(0, int)
        for j in range(len(C[i])):
            tmp = np.append(tmp, C[i][j][0,x])
        dM = np.append(dM, [tmp], axis=0)
    return dM

def onatski(C, T, ss0, H_Z):
    Dr= np.array(H_Z['asset_mkt']['rshock'])

    onatskiLag  = ss0["phi"]*Dr[0,0]/ss0["A"]

    lambdas = np.linspace(0, 2*np.pi, 10000)
    valuesF = np.empty(10000, complex)
    for i in range(10000):
        valuesF[i] = sum((onatskiMatrix(x, C))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0, T-1)) + onatskiLag*math.e**(np.sqrt(-1+0j)*lambdas[i])
    return valuesF
# coefs = np.array([[dpi]])
# onatskiMatrix = onatski(coefs)

def windingNumberClockwise(F):
    return sum((-1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real > F[i-1].real)  else 0) for i in range(len(F))) 

def windingNumberCounterClockwise(F):
    return sum((1 if (F[i].imag > 0) and (F[i].real*F[i-1].real < 0) and (F[i].real < F[i-1].real)  else 0) for i in range(len(F)))

    #def windingNumber(F):
    #    return windingNumberClockwise(F) + windingNumberCounterClockwise(F)
    #detA = self.onatskiMatrix#onatski(coefs) #---> an array of A(L) values for L from 0 to 2pi  

    # print("Winding number: " + str(windingNumber(detA)))
    # print("Winding number CW (Multiple Solution): " + str(windingNumberClockwise(detA)))
    # print("Winding number CCW (No Solution): " + str(windingNumberCounterClockwise(detA)))

    # if windingNumber(detA) == 0:
    #     print("The economy is DETERMINATE")
    # elif windingNumber(detA) > 0:
    #     print("The economy is INDETERMINATE (NO SOLUTION)")    
    # elif windingNumber(detA) < 0:
    #     print("The economy is INDETERMINATE (MULTIPLE SOLUTION)") 
def onatskiWindingNumber(self):
    return windingNumberClockwise(self.onatskiMatrix) + windingNumberCounterClockwise(self.onatskiMatrix)
