import numpy as np

import matplotlib.pyplot as plt

from sequence_jacobian import simple, solved, combine, create_model  # functions

from sequence_jacobian import hetblocks, grids      # modules

from sequence_jacobian.classes import FactoredJacobianDict

from sequence_jacobian.classes.sparse_jacobians import make_matrix

import math

import matplotlib.pyplot as plt

import os

import sys



class Model:

    def __init__(self, calibration):

       self.calibration = calibration

       self.hank = None

       self.ss = None

       self.ss0 = None

       self.onatskiMatrix = None



    def calibrate(self):

        print("Calibration:", self.calibration)



        hh = hetblocks.hh_sim.hh



        def make_grids(rho_e, sd_e, n_e, min_a, max_a, n_a):

            e_grid, pi_e , Pi = grids.markov_rouwenhorst(rho_e, sd_e, n_e)

            a_grid = grids.asset_grid(min_a, max_a, n_a)

            return e_grid, pi_e, Pi, a_grid



        def income(w, Tax, Div, pi_e, e_grid, a_grid):

            tax_rule = e_grid

            tax = Tax / np.sum(pi_e * tax_rule) * tax_rule

            y = w * e_grid - tax 

            return y



        hh1 = hh.add_hetinputs([make_grids, income])



        hh_ext = hh1.add_hetoutputs



        @simple

        def monetary(piL, rstar, rshock, rscale, phi):

            rN = (1 + rstar(-1) + rscale* phi * piL) -1 + rscale* rshock

    #       rNdummy = rN -( (1 + rstar + rscale* phi * pi(-1)) -1 + rscale* rshock)

            return rN



        @simple

        def monetaryReal(piL, Realshock, rstar, phi, rN, Div, B):

            r = (1 + rN) / (1 + piL(+1)) - 1 + Realshock + Div/B

            return r



        @simple

        def fiscal(rN, B, w, piL, taushock, G):

            Tax = ((1 + rN) / (1 + piL(+1)) - 1 ) * B + taushock + G

            Div = (1 - w)*0  

            return Tax, Div



        @simple 

        def mkt_clearing(A, C, B,r):

            asset_mkt = A - B

            test_mkt = r - 0.005

            goods_mkt = 1 - C

            return asset_mkt, goods_mkt, test_mkt



        @simple

        def nkpc_ss(mu):

            w = 1/mu

            return w



        blocks_ss = [hh1,monetary, monetaryReal, fiscal, mkt_clearing, nkpc_ss]



        hank_ss = create_model(blocks_ss, name="One-Asset Simple HANK SS")



        unknowns_ss = {'beta': 0.986}



        targets_ss = {'asset_mkt': 0}



        self.ss0 = hank_ss.solve_steady_state(self.calibration, unknowns_ss, targets_ss, solver="hybr")



        blocks = [hh1,monetary, monetaryReal, fiscal, mkt_clearing, nkpc_ss]



        self.hank = create_model(blocks, name="One-Asset Simple HANK")



        self.ss = self.hank.steady_state(self.ss0)



        print("Computations Done")



        print(self.ss0["rho_e"])



        print(self.ss0["phi"])



        # setup

    def onatski(self):
        T = 300

        exogenous = ['rstar','rshock','taushock','Realshock']

        unknowns = ['piL']

        targets = ['asset_mkt','goods_mkt','test_mkt']


        H_U = self.hank.jacobian(self.ss, unknowns, targets, T=T)

        H_Z = self.hank.jacobian(self.ss, exogenous, targets, T=T)

        DReal=np.array(H_Z['asset_mkt']['Realshock'])

        Dtau=np.array(H_Z['asset_mkt']['taushock'])



        Dr= np.array(H_Z['asset_mkt']['rshock'])



        DPi=np.array(H_U['asset_mkt']['piL'])



        Dmpc=np.array(H_Z['goods_mkt']['taushock'])



        #dr = Dr * (1+0.005)/ss0["A
        dpi = DPi /self.ss0["A"]

        dtau = Dtau 

        dReal = DReal*(1+self.ss0["rstar"]) /self.ss0["A"]

        dmpc = Dmpc
        onatskiLag  = self.ss0["phi"]*Dr[0,0]/self.ss0["A"]
        Onatskicoeff = np.empty(T)

        for x in range(0,T-1):
            Onatskicoeff[x] =  -dtau[0,x]*(1+self.ss0["rstar"])-dReal[0,x]+ ((1+self.ss0["rstar"])*dtau[0,x+1]+dReal[0,x+1])*self.ss0["phi"]      

        onatskidummy =  (dtau[0,0]*(1+self.ss0["rstar"])+dReal[0,0])*self.ss0["phi"]

        def onatskiMatrix(x, C):
            dM = np.empty((0,len(C[0])), int)
            for i in range(len(C)):
                tmp = np.empty(0, int)
                for j in range(len(C[i])):
                    tmp = np.append(tmp, C[i][j][0,x])
                dM = np.append(dM, [tmp], axis=0)
            return dM

        def onatski(C):
            lambdas = np.linspace(0, 2*np.pi, 10000)
            valuesF = np.empty(10000, complex)
            for i in range(10000):
                valuesF[i] = sum((onatskiMatrix(x, C))*math.e**(-np.sqrt(-1+0j)*(x-1)*lambdas[i]) for x in range(0, T-1))  + onatskiLag*math.e**(np.sqrt(-1+0j)*lambdas[i])
            return valuesF

        coefs = np.array([[dpi]])

        self.onatskiMatrix = onatski(coefs)



    def onatskiMatrix(self):

        return self.onatskiMatrix



    def onatskiWindingNumber(self):

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



        return windingNumberClockwise(self.onatskiMatrix) + windingNumberCounterClockwise(self.onatskiMatrix)



    def onatskiPlot(self):

        plt.plot(self.onatskiMatrix.real, self.onatskiMatrix.imag, color='blue',linewidth=3)



        plt.xlabel('Real - axis', fontsize=18)



        plt.ylabel('Imaginary - axis', fontsize=18)



        plt.title('Onatski graph')



        angle = np.deg2rad(45)



        cross_length = max(max(self.onatskiMatrix.real)-min(self.onatskiMatrix.real), max(self.onatskiMatrix.imag)-min(self.onatskiMatrix.imag)) * 0.065



        plt.plot([-cross_length * np.cos(angle), cross_length * np.cos(angle)],

                [-cross_length * np.sin(angle), cross_length * np.sin(angle)],

                color='red', linestyle='-', linewidth=3)  # Diagonal line /



        plt.plot([-cross_length * np.cos(angle), cross_length * np.cos(angle)],

                [cross_length * np.sin(angle), -cross_length * np.sin(angle)],

                color='red', linestyle='-', linewidth=3)  # Diagonal line \





        plt.show()

        # Get the filename of the notebook without extension



        #notebook_filename = os.path.splitext(os.path.basename(get_ipython().shell.filename))[0]



        # Get the notebook name without extension



        #notebook_name = os.path.splitext(os.path.basename(get_ipython().shell.display_pub.publishers['notebook'].filename))[0]



        # Get the notebook name without extension

        #notebook_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]



        if self.ss0["rho_e"] == 0.945:

            filename = f"REAL_FLEX_MPCLOW_phi{self.ss0['phi']}_Winding{self.onatskiWindingNumber()}"



        if self.ss0["rho_e"] == 0.9878:

            filename = f"REAL_FLEX_MPCHIGH_phi{self.ss0['phi']}_Winding{self.onatskiWindingNumber()}"



        # Create a subfolder if it doesn't exist

        subfolder = "Results_Paper"



        if not os.path.exists(subfolder):

            os.makedirs(subfolder)



        # Save the plot as a PNG file in the subfolder

        plt.savefig(os.path.join(subfolder, f"{filename}.png"))



        # Save the plot as a PNG file with the notebook's filename

        #plt.savefig(f"{filename}.png")



        

