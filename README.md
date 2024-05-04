# Local Determancy (LD)

A python package for assessing local determinacy in incomplete markets models. The theoretical framework framework is based on the working paper Local Determinacy in Incomplete-Markets Models, 2023 (Marcus Hagedorn) [available here](https://drive.google.com/file/d/1gCMGgjyLEas3xmcxcBPyQ2vLwuBdzhLu/view).

## Requirements and installation

LD runs on Python 3.7 or newer, and requires Python's core numerical libraries (NumPy, SciPy, Numba). LD relies on the SSJ package for defining models. You can find the latest version of SSJ here: https://github.com/shade-econ/sequence-jacobian. Note that all requirements will be installed automatically.

To install LD, open a terminal and type
```
pip install local-determancy
```

## Functions

The LD package has four main functions:
1) The onatski values of a jacobian
```
>>> onatski(targets, endogenous, scale, T, ss0, H_U)
#Returns a vector of Onatski function outputs
```

2) The winding number of a onatski function
```
>>> onatskiWindingNumber(onatski)
#Returns the winding number of a given sequence of Onatski function outputs
```

3) An assessment of local-determinacy
```
>>> checkSolutions(windingNumber)
#Returns a string assessment of local-determinacy
```

4) A plot of the Onatski function
```
>>> plot(Onatski)
#Returns a plot of the Onatski function in the complex space
```

## Usage

The LD package handles a variety of incomplete markets models. Please see the provided Jupyter notebooks for examples.
Given the jacobian of a model, LD assesses local determinacy as follows:
```
import local_determancy as ld

T = 300
exogenous = ['rstar', 'Z']
unknowns = ['pi', 'w', 'Y']
targets = ['nkpc_res', 'asset_mkt', 'labor_mkt']

H_U = hank.jacobian(ss, unknowns, targets, T=T)

onatski = ld.onatski(targets = targets, endogenous = unknowns, scale = 'A', T =T, ss0=ss0, H_U = H_U)

windingNumber = ld.onatskiWindingNumber(onatski)

windingNumber = ld.onatskiWindingNumber(onatski)
print(ld.checkSolutions(windingNumber))

ld.plot(onatski)
```

## Authors

This package was written by
- Alfred Løvgren
- Marcus Hagedorn
