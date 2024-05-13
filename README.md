# Local Determinacy (LD)

A python package for assessing local determinacy in incomplete markets models. The theoretical framework framework is based on the working paper [Local Determinacy in Incomplete-Markets Models, 2023 (Marcus Hagedorn)](https://drive.google.com/file/d/1gCMGgjyLEas3xmcxcBPyQ2vLwuBdzhLu/view).

## Methodology

- The first step in Hagedorn (23) is constructing the linearized version, ignoring exogenous shocks which are irrelevant for determinacy, of the incomplete-markets model
```math
\sum_{k=-j}^{\infty} M_k x_{t-k} = 0,
```

for a Period $t$ vector $x_t$ of endogenous variables and square matrices $(M_k)_{k=-j}^{\infty}$ describing equilibrium conditions.

-  In a simple incomplete-markets model the asset market clearing condition, $A(\pi_{t-1},\pi_t, \pi_{t+1}, \ldots, \pi_{t+k}, \ldots) - B = 0$ constitutes the only equilibrium condition. The linearized version of the asset market clearing condition

```math
		\sum_{k=-1}^{\infty} A_k E_t \pi_{t+k} = 0,
```	
 where inflation $\pi$ is the only endogenous variable

-  The SSJ package delivers the derivatives of the asset market clearing condition, $A(\pi_{t-1},\pi_t, \pi_{t+1}, \ldots, \pi_{t+k}, \ldots) - B = 0$,
```math
		A_k = \frac{\partial (A(\ldots) - B)}{\partial x_{t+k}} =  \frac{\partial (A(\ldots) - B)}{\partial \pi_{t-1+k}}, \text{ \hspace{0.2cm} where \hspace{0.1cm} $x_{t+k} = \pi_{t+k-1} \hspace{0.1cm}$, for \hspace{0.2cm}} k=-1,0,1,2,\ldots
```	

-  Onatski (06) defines the complex function
```math
			{\Theta(\lambda)} = det \sum_{k=-j}^{\infty} A_k e^{-i k \lambda},
```
- The Winding number is defined as  the number of times the graph of $\Theta(\lambda)$ rotates around zero counter-clockwise when $\lambda$ goes form $0$ to $2 \pi$.

- Ontaski (2006) shows
	
	- Determinacy (=unique bounded solution) if the winding number of $\Theta(\lambda)$ is equal to zero.

	- Multiple Solutions if winding number is less than zero.

	- No Solution if winding number is larger than zero.


## Requirements and installation

LD runs on Python 3.7 or newer, and requires Python's core numerical libraries (NumPy, SciPy, Numba).

LD uses the methodology developped in  Auclert, Bardóczy, Rognlie, Straub (2021): "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models" ([link to paper](https://www.bencebardoczy.com/publication/sequence-jacobian/sequence-jacobian.pdf)) to obtain the Jacobian matrix of the incomplete-markets model.

LD uses the SSJ package of Bence Bardóczy, Michael Cai, Matthew Rognlie, Adrien Auclert, Martin Souchier and Ludwig Straub. The latest version is available at https://github.com/shade-econ/sequence-jacobian. Note that all required packages will be installed automatically.

To install LD, open a terminal and type
```
pip install local-determinacy
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
Given the jacobian $H_U$ of a model, in which the asset market clering condition is the only equilibrium condition and lagged inflation $piL$ is the only endogenous vaiable. LD assesses local determinacy in a simple incomplete markets model as follows:
```
import local_determinacy as ld

T = 300   

unknowns = ['piL']
targets = ['asset_mkt']

H_U = ha.jacobian(ss, unknowns, targets, T=T)

onatski = ld.onatski(targets = targets, endogenous = unknowns, T = T, ss0 = ss0, H_U = H_U)

windingNumber = ld.onatskiWindingNumber(onatski)

windingNumber = ld.onatskiWindingNumber(onatski)
print(ld.checkSolutions(windingNumber))

ld.plot(onatski)
```
If government bonds are nominal, an addtional elasticity decscibing valuation effects need to be added.
The default setting assumes that all variables (for example lagged inflation) are predetermined. In general, LD requires specifying predetermined and Nnon-predetermnined variables.
The notebook examples illustrate these options.

<!-- ## Toolbox

The Toolbox allows using the package without a Python compiler. The toolbox offers several model classes and parameters (right now policy parameters) can be chosen by the user:

- One asset incomplete-markets model with flexible prices and real bonds
- One asset incomplete-markets model with sticky prices and real bonds
- One asset incomplete-markets model with flexible prices and nominal bonds
- One asset incomplete-markets model with sticky prices and nominal bonds
- One asset incomplete-markets model with flexible prices and FTPL tax rule
- One asset incomplete-markets model with sticky prices and FTPL tax rule
- ... -->

## Authors

This package was written by
- Alfred Løvgren
- Marcus Hagedorn
