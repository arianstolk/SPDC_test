#################################################################
##############	Supporting physics module	#####################
#################################################################

#################################################################
#### DIMENSIONS USED:
####	-Wavelength in [nm], because lambda is protected lam is used
####	-Dimensions in [mm]
####	-Time 		in [s] or [fs]
####	-Angles 	in	[] (rad)
##################################################################

import math
import sympy as sp
import numpy as np

def Sellmeier(coeff=[0,0,0,0],lam = 785):
	
	return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5

def DSellmeier(coeff=[0,0,0,0],lam = 785):
	x = sp.Symbol('x')
	ydiff = sp.diff(Sellmeier(coeff,x),x)
	f = sp.lambdify(x, ydiff, 'numpy')
	
	return f(lam)

def n_ext_effective(no=1,ne=1,theta=0):

	return ((np.sin(theta)/ne)**2+(np.cos(theta)/no)**2)**(-0.5)

def walkoff(cutangle=0,coeff=[0,0,0,0,0,0,0,0],lam=785,thickness=1):

	if len(coeff)<8:
		print("Please provide all 8 Sellmeier coeffs")
	else:
		n_ord=Sellmeier(coeff[0:4],lam)
		n_ext=Sellmeier(coeff[4:8],lam)

	return 0.5*thickness*(n_ext_effective(theta=cutangle,no=n_ord,ne=n_ext)**2)*(n_ord**(-2)-n_ext**(-2))*np.sin(2*cutangle)

def v_group(lam=785,n=1,dn=0):

	return (2.998e11)/(n-lam*dn)

