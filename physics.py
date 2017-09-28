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
from scipy import interpolate
import numpy as np
from sklearn.preprocessing import normalize
import random

import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


def Sellmeier(coeff=[0,0,0,0],lam = 785):
	
	return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5

# def dSellmeier(coeff=[0,0,0,0],lam = 785):
# 	c = coeff
# 	return ((1e-6)*lam*(-c[3]-(c[2])/(c[2]-(1e-6)*lam**2)))/(c[0]-(1e-6)*c[3]*lam**2-(c[2])/(c[2]-(1e-6)*lam**2))

# def v_group(lam=785,n=1,dn=0):

# 	return (2.998e11)/(n-lam*dn)

# def dn_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
# 	c=coeff
# 	return -((1e-6)*lam*(((c[1]+c[3]*(c[2]-(1e-6)*lam**2)**2)*cos(theta)**2)/(c[1]+(c[2]-(1e-6)*lam**2)*(-c[0]+(1e-6)*c[3]*lam**2))**2\
# 			+(((c[5]+c[7]*(c[6]-(1e-6)*lam**2)**2)*sin(theta)**2)/(c[5]+(c[6]-(1e-6)*lam**2)*(-c[4]+(1e-6)*c[3]*lam**2))**2))\
# 			/((cos(theta)**2)/(c[0]-(1e-6)*c[3]*lam**2-c[1]/(c[2]-(1e-6)*lam**2))+(sin(theta)**2)/(c[4]-(1e-6)*c[7]*lam**2-c[5]/(c[6]-(1e-6)*lam**2)))**(3/2))


# def n_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
# 	c=coeff
# 	return ((sin(theta)/(c[4]+(c[5])/((lam*1e-3)**2-c[6])-(c[7])*(lam*1e-3)**2)**0.5)**2+(cos(theta)/(c[0]+(c[1])/((lam*1e-3)**2-c[2])-(c[3])*(lam*1e-3)**2)**0.5)**2)**(-0.5)

# def walkoff(theta=0,coeff=[0,0,0,0,0,0,0,0],lam=785,thickness=1):

# 	# if len(coeff)<8:
# 	# 	print("Please provide all 8 Sellmeier coeffs")
# 	# else:
# 	n_ord=Sellmeier(coeff[0:4],lam)
# 	n_ext=Sellmeier(coeff[4:8],lam)
# 	return 0.5*thickness*(n_ext_effective(coeff=coeff,theta=theta,lam=lam)**2)*((n_ord**-2)-(n_ext**-2))*sin(2*theta)

def phasefunction(lpump=405,l_range=785,theta_range=0,coeff=[2.7359, 0.01878, 0.01822, 0.01354, 2.3753, 0.01224, 0.01667, 0.01516],cutangle=28.7991*np.pi/180,crystal_length=6,
	show_plot=False,return_grid=False):
	if isinstance(l_range,list):
		lsignal=l_range[2]*np.array(list(range(round((1/l_range[2])*l_range[0]),round((1/l_range[2])*l_range[1]),1)))
		lamsize=len(lsignal)

	else:
		lsignal=np.array([l_range])
		lamsize=1
		

	if isinstance(theta_range,list):
		theta_o=(np.pi/180)*(theta_range[2])*np.array(list(range(round(1/(theta_range[2])*theta_range[0]),round(1/(theta_range[2])*theta_range[1]),1)))
		thetasize=len(theta_o)
		
	else:
		theta_o=np.array([theta_range])
		thetasize=1
		

	c=coeff

	lp=lpump
	ls=lsignal
	li=ls*lp/(ls-lp);

	wp,ws,wi = (2*np.pi)/lp,(2*np.pi)/ls,(2*np.pi)/li

	nop = Sellmeier(coeff=c[0:4],lam=lp)
	nep = Sellmeier(coeff=c[4:8],lam=lp)
	nos = Sellmeier(coeff=c[0:4],lam=ls)
	nes = Sellmeier(coeff=c[4:8],lam=ls)
	noi = Sellmeier(coeff=c[0:4],lam=li)
	nei = Sellmeier(coeff=c[4:8],lam=li)

	nairs = 1+(0.05792105)/(238-(ls*1e-3)**-2)+(0.00167917)/(57.362-(ls*1e-3)**-2)
	thet_cut=cutangle
	npeff=np.sqrt(1/((np.cos(thet_cut)/nop)**2+(np.sin(thet_cut)/nep)**2))
	L=crystal_length
	W=0.1



	wsmesh,theta_omesh=np.meshgrid(ws,theta_o)
	wimesh,theta_omesh=np.meshgrid(wi,theta_o)

	thet_s=theta_omesh
	thet_i=np.arcsin(nos*wsmesh*np.sin(thet_s)/(noi*wimesh))
	dkz=npeff*wp-nos*wsmesh*np.cos(thet_s)-noi*wimesh*np.cos(thet_i)
	dky=-nos*wsmesh*np.sin(thet_s)+noi*wimesh*np.sin(thet_i)
	phi=np.exp(-((W*1e6)**2)*(dky**2)/2)*(np.sin(0.5*dkz*L*1e6)/(0.5*dkz*L*1e6))**2

	result = phi

	if show_plot:
		
		if lamsize == 1:
			plt.figure()
			plt.plot(theta_o*(180/np.pi),result)
			plt.xlabel('opening angle [*]')
			plt.show()

		elif thetasize==1:

			plt.figure()
			plt.plot(lsignal,result[0])
			plt.xlabel("wavelength [nm]")
			plt.show()

		else:
			plt.figure()
			CS=plt.contourf(lsignal,(180/np.pi)*theta_o,result,levels=np.linspace(0,1,20))
			plt.xlabel('wavelength [nm]')
			plt.ylabel('opening angle [*]')
			plt.show()
	
	if not show_plot:
		if not return_grid:
			if lamsize == 1 and thetasize == 1:
				return float(result)
			else:
				return result
		else:
			return (lsignal,theta_o,result)


def phasefunction_vec(lpump=405,l_list=[785],theta_list=[0],coeff=[2.7359, 0.01878, 0.01822, 0.01354, 2.3753, 0.01224, 0.01667, 0.01516],cutangle=28.7991*np.pi/180,crystal_length=6):

	c=coeff

	lp=lpump
	ls=l_list
	li=ls*lp/(ls-lp);

	theta_o=theta_list

	wp,ws,wi = (2*np.pi)/lp,(2*np.pi)/ls,(2*np.pi)/li

	nop = Sellmeier(coeff=c[0:4],lam=lp)
	nep = Sellmeier(coeff=c[4:8],lam=lp)
	nos = Sellmeier(coeff=c[0:4],lam=ls)
	nes = Sellmeier(coeff=c[4:8],lam=ls)
	noi = Sellmeier(coeff=c[0:4],lam=li)
	nei = Sellmeier(coeff=c[4:8],lam=li)

	npeff=np.sqrt(1/((np.cos(cutangle)/nop)**2+(np.sin(cutangle)/nep)**2))
	L=crystal_length
	W=0.1

	thet_s=theta_o
	thet_i=np.arcsin(nos*ws*np.sin(thet_s)/(noi*wi))
	dkz=npeff*wp-nos*ws*np.cos(thet_s)-noi*wi*np.cos(thet_i)
	dky=-nos*ws*np.sin(thet_s)+noi*wi*np.sin(thet_i)
	phi=np.exp(-((W*1e6)**2)*(dky**2)/2)*(np.sin(0.5*dkz*L*1e6)/(0.5*dkz*L*1e6))**2

	return phi

	


def get_N_from_phasefunc(N=1,factor=1e-3,lpump=405,l_min=785,l_max=900,theta_min=-3,theta_max=3,cutangle=28.76*np.pi/180):
	
	randN_l		= np.random.uniform(l_min,l_max,N)
	randN_theta	= np.random.uniform(theta_min,theta_max,N)*np.pi/180
	rand_check	= np.random.uniform(0,1,N)

	weight=factor*phasefunction_vec(l_list=randN_l,theta_list=randN_theta,lpump=lpump)

	doubles = np.array([randN_l,randN_theta]).T

	return doubles[np.where(weight>rand_check)]



def get_SPDC_rayset_adv(self,N=1,nr_crystals=1,pumpray=[],pump_waist=[0,0],pump_focus=[0,0],cutangle=28.76*np.pi/180,l_min=0,l_max=0,theta_min=0,theta_max=0,factor=1e-2):
	
	ray_list = []
	lpump=pumpray.wavelength

	rand_u = np.random.uniform(0,1,N)
	rand_uPhase = np.random.uniform(0,1,N)

	randN_l		= np.random.uniform(l_min,lpump/2,N)
	randN_theta	= np.random.uniform(theta_min,theta_max,N)*np.pi/180
	rand_check	= np.random.uniform(0,1,N)

	w0x=pump_waist[0]
	w0y=pump_waist[1]

	zR_pump_x=(np.pi*w0x**2)/(lpump*1e-6)
	zR_pump_y=(np.pi*w0y**2)/(lpump*1e-6)


	for i in range(nr_crystals):
		
		crystal=self.setup.crystals[i]
		pumpray.position = np.array([pumpray.position[0],pumpray.position[1],crystal.fsurf])
		w_beg=pumpray.position
		w_end=w_beg+crystal.getwalkoff(pumpray)+np.array([0,0,crystal.thickness])
		pumpray.position += crystal.getwalkoff(pumpray)

		start_pos = np.repeat(np.array([w_beg]),N,axis=0) + np.outer(rand_u,(w_beg - w_end))
		
		start_z = (start_pos.T)[2]

		"Introduce effects of gaussian pump"
		
		wzx,wzy = w0x*np.sqrt(1+((start_z-pump_focus[0])/zR_pump_x)**2),w0y*np.sqrt(1+((start_z-pump_focus[1])/zR_pump_y)**2)

		sx,sy=wzx/sqrt(2),wzy/sqrt(2)
	
		gauss_x=np.random.normal(np.zeros((1,N)),sx)
		gauss_y=np.random.normal(np.zeros((1,N)),sy)

		gauss_xy=np.concatenate((gauss_x,gauss_y)).T

		start_angles_gauss=np.concatenate((-np.arcsin(gauss_x/(start_z-pump_focus[0]+((zR_pump_x)**2)/(start_z-pump_focus[0]))),-np.arcsin(gauss_y/(start_z-pump_focus[1]+((zR_pump_y)**2)/(start_z-pump_focus[1]))))).T

		#"Sampling the phasefunction for the N randomly generated angles. the weight will later be used to accept or reject samples"
		weight=factor*phasefunction_vec(l_list=randN_l,theta_list=randN_theta,lpump=lpump,cutangle=cutangle-(start_angles_gauss.T)[0])

		#"Select the samples that will be used for signal/idler generation"
		gauss_pad=np.zeros(start_pos.shape)
		gauss_pad[:gauss_xy.shape[0],:gauss_xy.shape[1]]=gauss_xy

		sampled_params=np.concatenate((start_pos+gauss_pad,start_angles_gauss,np.array([randN_l]).T,np.array([randN_theta]).T),axis=1)#list of important simulation parameters

		final_params=sampled_params[np.where(weight>rand_check)]#filters the params for the succesfully drawn samples
		Nsuc=len(final_params)

		[ray_start,ray_angle,ls,opening_angle]=[final_params[:,0:3].T,final_params[:,3:5].T,final_params[:,5:6].T,final_params[:,6:7].T]

		rand_nr = np.random.uniform(0,1,len(final_params))
		li = ls*(lpump)/(ls-lpump)
		

		azim=np.cos(rand_nr*np.pi)*opening_angle
		horiz=np.sign(azim)*np.sign(opening_angle)*np.sqrt(opening_angle**2 - azim**2)

		prop_angles=np.concatenate((azim,horiz)).T
		

		"Assembling the final list of parameters, combining all the angular and spatial effects"

		Srays=[ray_start,prop_angles+ray_angle.T,ls]
		Irays=[ray_start,-prop_angles+ray_angle.T,li]
	
	print("jippie no errors,{},{}".format(len(ls.T),len(li.T)))


def generate_N_rays(N=1,polarization="V",ls=[],theta_o=[],pos_beg=[0,0],pos_end=[1,1],pumpray=[]):

	N = len(ls)
	rand_pos = np.random.uniform(0,1,2*N)
	li = ls*(pumpray.wavelength)/(ls-pumpray.wavelength)
	wave_double=list(zip(li,ls))

	li_max = [max(x) for x in wave_double]
	ls_min = [min(x) for x in wave_double]

	li=li_max
	ls=ls_min

	azim=np.cos(rand_pos[N-1:-1]*np.pi)*theta_o
	horiz=np.sign(azim)*np.sign(theta_o)*np.sqrt(theta_o**2 - azim**2)

	angle_list=np.array([azim,horiz]).T

	return [[Ray(position=[pos_beg[0]+rand_pos[i-1]*(pos_end[0]-pos_beg[0]),pos_beg[1]+rand_pos[i-1]*(pos_end[1]-pos_beg[1]),pos_beg[2]+(pos_end[2]-pos_beg[2])*rand_pos[i-1]],name="Sig_ray_{}".format(i),angles=angle_list[i],wavelength=ls[i],polarization = polarization) for i in range(N)],
			 [Ray(position=[pos_beg[0]+rand_pos[i-1]*(pos_end[0]-pos_beg[0]),pos_beg[1]+rand_pos[i-1]*(pos_end[1]-pos_beg[1]),pos_beg[2]+(pos_end[2]-pos_beg[2])*rand_pos[i-1]],name="Idl_ray_{}".format(i),angles=-1*angle_list[i],wavelength=li[i],polarization = polarization) for i in range(N)]]







