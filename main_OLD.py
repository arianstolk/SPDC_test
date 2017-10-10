#########################################################
### Analysis/Ray-tracing module for Bulk SPDC sources ###
### Written by: Arian Stolk							  ###
### Date: September 2017							  ###
#########################################################

###The goal of this project is to provide a tool that will allow the investigation of SPDC sources made of bulk non-linear crystals.###
###Import library####

from math import *
import random
import numpy as np
import sympy as sp
import matplotlib
import collections
from itertools import compress

import matplotlib.pyplot as plt
import time, sys
from vpython import *
import multiprocessing as mp


debug=False
################################################################################################################
def Sellmeier(coeff=[0,0,0,0],lam = 785):
	
	return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5

def dSellmeier(coeff=[0,0,0,0],lam = 785):
	c = coeff
	return ((1e-6)*lam*(-c[3]-(c[1])/(c[2]-(1e-6)*lam**2)**2))/sqrt((c[0]-(1e-6)*c[3]*lam**2-(c[1])/(c[2]-(1e-6)*lam**2)))

def v_group(lam=785,n=1,dn=0):

	return (2.998e11)/(n-lam*dn)

def dn_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
	c=coeff
	return -((1e-6)*lam*(((c[1]+c[3]*(c[2]-(1e-6)*lam**2)**2)*cos(theta)**2)/(c[1]+(c[2]-(1e-6)*lam**2)*(-c[0]+(1e-6)*c[3]*lam**2))**2\
			+(((c[5]+c[7]*(c[6]-(1e-6)*lam**2)**2)*sin(theta)**2)/(c[5]+(c[6]-(1e-6)*lam**2)*(-c[4]+(1e-6)*c[3]*lam**2))**2))\
			/((cos(theta)**2)/(c[0]-(1e-6)*c[3]*lam**2-c[1]/(c[2]-(1e-6)*lam**2))+(sin(theta)**2)/(c[4]-(1e-6)*c[7]*lam**2-c[5]/(c[6]-(1e-6)*lam**2)))**(3/2))

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def n_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
	c=coeff
	return ((sin(theta)/(c[4]+(c[5])/((lam*1e-3)**2-c[6])-(c[7])*(lam*1e-3)**2)**0.5)**2+(cos(theta)/(c[0]+(c[1])/((lam*1e-3)**2-c[2])-(c[3])*(lam*1e-3)**2)**0.5)**2)**(-0.5)

def walkoff(theta=0,coeff=[0,0,0,0,0,0,0,0],lam=785,thickness=1):

	# if len(coeff)<8:
	# 	print("Please provide all 8 Sellmeier coeffs")
	# else:
	n_ord=Sellmeier(coeff[0:4],lam)
	n_ext=Sellmeier(coeff[4:8],lam)
	return 0.5*thickness*(n_ext_effective(coeff=coeff,theta=theta,lam=lam)**2)*((n_ord**-2)-(n_ext**-2))*sin(2*theta)

def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

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



##################################################################################################################
class Simulation(object):
	"This class describes the interaction of Ray and ExpSetup, where the Rays are traced through the setup using the methods described in this class"
	debug=False

	def __init__(self,debug=debug,default_rays=False,**k):
		try:
			self.setup=k.pop('setup')
			self.rays=k.pop('rays',None)


			if not type(self.rays[0])==list:
				self.rays=[self.rays]

			if default_rays:
				print("Please use the get_SPDC_rayset function to generate your rays")

			self.numrays=len(self.rays)
			self.numobj=self.setup.nr_elements
			self.num_interfaces=len(self.setup.grouped_elements)

			self.store_path = k.pop("store_path",False)
			self.store_time = k.pop("store_time",False)

		except KeyError:
			print("Please provide rays and setup to initiate the simulation")

	def __repr__(self):
		return "Simulation of {} rays in a setup with {} elements".format(self.numrays,self.numobj)

	def run(self,store=False,multithread=False):

		
			for raylist in self.rays:
				if not multithread:
					print("Running simulation for {} rays throug {} optical elements. Storing result is: {}".format(len(raylist),self.numobj,store))
					for ray in raylist:
						self.complete_ray_propagation(ray)
		
		
		# else:
		# 	print("Running multi process")
		# 	pool_size=8
		# 	pool=mp.Pool(processes=pool_size)
		# 	pool.map(self.complete_ray_propagation,self.rays)
		# 	pool.close()
		# 	pool.join()
	def show_filtered(self,time_diff=False,bins=50):

		if not time_diff:
			timelist=[[x.arrivaltime()*(1e15) for x in raylist] for raylist in self.filtered_rays]
		else:
			arrivtimes =[(1e15)*np.array([x.arrivaltime() for x in raylist]) for raylist in self.filtered_rays]
			timelist = [arrivtimes[0]-arrivtimes[1],arrivtimes[2]-arrivtimes[3]]
		 
		plt.figure()

		
		for times in timelist:
			plt.hist(times,bins,alpha=0.5)
			print(np.mean(times))
		
		plt.xlabel("arrival time with respect to earliest photon [fs]")
		plt.ylabel("Occurance")
		plt.show()


	
	def showtimes(self,time_diff=False,bins=50):
		if not time_diff:
			timelist=(1e15)*(np.array([[x.arrivaltime() for x in raylist] for raylist in self.rays]))

		else:
			arrivtimes = ((1e15)*np.array([[x.arrivaltime() for x in raylist] for raylist in self.rays]))
			timelist = np.array([arrivtimes[0]-arrivtimes[1],arrivtimes[2]-arrivtimes[3]])
		
		plt.figure()
		
		for times in timelist:
			plt.hist(times,bins,alpha=0.5)
			print(np.mean(times),len(times))
		
		plt.xlabel("arrival time with respect to earliest photon [fs]")
		plt.ylabel("Occurance")
		plt.show()

	

	def refract(ray,optic_elements=[]):
		element1=optic_elements[0]
		element2=optic_elements[1]
		nin=element1.getn(ray)
		nout=element2.getn(ray)
		# print(nin,nout,ray.angles,ray.position,element1.position,element2.position)

		if not nin == nout:
			if hasattr(element1,"ROC"):
				pos=ray.position-element1.position
				ray.angles =  [asin(sin(x)*(nin/nout))+pos[i]*((nin-nout)/(nout*element1.ROC[1])) for i,x in enumerate(ray.angles)]
			elif hasattr(element2,"ROC"):
				pos=ray.position-element2.position
				ray.angles =  [asin(sin(x)*(nin/nout))+pos[i]*((nin-nout)/(nout*element2.ROC[0])) for i,x in enumerate(ray.angles)]
			else:
				ray.angles = [asin(sin(x)*(nin/nout)) for x in ray.angles]

	def translate(ray,optic_element):

		if not ray.position[2]>optic_element.bsurf:
			optic_element.translate(ray)
		
		if debug:
			print("I traced {} in {} to {}".format(ray,optic_element,ray.position))

	def propagate(ray,optic_elements):
		Simulation.translate(ray,optic_elements[0])
		Simulation.refract(ray,optic_elements)

	def complete_ray_propagation(self,ray):
		grouped_elements = self.setup.grouped_elements
		if debug:
			print("I am going to trace {} through {} ".format(ray.showRay(),grouped_elements))
			

		if self.store_path == True and self.store_time == True:
			poslist=list(np.array([ray.position]))
			timelist=list(ray.time)
			
			for elements in grouped_elements:
				cur_pos=poslist[-1]
				Simulation.propagate(ray,elements)
				dt=(((ray.position[0]-cur_pos[0])**2+(ray.position[1]-cur_pos[1])**2+(ray.position[2]-cur_pos[2])**2)**0.5)/v_group(lam=ray.wavelength,n=elements[0].getn(ray),dn=elements[0].getdn(ray))
				# print('I have added {} to the timelist'.format(dt))
				
				poslist.append(np.array(ray.position))
				timelist.append(dt)

			ray.time=timelist
			ray.path=poslist
		
				
		elif self.store_path == True:
			
			poslist=list(np.array([ray.position]))
			
			for elements in grouped_elements:
				Simulation.propagate(ray,elements)				
				poslist.append(ray.position)
			ray.path=poslist	
			

		elif self.store_time == True:
			timelist=list(ray.time)

			for elements in grouped_elements:
				cur_pos=poslist[-1]
				Simulation.propagate(ray,elements)
				dt=(((ray.position[0]-cur_pos[0])**2+(ray.position[1]-cur_pos[1])**2+(ray.position[2]-cur_pos[2])**2)**0.5)/v_group(lam=ray.wavelength,n=elements[0].getn(ray),dn=elements[0].getdn(ray))
				
				timelist.append(dt)
		

		elif self.store_path == False and self.store_time == False:

			for elements in grouped_elements:
				Simulation.propagate(ray,elements)

	def get_SPDC_rayset_adv(self,N=1,nr_crystals=1,pumpray=[],pump_waist=[0,0],pump_focus=[0,0],cutangle=28.8*np.pi/180,l_min=0,l_max=0,theta_min=0,theta_max=0,factor=1e-2):
		
		ray_list = []
		lpump=pumpray.wavelength

		rand_u = np.random.uniform(0,1,N)
		rand_uPhase = np.random.uniform(0,1,N)

		randN_l		= np.random.uniform(l_min,2*lpump,N)
		randN_theta	= np.random.uniform(theta_min,theta_max,N)*np.pi/180
		rand_check	= np.random.uniform(0,1,N)

		w0x=pump_waist[0]
		w0y=pump_waist[1]

		zR_pump_x=(np.pi*w0x**2)/(lpump*1e-6)
		zR_pump_y=(np.pi*w0y**2)/(lpump*1e-6)

		lreturn_list=[]

		for i in range(nr_crystals):
			
			crystal=self.setup.crystals[i]
			pumpray.position = np.array([pumpray.position[0],pumpray.position[1],crystal.fsurf])
			w_beg=pumpray.position
			w_end=w_beg+crystal.getwalkoff(pumpray)+np.array([0,0,crystal.thickness])
			pumpray.position += crystal.getwalkoff(pumpray)

			start_pos = np.repeat(np.array([w_beg]),N,axis=0) + np.outer(rand_u,(w_end-w_beg))

			
			start_z = (start_pos.T)[2]

			plt.figure()
			plt.hist(start_z-pump_focus[0])
			plt.show

			"Introduce effects of gaussian pump"
			
			wzx,wzy = w0x*np.sqrt(1+((start_z-pump_focus[0])/zR_pump_x)**2),w0y*np.sqrt(1+((start_z-pump_focus[1])/zR_pump_y)**2)

			sx,sy=wzx,wzy
		
			gauss_x=np.random.normal(np.zeros((1,N)),sx)
			gauss_y=np.random.normal(np.zeros((1,N)),sy)

			gauss_xy=np.concatenate((gauss_x,gauss_y)).T

			zpos_rel_wx=np.where(np.abs(-pump_focus[0]+start_z)>1e-9,(pump_focus[0]+start_z),1e-9)
			zpos_rel_wy=np.where(np.abs(-pump_focus[1]+start_z)>1e-9,(pump_focus[1]+start_z),1e-9)

			start_angles_gauss=np.concatenate((-np.arcsin(gauss_x/(zpos_rel_wx+((zR_pump_x)**2)/(zpos_rel_wx))),-np.arcsin(gauss_y/(zpos_rel_wy+((zR_pump_y)**2)/(zpos_rel_wy)))).T

			plt.figure()
			plt.hist(start_angles_gauss.T[0])
			plt.show()

			#"Sampling the phasefunction for the N randomly generated angles. the weight will later be used to accept or reject samples"
			weight=factor*phasefunction_vec(l_list=randN_l,theta_list=randN_theta,lpump=lpump,cutangle=cutangle+(start_angles_gauss.T)[0])

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

			lreturn_list.append(np.concatenate((ls.T,li.T)))
		
		return lreturn_list


	def get_SPDC_rayset(self,N=1,nr_crystals=1,pumpray=[],pumpw0=0,pumpfocus=0,ls=[],theta_o=[]):

		ray_list = []
	
		for i in range(nr_crystals):
			
			crystal=self.setup.crystals[i]
			pumpray.position = np.array([pumpray.position[0],pumpray.position[1],crystal.fsurf])
			walkoff_line_begin=pumpray.position

			walkoff_line_end=walkoff_line_begin+crystal.getwalkoff(pumpray)+np.array([0,0,crystal.thickness])
			ray_list.append(self.generate_N_rays(pumpray=pumpray,pos_beg=walkoff_line_begin,pos_end=walkoff_line_end,polarization = 'V',ls=ls,theta_o=theta_o))
			pumpray.position += crystal.getwalkoff(pumpray)

		return [rayset for crystalset in ray_list for rayset in crystalset]

	def generate_N_rays(N=1,polarization="V",ls=[],theta_o=[],pos_beg=[0,0],pos_end=[1,1],pumpray=[]):

		N = len(ls)
		rand_pos	= np.random.uniform(0,1,2*N)
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

	def filter_results(self,fibre_pos=np.array([0,0,0]),core_diam=0,Num_Ap=0,raw_return=True):
		"Filter function giving singles and coincidences that enter the fibre at fibre_pos with specified dimensions"

		filter_cond_match_list=[]
		singles=[]
		for ray_set in self.rays:
			filter_cond_match=self.filter_func(rays=ray_set,fibre_pos=fibre_pos,core_diam=core_diam, Num_Ap = Num_Ap)
			singles.append(list(compress(ray_set,filter_cond_match)))
			filter_cond_match_list.append(filter_cond_match)

		coincidences=[list(compress(self.rays[0],[x[0] and x[1] for x in zip(filter_cond_match_list[0],filter_cond_match_list[1])])),list(compress(self.rays[1],[x[0] and x[1] for x in zip(filter_cond_match_list[0],filter_cond_match_list[1])])),list(compress(self.rays[2],[x[0] and x[1] for x in zip(filter_cond_match_list[2],filter_cond_match_list[3])])),list(compress(self.rays[3],[x[0] and x[1] for x in zip(filter_cond_match_list[2],filter_cond_match_list[3])]))]

		self.filtered_rays=map(np.array,coincidences)
		print("added filtered singles to simulation,coincidences are {} and {},singles are {} and {}".format(len(coincidences[0]),len(coincidences[2]),len(singles[0])+len(singles[1]),len(singles[2])+len(singles[3])))


	def filter_func(self,rays=[],fibre_pos=np.array([0,0,0]),core_diam=0,Num_Ap=0):

		ray_pos_array 		=	np.array([ray.position for ray in rays])
		ray_angles_array	=	np.array([ray.angles for ray in rays])

		core_dist_check=np.linalg.norm(ray_pos_array-fibre_pos,axis=1)<core_diam/2
		inc_angle_check=np.linalg.norm(ray_angles_array,axis=1)<Num_Ap

		truth_list=[x[0] and x[1] for x in zip(core_dist_check,inc_angle_check)]

		return truth_list



class ExpSetup(object):
	"All instances of this class are a set of Optic elements along the z-direction. This class will handle all the preparation of the Optic elements to make them ready for the sequential ray tracing"

	def __init__(self,*args):
		if len(args)==0:
			raise Exception("Please provide the optical elements for the setup")
		self.elements = list(flatten([args]))
		self.crystals = [x for x in self.elements if type(x) == Crystal]		
		self.nr_elements = len(self.elements)
		self.insert_air()
		self.group()

	def __repr__(self):
		return "Experimental Setup"
	
	def add_element(self,optic_element):
		self.elements.append(optic_element)
		self.nr_elements +=1

	def insert_air(self):
		"Inserts air pieces in the setup between the crystals when the distance between the crystals is nonzero"		
		if self.nr_elements == 0:
			print("Please add a crystal to the experimental setup")
		elif self.nr_elements == 1:
			if self.elements[0].fsurf > 0:
				self.elements.insert(0,Optic(name="Air_Begin",material="Air",position=np.array([0,0,(self.elements[0].fsurf)/2]),thickness=self.elements[0].fsurf))
			self.elements.insert(-1,Optic(name="Air_End",material="Air",position=np.array([0,0,self.elements[0].bsurf+2.5]),thickness=5))

		else:
			for i in range(self.nr_elements-1,0,-1):
				gap =  self.elements[i].fsurf - self.elements[i-1].bsurf
				if gap > 0:
					self.elements.insert(i,Optic(name="Air_after_{}".format(self.elements[i].name),material="Air",position=np.array([0,0,self.elements[i-1].bsurf+0.5*(self.elements[i].fsurf-self.elements[i-1].bsurf)]),thickness=gap))
			if self.elements[0].fsurf > 0:
				self.elements.insert(0,Optic(name="Air_Begin",material="Air",position=np.array([0,0,(self.elements[0].fsurf)/2]),thickness=self.elements[0].fsurf))
			self.elements.append(Optic(name="Air_End",material="Air",position=np.array([0,0,self.elements[-1].bsurf+2.5]),thickness=5))
	
	def visualize(self,centre = [0,0,0]):
		boxes=[]
		for elem in self.elements:
			boxes.append(box(pos=vec(0-centre[0],0-centre[1],elem.position[2]-centre[2]),size=vec(2,2,elem.thickness),opacity=0.3,color=elem.get_colour()))
		return boxes 

	def group(self):
		lis=self.elements
		self.grouped_elements = [[lis[i],lis[i+1]] for i in range(0,len(lis)-1,1)]

	

class Ray(object):
	"All instances of this class are monochromatic rays of light, resembling individual photons"

	def __init__(self,**k):
		try:
			self.name 			= k.pop('name',"ray")
			self.position 		= np.array(k.pop('position'))
			self.angles 		= k.pop('angles')
			self.polarization	= k.pop('polarization')
			self.wavelength		= k.pop('wavelength')
			self.path			= [self.position]
			self.time			= [0]
		except KeyError:
			print('Please provide at least a name, position, angles, polarization and wavelength')

	def showRay(self):
		print("Position: {}, Angles: {}, Polarization: {}, Wavevlength: {}".format(self.position,self.angles,self.polarization,self.wavelength))

	def arrivaltime(self):
		return np.sum(self.time)




class Optic(object):
	"All instances of this class used to build a virtual setup that resembles the SPDC source that do NOT EMIT photons, but only act on them. E.G. non-linear crystals/lenses/HWP etc."
	
	def __repr__(self):
		return "Optic_{}".format(self.name)

	def __init__(self,**k):
		try:
			self.name 		=  k.pop('name');
			self.material	=  k.pop('material',None)
			self.position	=  np.array(k.pop('position'))
			self.thickness 	=  k.pop('thickness',0)
			self.fsurf 		= self.position[2] - 0.5*self.thickness
			self.bsurf 		= self.position[2] + 0.5*self.thickness
		except KeyError:
			print("Please provide at least a name, material, position to initiate an Optics object")

	def calcsurfaces(self):
		self.fsurf = self.position[2] - 0.5*self.thickness
		self.bsurf = self.position[2] + 0.5*self.thickness
	

	def showObject(self):
		print ("Name: {}, Material: {},  Position: {}".format(self.name,self.material,self.position))

	def get_colour(self):
		"Gets colour for visualization as RGB Triplet"
		colour_dict={'BBO':vec(0,0,1),'YVO4':vec(0,1,0),'Air':vec(1,1,1)}
		
		colour=colour_dict.pop(self.material,vec(1,1,1))

		return colour

	def getn(self,ray):
		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")

		lam=ray.wavelength

		return 1+(0.05792105)/(238-(lam*1e-3)**-2)+(0.00167917)/(57.362-(lam*1e-3)**-2)

	def getdn(self,ray):

		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")

		lam=ray.wavelength

		return -(3358.34)/(((57.362-(1e6)/lam**2)**2)*lam**3)-(115842)/(((238-(1e6)/lam**2)**2)*lam**3)

	def translate(self,ray):
		ray.position = (ray.position + np.array([tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]))




class Crystal(Optic):
	"All instances of this class (child of Optics), are objects that are (non-linear) crystals that act on the photons. They are all rectangular boxes that are made of a birefringent material"
	
	materials 	=	["BBO","YVO4"]

	selm_coeff 	= {	"BBO" 		: 	[2.7359, 0.01878, 0.01822, 0.01354, 2.3753, 0.01224, 0.01667, 0.01516],
 					"YVO4" 		: 	[3.77834, 0.069736, 0.04724, 0.0108133, 4.59905, 0.110534, 0.04813, 0.0122676]
 				}

	def __init__(self,**k):
		try:
			self.orientation 	= k.pop('orientation')
			self.cutangle		= k.pop('cutangle')
		except KeyError:
			print("Please provide at least orientation and cutangle to initiate a Crystal object")
		super(Crystal,self).__init__(**k)

	def getn(self,ray):

		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lamb=ray.wavelength

		if ray.polarization == "H":
			if self.orientation in {"left","right"}:
				return (c[0]+(c[1])/((lamb*1e-3)**2-c[2])-c[3]*(lamb*1e-3)**2)**0.5

			else:
				if self.orientation is "up":
					return n_ext_effective(coeff = c,theta=self.cutangle-ray.angles[0],lam=lamb)

				elif self.orientation is "down":
					return n_ext_effective(coeff = c,theta = self.cutangle+ray.angles[0],lam=lamb)

		elif ray.polarization == "V":
			if self.orientation in {"up","down"}:
				return (c[0]+(c[1])/((lamb*1e-3)**2-c[2])-c[3]*(lamb*1e-3)**2)**0.5
			else:
				if self.orientation is "left":
					return n_ext_effective(coeff = c,theta = self.cutangle-ray.angles[1],lam=lamb)

				elif self.orientation is "right":
					return n_ext_effective(coeff = c,theta = self.cutangle+ray.angles[1],lam=lamb)
	
	def getdn(self,ray):

		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
					
		c=self.selm_coeff[self.material]
		lamb=ray.wavelength

		if ray.polarization == "H":
			if self.orientation in {"left","right"}:
				return dSellmeier(coeff=c[0:4],lam=lamb)
				
			else:
				if self.orientation is "up":

					return dn_ext_effective(coeff=c,theta=self.cutangle-ray.angles[0],lam=lamb)

				elif self.orientation is "down":
					
					return dn_ext_effective(coeff=c,theta=self.cutangle+ray.angles[0],lam=lamb)
					
	
		elif ray.polarization == "V":
			if self.orientation in {"up","down"}:
				
				return dSellmeier(coeff=c[0:4],lam=lamb)
				
			else:
				if self.orientation is "left":
					
					return dn_ext_effective(coeff=c,theta=self.cutangle-ray.angles[1],lam=lamb)
					

				elif self.orientation is "right":
					
					return dn_ext_effective(coeff=c,theta=self.cutangle+ray.angles[1],lam=lamb)
					

	def getwalkoff(self,ray):

		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the walkoff for!")

		c=self.selm_coeff[self.material]

			
		if ray.polarization == "H":
			if self.orientation == "up":
				w =  walkoff(theta=self.cutangle-ray.angles[0],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return np.array([w,0,0])
			elif self.orientation == "down":
				w =  -1*walkoff(theta=self.cutangle+ray.angles[0],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return np.array([w,0,0])
			else:
				w = 0
				return np.array([0,w,0])
		elif ray.polarization == "V":
			if self.orientation == "left":
				w =  walkoff(theta=self.cutangle-ray.angles[1],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return np.array([0,w,0])
			elif self.orientation == "right":
				w = -1*walkoff(theta=self.cutangle+ray.angles[1],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return np.array([0,w,0])
			else:
				w = 0
				return np.array([0,w,0])


	def translate(self,ray):
		walkoff=self.getwalkoff(ray)
		ray.position = (ray.position + walkoff + np.array([tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]))


class Lens(Optic):
	"All instances of this class (Child of optics) are objects that are curved surfaces (spherical) that act on the photons. One can have two different Radii of Curvature, so this field should be a vector"
	
	materials 	= ["N-SF6HT","N-LAK22"]

	selm_coeff 	= {	"N-SF6HT" 	:	[1.77931763, 0.0133714182, 0.338149866, 0.0617533621, 2.08734474, 174.01759],
					"N-LAK22" 	:	[1.14229781, 0.00585778594, 0.535138441, 0.0198546147, 1.04088385, 100.834017]
 					}

	def __init__(self,**k):
		super(Lens,self).__init__(**k)
		try:
			self.ROC 		= k.pop('ROC')
			self.centre 	= k.pop('centre')
			self.position 	= self.position+self.centre
		except KeyError:
			print("Please provide at least ROC (Radii of Curvature) and centre position to initiate a Lens object")
	
	def achromat(position=[],centre=[],f=0):
		"Create achromat at position"
		lens1=Lens(name='Lens1',material ="N-SF6HT",orientation=None,position=np.array(position)+np.array([0,0,-2.3/2]),thickness=2.3,ROC=[13.5,-10.6],centre=[centre[0],centre[1],0])
		lens2=Lens(name='Lens2',material ="N-LAK22",orientation=None,position=np.array(position)+np.array([0,0,+1.3/2]),thickness=1.3,ROC=[-10.6,-47.8],centre=[centre[0],centre[1],0])
		airf = Optic(name='Air_f',material="Air",orientation = None, position = np.array(position)+np.array([0,0,1.3+(f-1.3)/2]),thickness=f-1.3)

		return [lens1,lens2,airf]

	def getn(self,ray):
		
		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lam=(1e-3)*ray.wavelength

		return (1+(c[0]*(lam**2))/(lam**2-c[1])+(c[2]*(lam**2))/(lam**2-c[3])+(c[4]*(lam**2))/(lam**2-c[5]))**(0.5)

	def getdn(self,ray):
		
		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lam=ray.wavelength*1e-3
	
		return 1e-6*lam*((-(c[0]*c[1])/(c[1]-lam**2)**2-(c[2]*c[3])/(c[3]-lam**2)**2-(c[4]*c[5])/(c[5]-lam**2)**2)/sqrt(1+(c[0]*lam**2)/(-c[1]+lam**2)+(c[2]*lam**2)/(-c[3]+lam**2)+(c[4]*lam**2)/(-c[5]+lam**2)))

	def translate(self,ray):
		ray.position = (ray.position + np.array([tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]))


class HWP(Optic):
	"All instances of this class (Child of optics) are objects that perform the mathematical operation of H->V and V->H for a certain wavelenth range"

	def __init__(self,**k):
		try:
			self.cutoff=k.pop('cutoff')
		except KeyError:
			print("Please provide the cutoff wavelenght in nm for the HWP")

		super(HWP,self).__init__(**k)

	def show_cutoff(self):
		print("I act on a waveplate for photons with wavelength above {}nm".format(self.cutofff))

	def propagate(self,ray):
		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to act on!")

		if ray.wavelength > self.cutoff:
			if ray.polarization == "H":
				ray.polarization = "V"
			else:
				ray.polarization = "H"

	def translate(self,ray):
		self.propagate(ray)
		

