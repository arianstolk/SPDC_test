#########################################################
### Analysis/Ray-tracing module for Bulk SPDC sources ###
### Written by: Arian Stolk							  ###
### Date: September 2017							  ###
#########################################################

###The goal of this project is to provide a tool that will allow the investigation of SPDC sources made of bulk non-linear crystals.###
###Import library####

from math import *
import random
import scipy
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
	"Takes the wavelength [nm] array lam and coeff and outputs the refractive index. Used for BBO and YVO4"
	
	return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5

def dSellmeier(coeff=[0,0,0,0],lam = 785):
	"Takes the wavelength [nm] array lam and coeff and outputs the derivate of refractive index with respect to wavelength. Used for BBO and YVO4"
	c = coeff
	return ((1e-6)*lam*(-c[3]-(c[1])/(c[2]-(1e-6)*lam**2)**2))/np.sqrt((c[0]-(1e-6)*c[3]*lam**2-(c[1])/(c[2]-(1e-6)*lam**2)))

def v_group(lam=785,n=1,dn=0):
	"Takes the arrays lam, n and dn and outputs the array of group velocities"

	return (2.998e11)/(n-lam*dn)

def n_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
	"takes the full sellmeier coefficients, the angle array theta and the wavelength array wavelength and outputs the refractive index in an array "
	c=coeff
	# try:
	# 	print(lam.shape,theta.shape)
	# except AttributeError:
	# 	pass
	return ((np.sin(theta)/(c[4]+(c[5])/((lam*1e-3)**2-c[6])-(c[7])*(lam*1e-3)**2)**0.5)**2+(np.cos(theta)/(c[0]+(c[1])/((lam*1e-3)**2-c[2])-(c[3])*(lam*1e-3)**2)**0.5)**2)**(-0.5)

def dn_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
	"takes the full sellmeier coefficients, the angle array theta and the wavelength array wavelength and outputs the derivate of the refractive index in an array "
	c=coeff
	return -((1e-6)*lam*(((c[1]+c[3]*(c[2]-(1e-6)*lam**2)**2)*np.cos(theta)**2)/(c[1]+(c[2]-(1e-6)*lam**2)*(-c[0]+(1e-6)*c[3]*lam**2))**2\
			+(((c[5]+c[7]*(c[6]-(1e-6)*lam**2)**2)*np.sin(theta)**2)/(c[5]+(c[6]-(1e-6)*lam**2)*(-c[4]+(1e-6)*c[3]*lam**2))**2))\
			/((np.cos(theta)**2)/(c[0]-(1e-6)*c[3]*lam**2-c[1]/(c[2]-(1e-6)*lam**2))+(np.sin(theta)**2)/(c[4]-(1e-6)*c[7]*lam**2-c[5]/(c[6]-(1e-6)*lam**2)))**(3/2))

def flatten(l):
	"flattens list l to having only 1 dimension"
	for el in l:
		if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
			yield from flatten(el)
		else:
			yield el

def walkoff(theta=0,coeff=[0,0,0,0,0,0,0,0],lam=785,thickness=1):
	"Takes the angle array theta, full Sellmeiercoeff, wavelength array lam [nm] and thickness array [mm] and returns the 1d value of the walkoff (SO NOT AN ARRAY) "

	# if len(coeff)<8:
	# 	print("Please provide all 8 Sellmeier coeffs")
	# else:
	n_ord=Sellmeier(coeff[0:4],lam)
	n_ext=Sellmeier(coeff[4:8],lam)
	return 0.5*thickness*(n_ext_effective(coeff=coeff,theta=theta,lam=lam)**2)*((n_ord**-2)-(n_ext**-2))*np.sin(2*theta)

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
	"Input float of pump wavelength, and arrays size (1,n) of l_list, theta_list, cutangle and crystal_length output provides the array size (1,n) of the phasematching fucntion. "
	c=coeff
	#create lists of pump,signal and idler wavelengths
	lp=lpump
	ls=l_list
	li=ls*lp/(ls-lp);
	#calculate the angular frequency from wavelength
	wp,ws,wi = (2*np.pi)/lp,(2*np.pi)/ls,(2*np.pi)/li

	#calculate the ord/extraord refractive indices for the wavelenghts
	nop = Sellmeier(coeff=c[0:4],lam=lp)
	nep = Sellmeier(coeff=c[4:8],lam=lp)
	nos = Sellmeier(coeff=c[0:4],lam=ls)
	nes = Sellmeier(coeff=c[4:8],lam=ls)
	noi = Sellmeier(coeff=c[0:4],lam=li)
	nei = Sellmeier(coeff=c[4:8],lam=li)

	#calculate the effective pump refractive index
	npeff=np.sqrt(1/((np.cos(cutangle)/nop)**2+(np.sin(cutangle)/nep)**2))
	L=crystal_length
	W=0.1

	thet_s=theta_list 																#opening angle of the signal photon
	thet_i=np.arcsin(nos*ws*np.sin(thet_s)/(noi*wi))								#corresponding opening angle of the idler photon based on refractive indices
	dkz=npeff*wp-nos*ws*np.cos(thet_s)-noi*wi*np.cos(thet_i)						#corresponding phase mismatch in z (propagationdirection) per unit length
	dky=-nos*ws*np.sin(thet_s)+noi*wi*np.sin(thet_i)								#mismatch in pphase in y (direction perp to z) per unit length
	phi=np.exp(-((W*1e6)**2)*(dky**2)/2)*(np.sin(0.5*dkz*L*1e6)/(0.5*dkz*L*1e6))**2	#sinc2() func of the mismatch over the length of the crystal

	return phi #(1,n) array

def dens_hist(data):
   

    #data definition
    xdat, ydat = data[0],data[1]
    
    x_range=np.array([-2,2])+np.mean(xdat)
    y_range=np.array([-2,2])+np.mean(ydat)
    xyrange = [x_range,y_range] # data range
    bins = [100,100] # number of bins
    thresh = 1  #density threshold

    # histogram the data
    hh, locx, locy = scipy.histogram2d(xdat, ydat, range=xyrange, bins=bins)
    posx = np.digitize(xdat, locx)
    posy = np.digitize(ydat, locy)

    #select points within the histogram
    ind = (posx > 0) & (posx <= bins[0]) & (posy > 0) & (posy <= bins[1])
    hhsub = hh[posx[ind] - 1, posy[ind] - 1] # values of the histogram where the points are
    xdat1 = xdat[ind][hhsub < thresh] # low density points
    ydat1 = ydat[ind][hhsub < thresh]
    hh[hh < thresh] = np.nan # fill the areas with low density by NaNs

    plt.imshow(np.flipud(hh.T),cmap='jet',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper')
    plt.title("xmean i {},xstd is {} and ymean is{}, ystd is {}".format(np.mean(xdat),np.std(xdat),np.mean(ydat),np.std(ydat)))
    plt.colorbar()   
    plt.show()
    

##################################################################################################################
class Simulation(object):
	"This class describes the interaction of Ray and ExpSetup, where the Rays are traced through the setup using the methods described in this class"

	def __init__(self,**k):
		try:
			self.setup=k.pop('setup')					
			self.rays=k.pop('rays',None)			#rays are always given in a list of length 2n, where n the number of crystals in which downconversion happens [[signal_rays1],[idler_rays1],[signal_rays2],[idler_rays2],..] etc 


			if not type(self.rays[0])==list:		#Check if it is given a list of lists, otherwise maks it a list of the list its given
				self.rays=[self.rays]

			self.numobj=self.setup.nr_elements      #Define the number of objects (without air) that will be traced
			self.num_interfaces=len(self.setup.grouped_elements) #Number of surfaces in the setup

			self.store_path = k.pop("store_path",False)	#Boolean that sets if the full path is being stored
			self.store_time = k.pop("store_time",False)	#Boolean that sets if the time is being stored

		except KeyError:
			print("Please provide rays and setup to initiate the simulation")

	def __repr__(self):
		return "Simulation of a setup with {} elements".format(self.numobj)

	def run(self,Ntot=100,store=False,multithread=False,gen_SPDC=False,pumpray=[],pump_waist=[0.04,0.04],pump_focus=[8,8],l_min=480,theta_min=-2,theta_max=2,cutangle=28.76*np.pi/180):
		"Most important method of the class. This will trace all the rays in self.start_list through the self.setup. If self.start_list is not present it will either get it from self.rays, or it will generate it using self.get_SPDC_rayset_adv. It will then store the result into self.complete_results"

		self.complete_results = []
		
		if not hasattr(self,'start_list'):
			self.start_list = []
			if not gen_SPDC:
				for list_of_rays in self.rays: #loop through all the signal and idler sets
					if not multithread:

						#print("Running simulation for {} rays throug {} optical elements. Storing result is: {}".format(len(list_of_rays),self.numobj,store))
						#Set initial values for this simulation. This is slow and an old way of importing the position,angle,wavelenght,time and polarization
						self.angle_list 	= np.array([x.angles for x in list_of_rays])
						self.lam_list 		= np.array([[x.wavelength] for x in list_of_rays])
						self.position_list 	= np.array([x.position for x in list_of_rays])
						self.time_list		= np.zeros((len(list_of_rays),1))
						self.polarization 	= list_of_rays[0].polarization
						#print(self.angle_list.shape,self.lam_list.shape,self.position_list.shape,self.time_list.shape)

						self.start_list.append([self.position_list,self.angle_list,self.lam_list,self.time_list,self.polarization]) # this start list is the shape of [(N,3,1),(N,2,1),(N,1,1),(N,1,1),(N,1,1)]

						self.complete_ray_propagation() #run the raytracing
			
			else:
				#Generate the SPDC rays using the get_SPDC_rayset_adv, which outputs the list of [[signals1],[idlers1],[signals2],[idlers2],...] etc
				self.rays = self.get_SPDC_rayset_adv(Ntot=Ntot,nr_crystals=2,pumpray=Ray(position=[0,0,0],angles=[0,0],wavelength=405,polarization="H"),pump_waist=pump_waist,pump_focus=pump_focus,l_min=l_min,theta_min=theta_min,theta_max=theta_max,cutangle=cutangle)
				
				for list_of_rays in self.rays: #loop through all the signal and idler sets
					if not multithread:

						#Set initial values for this simulation
						self.angle_list 	= list_of_rays[1]
						self.lam_list 		= np.reshape(list_of_rays[2],(len(list_of_rays[2]),1))
						self.position_list 	= list_of_rays[0]
						self.time_list		= np.zeros((len(list_of_rays[0]),1))
						self.polarization 	= list_of_rays[3]
						# print(self.position_list.shape,self.angle_list.shape,self.lam_list.shape,self.time_list.shape)
						self.start_list.append([self.position_list,self.angle_list,self.lam_list,self.time_list,self.polarization]) # this start list is the shape of [(N,3,1),(N,2,1),(N,1,1),(N,1,1),(N,1,1)]

						self.complete_ray_propagation() #run the raytracing
		else:
			# thestart list is the shape of [(N,3,1),(N,2,1),(N,1,1),(N,1,1),(N,1,1)], and because it already exists we can directly do the simulations
			for list_of_rays in self.start_list: #loop through all the signal and idler sets
				if not multithread:

					
					#Set initial values for this simulation
					self.angle_list 	= list_of_rays[1]
					self.lam_list 		= np.reshape(list_of_rays[2],(len(list_of_rays[2]),1))
					self.position_list 	= list_of_rays[0]
					self.time_list		= np.reshape(list_of_rays[3],(len(list_of_rays[3]),1))
					self.polarization 	= list_of_rays[4]
					# print(self.angle_list.shape,self.lam_list.shape,self.position_list.shape,self.time_list.shape)
					self.complete_ray_propagation() #run the raytracing

	def complete_ray_propagation(self):
		"This method traces the rays defined in self.run through the self.setup. It will save the times and positions of the rays through all the surfaces of the setup into its attribute self.complete_results."
		grouped_elements = self.setup.grouped_elements
		
		#Initialize the result arrays

		self.angle_list_results		=np.zeros((self.angle_list.shape[0],self.angle_list.shape[1],1+len(grouped_elements)))
		self.position_list_results  =np.zeros((self.position_list.shape[0],self.position_list.shape[1],1+len(grouped_elements)))
		self.time_list_results		=np.zeros((self.time_list.shape[0],self.time_list.shape[1],1+len(grouped_elements)))
		
		#Entering the initial values
		
		self.angle_list_results[0:self.angle_list.shape[0],0:self.angle_list.shape[1],0] 			 	= self.angle_list		
		self.position_list_results[0:self.position_list.shape[0],0:self.position_list.shape[1],0] 		= self.position_list
		self.time_list_results[0:self.time_list.shape[0],0:self.time_list.shape[1],0]					= self.time_list
		
		#wavelength is constant
		lam		= self.lam_list

		
		if debug:
			print("I am going to trace {} through {} ".format(ray.showRay(),grouped_elements))
			

		if self.store_path == True and self.store_time == True:

			for i,elements in enumerate(grouped_elements):
				
				elements[0].ray_pol,elements[1].ray_pol=self.polarization[0],self.polarization[0]

				cur_pos		=self.position_list_results[:,:,i]
				cur_time	=self.time_list_results[:,:,i]
				cur_angles	=self.angle_list_results[:,:,i]
				
				#call the two primitives that do the calculations for raytracing 
				new_pos=Simulation.translate(cur_pos,cur_angles,lam,elements[0])
				new_angle=Simulation.refract(cur_pos,cur_angles,lam,elements)

				#calculate the propagation time during this step
				new_time=np.linalg.norm(new_pos-cur_pos,axis=1,keepdims=True)/v_group(lam=lam,n=elements[0].getn(cur_angles,lam),dn=elements[0].getdn(cur_angles,lam))

				self.angle_list_results[:,:,i+1] 	= new_angle
				self.position_list_results[:,:,i+1]	= new_pos
				self.time_list_results[:,:,i+1]		= new_time

				self.polarization = elements[0].ray_pol

		self.complete_results.append([self.position_list_results[:,:,-1],self.angle_list_results[:,:,-1],self.lam_list,np.sum(self.time_list_results,axis=(1,2)),np.full((len(self.lam_list),1),self.polarization,dtype=str)])


	def refract(pos,angle,lam,optic_elements=[]):
		"Method to transform the angles of the ray at an interface. If it encounters a lens, the thin lens formula is used, otherwise snells law is applied"

		if not pos[0][2]>optic_elements[1].bsurf: #check if ray is not already past this surface due to starting position
			#get optic elements from list
			element1=optic_elements[0]
			element2=optic_elements[1]

			#make (N,2) array of the refractive indices for before and after the surface
			nin=np.repeat(element1.getn(angle,lam),2,axis=1)
			nout=np.repeat(element2.getn(angle,lam),2,axis=1)

			#calculate the fraction needed for snells law and thin lens equation
			indexfract = (nin/nout)		

			# print(nin,nout,ray.angles,ray.position,element1.position,element2.position)

			if hasattr(element1,"ROC"): #check if first element is lens, if yes use back surface ROC of element1
				position_on_lens=(pos-element1.position)[:,0:2]
				angles =  np.arcsin(np.sin(angle)*(indexfract))+position_on_lens*((nin-nout)/(nout*element1.ROC[1]))
			elif hasattr(element2,"ROC"): #check if second element is lens, if yes use front surface ROC of element2
				position_on_lens=(pos-element2.position)[:,0:2]
				angles =  np.arcsin(np.sin(angle)*(indexfract))+position_on_lens*((nin-nout)/(nout*element2.ROC[0]))
			else:#if not lenses, use snells law
				angles = np.arcsin(np.sin(angle)*(indexfract))

			return angles
			
		else:
			return angle

		

	def translate(pos,angle,lam,optic_element):
		"method for translating the rays throught the objects"

		if not pos[0][2]>optic_element.bsurf:  #check if ray is not already past this surface due to starting position
			return optic_element.translate(pos,angle,lam)
		else:
			return pos
		
		if debug:
			print("I traced {} in {} to {}".format(ray,optic_element,ray.position))


	
		

	def get_SPDC_rayset_adv(self,Ntot=1,nr_crystals=1,pumpray=[],pump_waist=[0,0],pump_focus=[0,0],cutangle=28.8*np.pi/180,l_min=0,l_max=0,theta_min=0,theta_max=0,factor=1e-2):
		"This is a complex method that generates the starting positions/angles/wavelenght/polarization according to the Type-1 phasematching conditions in BBO"
		#checks if the amount of trials is smaller than the maximum per loop: 10M.
		if Ntot<10000000:
			N=Ntot
		else:
			N=10000000

		Number_of_cycles = int(ceil(Ntot/N)) #Number of cyles of 10M

		final_list = []						#Initialization of list to store the cycles

		lpump=pumpray.wavelength 			#setting the wavelength and waists of the pump
		w0x=pump_waist[0]
		w0y=pump_waist[1]

		zR_pump_x=(np.pi*w0x**2)/(lpump*1e-6)#calculating the rayleigh ranges in x and y
		zR_pump_y=(np.pi*w0y**2)/(lpump*1e-6)
		

		for i in range(Number_of_cycles):

			#generating large arrays of random numbers

			rand_u = np.random.uniform(0,1,N)								#used for calculating position along walkoff path of ray
			
			randN_l		= np.random.uniform(l_min,2*lpump,N)				#generating random wavelengths for singal_min to 2*pump wavelength (phasefunction is symmetric in 2*pump wavelength)
			randN_theta	= np.random.uniform(theta_min,theta_max,N)*np.pi/180#random opening angles 
			rand_check	= np.random.uniform(0,1,N)							#the random numbers used in the check for the rejection sampling (could these be reused?)

			return_list=[]

			pumpray.position=np.array([0,0,0])								#For all the cycles the pump starts at 0,0,0

			for i in range(nr_crystals):#looping through all the crystals where downconversion happens
				
				crystal=self.setup.crystals[i]	#												
				pumpray.position = np.array([pumpray.position[0],pumpray.position[1],crystal.fsurf]) #downconversion begins at frontsurface
				w_beg=pumpray.position

				w_end=w_beg+crystal.getwalkoff_ray(pumpray)+np.array([0,0,crystal.thickness])		#downconversion ends at begin + walkoff + thickness
				start_pos = np.repeat(np.array([w_beg]),N,axis=0) + np.outer(rand_u,(w_end-w_beg))	#the random starting pos is just a linear combination of these

				
				start_z = (start_pos.T)[2]															#the starting positions along the z axis

				#"Introduce effects of gaussian pump"
				
				wzx,wzy = w0x*np.sqrt(1+((start_z-pump_focus[0])/zR_pump_x)**2),w0y*np.sqrt(1+((start_z-pump_focus[1])/zR_pump_y)**2)	#for all z, there is a beam waist

				sx,sy=wzx,wzy			#this beam waist is the standard deviation of the gaussian used for further sampling
			
				gauss_x=np.random.normal(np.zeros((1,N)),sx)	#random x position due to gaussian pump
				gauss_y=np.random.normal(np.zeros((1,N)),sy)	#random y position due to gaussian pump

				gauss_xy=np.concatenate((gauss_x,gauss_y)).T

				zpos_rel_wx=np.where(np.abs(-pump_focus[0]+start_z)>1e-9,(-pump_focus[0]+start_z),1e-9)	#we want the z pos with respect to focus. np.where is used to remove all values smaller than 1e-9 to avoid dividing by 0
				zpos_rel_wy=np.where(np.abs(-pump_focus[1]+start_z)>1e-9,(-pump_focus[1]+start_z),1e-9)

				start_angles_gauss=np.concatenate((-np.arcsin(gauss_x/(zpos_rel_wx+((zR_pump_x)**2)/(zpos_rel_wx))),-np.arcsin(gauss_y/(zpos_rel_wy+((zR_pump_y)**2)/(zpos_rel_wy))))).T #for ever (x,y) we can calculate propagation angles (a,b) using ROC(z) of the gaussian beam

				#"Sampling the phasefunction for the N randomly generated angles. the weight will later be used to accept or reject samples. Factor is number <1, the smaller this number the smoother the distribution, but the more trys needed"
				#TODO: need to choose which angle (a,b) it should add to the phasematching (currently always chooses a but it depends on crystal orientation)
				weight=factor*phasefunction_vec(l_list=randN_l,theta_list=randN_theta,lpump=lpump,cutangle=cutangle+(start_angles_gauss.T)[0])

				#reshape the gaussian coordinates x,y to the format [x,y,0] 
				gauss_pad=np.zeros(start_pos.shape)
				gauss_pad[:gauss_xy.shape[0],:gauss_xy.shape[1]]=gauss_xy

				#put all the parameters so far (position, angles, wavelenght, opening_angle) in an array with length 100.000.0000 so we can select all the succesfull ones at once
				sampled_params=np.concatenate((start_pos+gauss_pad,start_angles_gauss,np.array([randN_l]).T,np.array([randN_theta]).T),axis=1)#list of important simulation parameters

				final_params=sampled_params[np.where(weight>rand_check)]#filters the sampled_params for the succesfully drawn samples
				Nsuc=len(final_params)

				#extract the (position, angles, wavelenght, opening_angle) from the succesfull ones 
				[ray_start,ray_angle,ls,opening_angle]=[final_params[:,0:3].T,final_params[:,3:5].T,final_params[:,5:6].T,final_params[:,6:7].T]

				#get idler wavelengths
				li = ls*(lpump)/(ls-lpump)
				
				#calculate actual propagation angles of signal and idler (rotationally symmetric in space due to ordidnary polarization)
				rand_nr = np.random.uniform(0,1,len(final_params))
				azim=np.cos(rand_nr*np.pi)*opening_angle
				horiz=np.sign(azim)*np.sign(opening_angle)*np.sqrt(opening_angle**2 - azim**2)

				prop_angles=np.concatenate((azim,horiz)).T
				

				#Assembling the final list of parameters, combining all the angular and spatial effects				

				#TODO:Need to automatically decide on polarization, depends on wavematching/crystal orientation
				Srays=[ray_start.T,(ray_angle.T+prop_angles),ls.T,np.full((Nsuc,1),'V',dtype=str)]
				Irays=[ray_start.T,-(ray_angle.T+prop_angles),li.T,np.full((Nsuc,1),'V',dtype=str)]
				
				return_list.append(Srays)
				return_list.append(Irays)

				pumpray.position += crystal.getwalkoff_ray(pumpray) #update pumpray position for next crystal

			final_list.append(return_list) #update all the cycles
		
		
		#Create empty list the size of the list we want to return
		the_result_list= [[[] for i in range(4)] for j in range(4)]
		
		#loop through all the elements of the empty list, filling them with the result of the individual cycles 
		#THIS IS SLOW AND UGLY BUT I CANT FIND A BETTER WAY, maybe np.concatenate(final_list,axis=?)	
		for i in range(4):
			for j in range(4):
				the_result_list[i][j]=np.concatenate(tuple([final_list[k][i][j] for k in range(Number_of_cycles)]))
		
		return the_result_list


	def filter_results(self,fibre_pos=np.array([0,0,0]),core_diam=0,Num_Ap=0):
		"Filter function giving singles and coincidences that enter the fibre at fibre_pos with specified dimensions"

		filter_cond_match_list=[]
		self.singles=[]
		self.coincidences=[]

		for ray_set in self.complete_results:
			pos = ray_set[0]
			angle = ray_set[1]

			filter_cond_match=self.filter_func(pos,angle,fibre_pos=fibre_pos,core_diam=core_diam, Num_Ap = Num_Ap)
			filter_cond_match_list.append(filter_cond_match)

			singles_list =  [ray[np.where(filter_cond_match)] for ray in ray_set]
			self.singles.append(singles_list)

		self.coincidences =[[x[np.where(np.logical_and(filter_cond_match_list[2*round(i/2.1)],filter_cond_match_list[2*round(i/2.1)+1]))] for x in ray_set] for i,ray_set in enumerate(self.complete_results)] #2*round(i/2.1) does (0,1,2,3) -> (0,0,2,2) to compare signal and idler


	def filter_func(self,pos,angle,fibre_pos=np.array([0,0,0]),core_diam=0,Num_Ap=0):

		"method called by filter_results to check if the rays fall in fibre pos with angle < NA"
		core_dist_check=np.linalg.norm(pos-fibre_pos,axis=1)<core_diam/2
		inc_angle_check=np.linalg.norm(angle,axis=1)<Num_Ap

		return np.logical_and(core_dist_check,inc_angle_check)
	
	def show_filtered(self,time_diff=False,bins=50):
		"this function shows the filtered results in a histogram"

		if not time_diff:
			timelist=[1e15*raylist[3] for raylist in self.coincidences]
		else:
			arrivtimes =[(1e15)*raylist[3] for raylist in self.coincidences]
			timelist = [arrivtimes[0]-arrivtimes[1],arrivtimes[2]-arrivtimes[3]]
		 
		plt.figure()

		
		for times in timelist:
			plt.hist(times,bins,alpha=0.5)
			print(np.mean(times))
		
		plt.xlabel("arrival time with respect to earliest photon [fs]")
		plt.ylabel("Occurance")
		plt.show()


	
	def showtimes(self,time_diff=False,bins=50):
		"this function shows the temporal results of the raytracing"
		if not time_diff:

			timelist=(1e15)*np.array([x[3][:] for x in self.complete_results])

		else:
			arrivtimes = (1e15)*np.array([x[3][:] for x in self.complete_results])

			timelist = np.array([arrivtimes[0]-arrivtimes[1],arrivtimes[2]-arrivtimes[3]])
		
		plt.figure()
		
		for times in timelist:
			plt.hist(times,bins,alpha=0.5)
			print(np.mean(times),len(times))
		
		plt.xlabel("arrival time with respect to earliest photon [fs]")
		plt.ylabel("Occurance")
		plt.show()

	def showpos(self):
		"this function shows the position results of the raytracing"
		data_list = [x[0][:].T[0:2] for x in self.complete_results]

		for data in data_list:
			dens_hist(data)

	def calcfocus(self,factor=1,centre=[0,0,0]):
		"this function returns the 'focus', given as the average of the standard deviation of the spatial distribution"

		xmeans=list(map(np.mean,[x[0][:].T[0] for x in self.complete_results]))
		ymeans=list(map(np.mean,[x[0][:].T[1] for x in self.complete_results]))

		xmean = sum((xmeans-centre[0])**2)**0.5
		ymean = sum((ymeans-centre[1])**2)**0.5

		xfoc = np.array(list(map(np.std,[x[0][:].T[0] for x in self.complete_results])))
		yfoc = np.array(list(map(np.std,[x[0][:].T[1] for x in self.complete_results])))

		xfocdiff = abs(xfoc[0] + xfoc[1] - xfoc[2] - xfoc[3])
		yfocdiff = abs(yfoc[0] + yfoc[1] - yfoc[2] - yfoc[3])

		return (xfocdiff + yfocdiff) + factor*(xmean + ymean)

	def get_focus_pos(self):
		xmeans=np.array(list(map(np.mean,[x[0][:].T[0] for x in self.complete_results])))
		ymeans=np.array(list(map(np.mean,[x[0][:].T[1] for x in self.complete_results])))
		return np.mean(np.array([xmeans,ymeans]),axis=1)


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

	def set_pol_flag(self,pol):
		self.ray_pol = pol
	

	def showObject(self):
		print ("Name: {}, Material: {},  Position: {}".format(self.name,self.material,self.position))

	def get_colour(self):
		"Gets colour for visualization as RGB Triplet"
		colour_dict={'BBO':vec(0,0,1),'YVO4':vec(0,1,0),'Air':vec(1,1,1)}
		
		colour=colour_dict.pop(self.material,vec(1,1,1))

		return colour

	def getn(self,angle,lam):

		lamsq=(lam*1e-3)**-2

		return 1+(0.05792105)/(238-lamsq)+(0.00167917)/(57.362-lamsq)

	def getdn(self,angle,lam):

		return -(3358.34)/(((57.362-(1e6)/lam**2)**2)*lam**3)-(115842)/(((238-(1e6)/lam**2)**2)*lam**3)

	def translate(self,pos,angle,lam):
		return (pos + np.array([np.tan(angle[:,0])*(self.thickness-(pos[:,2]-self.fsurf)),np.tan(angle[:,1])*(self.thickness-(pos[:,2]-self.fsurf)),(self.thickness-(pos[:,2]-self.fsurf))]).T)


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

	def getn(self,angle,lam):

				
		c=self.selm_coeff[self.material]
		lamb=lam

		if self.ray_pol == "H":
			if self.orientation in {"left","right"}:
				return (c[0]+(c[1])/((lamb*1e-3)**2-c[2])-c[3]*(lamb*1e-3)**2)**0.5

			else:
				if self.orientation is "up":
					return n_ext_effective(coeff = c,theta = angle[:,[0]] - self.cutangle,lam=lamb)

				elif self.orientation is "down":
					return n_ext_effective(coeff = c,theta = angle[:,[0]] + self.cutangle,lam=lamb)

		elif self.ray_pol == "V":
			if self.orientation in {"up","down"}:
				return (c[0]+(c[1])/((lamb*1e-3)**2-c[2])-c[3]*(lamb*1e-3)**2)**0.5
			else:
				if self.orientation is "left":
					return n_ext_effective(coeff = c,theta = angle[:,[1]] - self.cutangle,lam=lamb)

				elif self.orientation is "right":
					return n_ext_effective(coeff = c,theta = angle[:,[1]] + self.cutangle,lam=lamb)
	
	
	def getdn(self,angle,lam):
					
		c=self.selm_coeff[self.material]
		lamb=lam

		if self.ray_pol == "H":
			if self.orientation in {"left","right"}:
				return dSellmeier(coeff=c[0:4],lam=lamb)
				
			else:
				if self.orientation is "up":
					return dn_ext_effective(coeff=c,theta=self.cutangle-angle[:,[0]],lam=lamb)

				elif self.orientation is "down":					
					return dn_ext_effective(coeff=c,theta=self.cutangle+angle[:,[0]],lam=lamb)
					
	
		elif self.ray_pol == "V":
			if self.orientation in {"up","down"}:				
				return dSellmeier(coeff=c[0:4],lam=lamb)
				
			else:
				if self.orientation is "left":					
					return dn_ext_effective(coeff=c,theta=self.cutangle-angle[:,[1]],lam=lamb)
					

				elif self.orientation is "right":					
					return dn_ext_effective(coeff=c,theta=self.cutangle+angle[:,[1]],lam=lamb)
					

	def getwalkoff(self,pos,angle,lam):

		
		c=self.selm_coeff[self.material]
		N=len(lam)
		w_ret=np.zeros((N,3))
			
		if self.ray_pol == "H":
			if self.orientation == "up":
				w =  walkoff(theta=self.cutangle-angle[:,[0]],coeff=c,lam=lam,thickness=self.thickness-(pos[:,[2]]-self.fsurf))
				w_ret[:,[0]] = w
				return w_ret
			elif self.orientation == "down":
				w =  -1*walkoff(theta=self.cutangle+angle[:,[0]],coeff=c,lam=lam,thickness=self.thickness-(pos[:,[2]]-self.fsurf))
				w_ret[:,[0]] = w
				return w_ret
			else:
				return w_ret
		elif self.ray_pol == "V":
			if self.orientation == "left":
				w =  walkoff(theta=self.cutangle-angle[:,[1]],coeff=c,lam=lam,thickness=self.thickness-(pos[:,[2]]-self.fsurf))
				w_ret[:,[1]] = w
				return w_ret
			elif self.orientation == "right":
				w = -1*walkoff(theta=self.cutangle+angle[:,[1]],coeff=c,lam=lam,thickness=self.thickness-(pos[:,[2]]-self.fsurf))
				w_ret[:,[1]] = w
				return w_ret
			else:
				return w_ret

	def getwalkoff_ray(self,ray):

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


	def translate(self,pos,angle,lam):
		#returns the updated position array
		walkoff=self.getwalkoff(pos,angle,lam)
	
		return (walkoff + pos + np.array([np.tan(angle[:,0])*(self.thickness-(pos[:,2]-self.fsurf)),np.tan(angle[:,1])*(self.thickness-(pos[:,2]-self.fsurf)),(self.thickness-(pos[:,2]-self.fsurf))]).T)
	

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

	def getn(self,angle,lam):
		
		c=self.selm_coeff[self.material]
		lamb=(1e-3)*lam

		return (1+(c[0]*(lamb**2))/(lamb**2-c[1])+(c[2]*(lamb**2))/(lamb**2-c[3])+(c[4]*(lamb**2))/(lamb**2-c[5]))**(0.5)

	def getdn(self,angle,lam):

		c=self.selm_coeff[self.material]
		lamb=lam*1e-3
	
		return 1e-6*lamb*((-(c[0]*c[1])/(c[1]-lamb**2)**2-(c[2]*c[3])/(c[3]-lamb**2)**2-(c[4]*c[5])/(c[5]-lamb**2)**2)/np.sqrt(1+(c[0]*lamb**2)/(-c[1]+lamb**2)+(c[2]*lamb**2)/(-c[3]+lamb**2)+(c[4]*lamb**2)/(-c[5]+lamb**2)))

	def translate(self,pos,angle,lam):
		
		return (pos + np.array([np.tan(angle[:,0])*(self.thickness-(pos[:,2]-self.fsurf)),np.tan(angle[:,1])*(self.thickness-(pos[:,2]-self.fsurf)),(self.thickness-(pos[:,2]-self.fsurf))]).T)


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

	def propagate(self,pos,angle,lam):

		if np.all(lam > self.cutoff):
			if self.ray_pol == "H":
				self.ray_pol = "V"
			else:
				self.ray_pol = "H"

	def translate(self,pos,angle,lam):
		self.propagate(pos,angle,lam)

		return pos
		

