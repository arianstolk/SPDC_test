#########################################################
### Analysis/Ray-tracing module for Bulk SPDC sources ###
### Written by: Arian Stolk							  ###
### Date: September 2017							  ###
#########################################################

###The goal of this project is to provide a tool that will allow the investigation of SPDC sources made of bulk non-linear crystals.###
###Import library###

import numpy as np
import sympy as sp
import matplotlib
from vpython import *

debug=True

################################################################################################################
def Sellmeier(coeff=[0,0,0,0],lam = 785):
	
	return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5

def n_ext_effective(coeff=[0,0,0,0,0,0,0,0],theta=0,lam=785):
	c=coeff
	return ((np.sin(theta)/(c[4]+(c[5])/((lam*1e-3)**2-c[6])-(c[7])*(lam*1e-3)**2)**0.5)**2+(np.cos(theta)/(c[0]+(c[1])/((lam*1e-3)**2-c[2])-(c[3])*(lam*1e-3)**2)**0.5)**2)**(-0.5)

def walkoff(theta=0,coeff=[0,0,0,0,0,0,0,0],lam=785,thickness=1):

	if len(coeff)<8:
		print("Please provide all 8 Sellmeier coeffs")
	else:
		n_ord=Sellmeier(coeff[0:4],lam)
		n_ext=Sellmeier(coeff[4:8],lam)
		return 0.5*thickness*(n_ext_effective(coeff=coeff,theta=theta,lam=lam)**2)*((n_ord**-2)-(n_ext**-2))*np.sin(2*theta)
##################################################################################################################
class Simulation(object):
	"This class describes the interaction of Ray and ExpSetup, where the Rays are traced through the setup using the methods described in this class"

	def __init__(self,debug=debug,**k):
		try:
			self.setup=k.pop('setup')
			self.rays=k.pop('rays')
			self.numrays=len(self.rays)
			self.numobj=self.setup.nr_elements
			self.num_interfaces=len(self.setup.grouped_elements)
		except KeyError:
			print("Please provide rays and setup to start the simulation")

	def refract(ray,optic_elements=[]):
		element1=optic_elements[0]
		element2=optic_elements[1]
		nin=element1.getn(ray)
		nout=element2.getn(ray)

		if not nin == nout:
			if hassatr(element1,ROC):
				pos=norm(ray.position[0:2]-element1.position[0:2])
				ray.angles =  [np.arcsin(np.sin(x)(nin/nout))+pos*(nin-nout)/(nout*element1.ROC[1]) for x in ray.angles]
			elif hassattr(element2,ROC):
				pos=norm(ray.position[0:2]-element2.position[0:2])
				ray.angles =  [np.arcsin(np.sin(x)(nin/nout))+pos*(nin-nout)/(nout*element1.ROC[0]) for x in ray.angles]
			else:
				ray.angles = [p.arcsin(np.sin(x)(nin/nout)) for x in ray.angles]

	def translate(ray,optic_element=[]):
		optic_element.translate(ray)


class ExpSetup(object):
	"All instances of this class are a set of Optic elements along the z-direction. This class will handle all the preparation of the Optic elements to make them ready for the sequential ray tracing"

	def __init__(self,*args):
		if len(args)==0:
			raise Exception("Please provide the optical elements for the setup")
		self.elements = list(args)
		self.nr_elements = len(self.elements)

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
				self.elements.insert(0,Optic(name="Air_Begin",material="Air",position=[0,0,(self.elements[0].fsurf)/2],thickness=self.elements[0].fsurf))
			self.elements.insert(-1,Optic(name="Air_End",material="Air",position=[0,0,self.elements[0].bsurf+2.5],thickness=5))

		else:
			for i in range(self.nr_elements-1,0,-1):
				gap =  self.elements[i].fsurf - self.elements[i-1].bsurf
				if gap > 0:
					self.elements.insert(i,Optic(name="Air_after_{}".format(self.elements[i].name),material="Air",position=[0,0,self.elements[i-1].bsurf+0.5*(self.elements[i].fsurf-self.elements[i-1].bsurf)],thickness=gap))
			if self.elements[0].fsurf > 0:
				self.elements.insert(0,Optic(name="Air_Begin",material="Air",position=[0,0,(self.elements[0].fsurf)/2],thickness=self.elements[0].fsurf))
			self.elements.append(Optic(name="Air_End",material="Air",position=[0,0,self.elements[-1].bsurf+2.5],thickness=5))
	
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
			self.position 		= k.pop('position')
			self.angles 		= k.pop('angles')
			self.polarization	= k.pop('polarization')
			self.wavelength		= k.pop('wavelength')
		except KeyError:
			print('Please provide at least a name, position, angles, polarization and wavelength')

	def showRay(self):
		print("Position: {}, Angles: {}, Polarization: {}, Wavevlength: {}".format(self.position,self.angles,self.polarization,self.wavelength))



class Optic(object):
	"All instances of this class used to build a virtual setup that resembles the SPDC source that do NOT EMIT photons, but only act on them. E.G. non-linear crystals/lenses/HWP etc."
	
	def __repr__(self):
		return "Optic_{}".format(self.name)

	def __init__(self,**k):
		try:
			self.name 		=  k.pop('name');
			self.material	=  k.pop('material',None)
			self.position	=  k.pop('position')
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
		ray.position+= [np.tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),np.tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]




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
				x=sp.Symbol('x')
				ydiff=sp.diff((c[0]+(c[1])/((x*1e-3)**2-c[2])-c[3]*(x*1e-3)**2)**0.5,x)
				f = sp.lambdbify(x,ydiff,'numpy')
				return f(lamb)
				
			else:
				if self.orientation is "up":
					x=sp.Symbol('x')
					ydiff=sp.diff(n_ext_effective(coeff=c,theta=self.cutangle-ray.angles[0],lam=x))
					f = sp.lambdify(x,ydiff,'numpy')
					return f(lamb)

				elif self.oribentation is "down":
					x=sp.Symbol('x')
					ydiff=sp.diff(n_ext_effective(coeff=c,theta=self.cutangle+ray.angles[0],lam=x))
					f = sp.lambdify(x,ydiff,'numpy')
					return f(lamb)
	
		elif ray.polarization == "V":
			if self.orientation in {"up","down"}:
				x=sp.Symbol('x')
				ydiff=sp.diff((c[0]+(c[1])/((x*1e-3)**2-c[2])-c[3]*(x*1e-3)**2)**0.5,x)
				f = sp.lambdify(x,ydiff,'numpy')
				return f(lamb)
				
			else:
				if self.orientation is "left":
					x=sp.Symbol('x')
					ydiff=sp.diff(n_ext_effective(coeff=c,theta=self.cutangle-ray.angles[1],lam=x))
					f = sp.lambdify(x,ydiff,'numpy')
					return f(lamb)

				elif self.orientation is "right":
					x=sp.Symbol('x')
					ydiff=sp.diff(n_ext_effective(coeff=c,theta=self.cutangle+ray.angles[1],lam=x))
					f = sp.lambdify(x,ydiff,'numpy')
					return f(lamb)

	def getwalkoff(self,ray):

		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the walkoff for!")

		c=self.selm_coeff[self.material]

		if ray.polarization == "H":
			if self.orientation == "up":
				w =  walkoff(theta=self.cutangle-ray.angles[0],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return [w,0,0]
			elif self.orientation == "down":
				w =  -1*walkoff(theta=self.cutangle+ray.angles[0],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return [w,0,0]
			else:
				w = 0
				return [w,0,0]
		elif ray.polarization == "V":
			if self.orientation == "left":
				w =  walkoff(theta=self.cutangle-ray.angles[1],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return [0,w,0]
			elif self.orientation == "right":
				w = -1*walkoff(theta=self.cutangle+ray.angles[1],coeff=c,lam=ray.wavelength,thickness=self.thickness-(ray.position[2]-self.fsurf))
				return [0,w,0]
			else:
				w = 0
				return [w,0,0]

	def translate(self,ray):
		walkoff=self.gewalkoff(ray)
		ray.position+= walkoff + [np.tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),np.tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]



class Lens(Optic):
	"All instances of this class (Child of optics) are objects that are curved surfaces (spherical) that act on the photons. One can have two different Radii of Curvature, so this field should be a vector"
	
	materials 	= ["N-SF6HT","N-LAK22"]

	selm_coeff 	= {	"N-SF6HT" 	:	[1.77931763, 0.0133714182, 0.338149866, 0.0617533621, 2.08734474, 174.01759],
					"N-LAK22" 	:	[1.14229781, 0.00585778594, 0.535138441, 0.0198546147, 1.04088385, 100.834017]
 					}

	def __init__(self,**k):
		try:
			self.ROC 	= k.pop('ROC')
			self.centre = k.pop('centre')
		except KeyError:
			print("Please provide at least ROC (Radii of Curvature) and centre position to initiate a Lens object")
		super(Lens,self).__init__(**k)

	def getn(self,ray):
		
		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lam=ray.wavelength

		return (1+(c[0]*(lam*1e-3)**2)/(c[1]-(lam*1e-3)**2)+(c[2]*(lam*1e-3)**2)/(c[3]-(lam*1e-3)**2)+(c[4]*(lam*1e-3)**2)/(c[5]-(lam*1e-3)**2))**(0.5)

	def getdn(self,ray):
		
		if not isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lam=ray.wavelength
		x=sp.Symbol('x')
		ydiff=sp.diff((1+(c[0]*(x*1e-3)**2)/(c[1]-(x*1e-3)**2)+(c[2]*(x*1e-3)**2)/(c[3]-(x*1e-3)**2)+(c[4]*(x*1e-3)**2)/(c[5]-(x*1e-3)**2))**(0.5),x)
		f = sp.lambdify(x,ydiff,'numpy')
		return f(lam)

	def translate(self,ray):
		ray.position+= [np.tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),np.tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]


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
		ray.position+= [np.tan(ray.angles[0])*(self.thickness-(ray.position[2]-self.fsurf)),np.tan(ray.angles[1])*(self.thickness-(ray.position[2]-self.fsurf)),(self.thickness-(ray.position[2]-self.fsurf))]





