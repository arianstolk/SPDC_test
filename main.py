#########################################################
### Analysis/Ray-tracing module for Bulk SPDC sources ###
### Written by: Arian Stolk							  ###
### Date: September 2017							  ###
#########################################################

###The goal of this project is to provide a tool that will allow the investigation of SPDC sources made of bulk non-linear crystals.###
###Import library###

import numpy as np
import matplotlib
from vpython import *

class ExpSetup(object):
	"All instances of this class are a set of Optic elements along the z-direction. This class will handle all the preparation of the Optic elements to make them ready for the sequential ray tracing"

	def __init__(self, optic_elements=[]):
		self.elements = optic_elements
		self.nr_elements = len(optic_elements)

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
			self.material	=  k.pop('material')
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
		if !isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")

		lam=ray.wavelength

		return 1+(0.05792105)/(238-(lam*1e-3)**-2)+(0.00167917)/(57.362-(lam*1e-3)**-2)

	def getdn(self,ray):

		if !isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")

		lam=ray.wavelength

		return -(3358.34)/(((57.362-(1e6)/lam**2)**2)*lam**3)-(115842)/(((238-(1e6)/lam**2)**2)*lam**3)




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

		if !isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff(self.material)
		lam=ray.wavelength

		if ray.polarization == "H":
			if self.orientation in {"left","right"}:
				return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5
			else:
				return (coeff[4]+(coeff[5])/((lam*1e-3)**2-coeff[6])-coeff[7]*(lam*1e-3)**2)**0.5
		else:
			if self.orientation in {"up","down"}:
				return (coeff[0]+(coeff[1])/((lam*1e-3)**2-coeff[2])-coeff[3]*(lam*1e-3)**2)**0.5
			else:
				return (coeff[4]+(coeff[5])/((lam*1e-3)**2-coeff[6])-coeff[7]*(lam*1e-3)**2)**0.5
	
	def getdn(self,ray):

		if !isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff(self.material)
		lam=ray.wavelength
		if ray.polarization == "H":
			if self.orientation in {"left","right"}:
				x=sp.Symbol('x')
				ydiff=sp.diff((coeff[0]+(coeff[1])/((x*1e-3)**2-coeff[2])-coeff[3]*(x*1e-3)**2)**0.5,x)
				f = sp.lambdify(x,ydiff,'numpy')
				return f(lam)
			else:
				x=sp.Symbol('x')
				ydiff=sp.diff((coeff[4]+(coeff[5])/((x*1e-3)**2-coeff[6])-coeff[7]*(x*1e-3)**2)**0.5,x)
				f = sp.lambdify(x,ydiff,'numpy')
				return f(lam)
		else:
			if self.orientation in {"up","down"}:
				x=sp.Symbol('x')
				ydiff=sp.diff((coeff[0]+(coeff[1])/((x*1e-3)**2-coeff[2])-coeff[3]*(x*1e-3)**2)**0.5,x)
				f = sp.lambdify(x,ydiff,'numpy')
				return f(lam)
			else:
				x=sp.Symbol('x')
				ydiff=sp.diff((coeff[4]+(coeff[5])/((x*1e-3)**2-coeff[6])-coeff[7]*(x*1e-3)**2)**0.5,x)
				f = sp.lambdify(x,ydiff,'numpy')
				return f(lam)


	def DSellmeier(coeff=[0,0,0,0],lam = 785):
	x = sp.Symbol('x')
	ydiff = sp.diff(Sellmeier(coeff,x),x)
	f = sp.lambdify(x, ydiff, 'numpy')
	
	return f(lam)

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
		super(Lens,self),__init__(**k)

	def getn(self,ray):
		
		if !isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lam=ray.wavelength

		return (1+(c[0]*(lam*1e-3)**2)/(c[1]-(lam*1e-3)**2)+(c[2]*(lam*1e-3)**2)/(c[3]-(lam*1e-3)**2)+(c[4]*(lam*1e-3)**2)/(c[5]-(lam*1e-3)**2))**(0.5)

	def getdn(self,ray):
		
		if !isinstance(ray,Ray):
			raise Exception("Please provide me with a ray to calculate the refractive index for!")
		
		c=self.selm_coeff[self.material]
		lam=ray.wavelength
		x=sp.Symbol('x')
		ydiff=sp.diff((1+(c[0]*(x*1e-3)**2)/(c[1]-(x*1e-3)**2)+(c[2]*(x*1e-3)**2)/(c[3]-(x*1e-3)**2)+(c[4]*(x*1e-3)**2)/(c[5]-(x*1e-3)**2))**(0.5),x)
		f = sp.lambdify(x,ydiff,'numpy')
		return f(lam)





