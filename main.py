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
	
	def visualize(self):
		boxes=[]
		for elem in self.elements:
			boxes.append(box(pos=vec(0,0,elem.position[2]),size=vec(2,2,elem.thickness),opacity=0.3,color=elem.get_colour()))
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
		"""Name is not purely cosmetic, it has to follow these rules:"""
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

	####NEED TO ADD VISUALIZATION TOOLS###


class Crystal(Optic):
	"All instances of this class (child of Optics), are objects that are (non-linear) crystals that act on the photons. They are all rectangular boxes that are made of a birefringent material"

	def __init__(self,**k):
		try:
			self.orientation 	= k.pop('orientation')
			self.cutangle		= k.pop('cutangle')
		except KeyError:
			print("Please provide at least orientation and cutangle to initiate a Crystal object")
		super(Crystal,self).__init__(**k)

class Lens(Optic):
	"All instances of this class (Child of optics) are objects that are curved surfaces (spherical) that act on the photons. One can have two different Radii of Curvature, so this field should be a vector"

	def __init__(self,**k):
		try:
			self.ROC 	= k.pop('ROC')
			self.centre = k.pop('centre')
		except KeyError:
			print("Please provide at least ROC (Radii of Curvature) and centre position to initiate a Lens object")
		super(Lens,self),__init__(**k)
