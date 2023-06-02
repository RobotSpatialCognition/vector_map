#raster: Point cloud representation
#map: Vector representation

from enum import Enum
import numpy as np
import os
import cv2
import yaml
import copy

from . import vectorize
from .geometric_map import World

# class to keep and handle raw data of pixel data
class Raster:
	# Raster(data, resolution)
	#    data: ndarray
	#    resolution: size in meter / pixel
	def __init__(self, data, resolution) -> None:
		self.data = data
		self.resolution = resolution
		self.offset_x = 0
		self.offset_y = 0
		
		self.shape = data.shape
		self.rotation = None
	#data:boolの配列にすべき
	
	def pix_to_coord(self, pix_x, pix_y):
		if self.rotation:
			#pix_x += self.clipped_origin_x
			#pix_y += self.clipped_origin_y
			reverse_mat = [[np.cos(self.rotation),-1 * np.sin(self.rotation)],[np.sin(self.rotation),np.cos(self.rotation)]]
			pix_x,pix_y = np.dot(reverse_mat,np.array([pix_x,pix_y])) # ndarray(y, x)
		coord_x = float(pix_x)*self.resolution + self.offset_x
		height = self.data.shape[0] * self.resolution
		coord_y = -float(pix_y)*self.resolution + height + self.offset_y
		return coord_x, coord_y
	
	# move the origin
	def move(self, x, y):
		if self.rotation:
			self.post_offset_x += x
			self.post_offset_y += y
		else:
			self.offset_x += x
			self.offset_y += y
	
	# rotate the map image. The center of rotation is the center of map image
	def rotate(self, angle):
		if angle == 0: return
		self.rotation = angle
		self.data = vectorize.rotation(self.data, self.rotate)
		self.post_offset_x = 0 # offset for post rotation
		self.post_offset_y = 0

	# cut off unnecessary parts of the map image
	def clip(self):
		self.data, (clipped_origin_x,clipped_origin_y) = vectorize.img_crop(self.data)
		# adjust offset to cancel clipping effect
		self.move(clipped_origin_x*self.resolution, clipped_origin_y*self.resolution)
	
	# thinning map lines
	def denoize(self, ksize):
		self.data = vectorize.gen_sk_map(self.data, ksize)
	
	# adding margin to the map perimeter
	def pad(self):
		self.data = np.pad(self.data, 10, constant_values=0)
		self.move(-10 * self.resolution, -10 * self.resolution) 

class PixType(Enum):
	INNER = 2
	OUTER = 1
	WALL   = 12
	CORNER = 15

# low layer map class to handle pixel level operations
class VectorMap:
	def __init__(self,raster,ksize=11, epsilon=3):
		self.raster = raster

		# create target raster to operate: bin_raster
		bin_raster = copy.copy(raster)
		bin_raster.clip()
		bin_raster.denoize(4) # ksize=5?
		self.denoised_raster = copy.copy(bin_raster)
		bin_raster.pad()
		self.bin_raster = bin_raster

		# generate corner coordinate list
		tmp_property, corner_list = vectorize.addition_property(bin_raster.data)
		_, clist, _ = vectorize.approximate_corner(tmp_property, corner_list)
		self.corners = np.empty((len(clist),2),dtype=np.int64)
		for n,c in enumerate(clist):
			self.corners[n][0] = c[0]
			self.corners[n][1] = c[1]
		
		# get subregions
		_, _, _, _, labelImage = vectorize.getMovePoint(bin_raster.data)
		subregions = vectorize.get_subregion_points(labelImage)	
		print(subregions)

	def get_denoised_raster(self):
		return self.denoised_raster

	# generates Cartesian coordinate of corners
	def get_corners(self):
		points = []
		for py,px in self.corners:
			points.append(self.bin_raster.pix_to_coord(px, py))
		print(points)
		return points

# generates World map from map files in ROS format
def get_map_ROS(dir):
	if dir.startswith('~'):
		dir = os.path.expanduser(dir)
	map_file = dir + ".pgm"
	meta_file = dir + ".yaml"
	map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)

    # read meta data
	with open(meta_file, 'r') as yml:
		config = yaml.safe_load(yml)
	resolution = float(config['resolution'])
	origin = config['origin']

	# create VectorMap form raster object
	raster = Raster(map_img, resolution)
	raster.move(origin[0],origin[1])
	map = VectorMap(raster)

	return World(map)
