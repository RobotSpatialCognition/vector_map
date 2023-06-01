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

#@dataclass
class Raster:
	def __init__(self, data, resolution) -> None:
		self.data = data
		self.resolution = resolution
		self.offset_x = 0
		self.offset_y = 0
		
		self.shape = data.shape
		# self.height = data.shape[1] * resolution
		self.rotation = None
		self.clipped_origin_x = 0
		self.clipped_origin_y = 0
	#data:boolの配列にすべき
	
	def pix_to_coord(self, pix_x, pix_y):
		if self.rotation:
			#pix_x += self.clipped_origin_x
			#pix_y += self.clipped_origin_y
			reverse_mat = [[np.cos(self.rotation),-1 * np.sin(self.rotation)],[np.sin(self.rotation),np.cos(self.rotation)]]
			pix_x,pix_y = np.dot(reverse_mat,np.array([pix_y,pix_x])) # ndarray(y, x)
		else:
			x = pix_x
			pix_x = pix_y
			pix_y = x
		coord_x = float(pix_x)*self.resolution + self.offset_x
		height = self.data.shape[0] * self.resolution
		coord_y = -float(pix_y)*self.resolution + height + self.offset_y
		return coord_x, coord_y
	
	def move(self, x, y):
		if self.rotation:
			self.post_offset_x += x
			self.post_offset_y += y
		else:
			self.offset_x += x
			self.offset_y += y
	
	def rotate(self, angle):
		if angle == 0: return
		self.rotation = angle
		self.data = vectorize.rotation(self.data, self.rotate)
		self.post_offset_x = 0 # offset for post rotation
		self.post_offset_y = 0

	def clip(self):
		self.data, (clipped_origin_x,clipped_origin_y) = vectorize.img_crop(self.data)
		# adjust offset to cancel clipping effect
		self.move(clipped_origin_x*self.resolution, clipped_origin_y*self.resolution)
	
	def denoize(self, ksize):
		self.data = vectorize.gen_sk_map(self.data, ksize)
	
	def pad(self):
		self.data = np.pad(self.data, 10, constant_values=0)
		self.offset_x += 10 * self.resolution 
		self.offset_y += 10 * self.resolution 

class PixType(Enum):
	INNER = 2
	OUTER = 1
	WALL   = 12
	CORNER = 15

class VectorMap:
	def __init__(self,raster,ksize=11, epsilon=3):
		self.raster = raster

		# create target raster to operate: bin_raster
		bin_raster = copy.copy(raster)
		bin_raster.clip()
		bin_raster.denoize(ksize) # ksize=5?
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

	def get_denoised_raster(self):
		return self.denoised_raster

	def get_corners(self):
		# デカルト座標でoriginを原点とした座標点のリスト
		points = []
		print(len(self.corners))
		for px,py in self.corners:
			points.append(self.bin_raster.pix_to_coord(px, py))
		print(points)
		return points

	def set_property(self, prop):
		self.pix_property = prop

	def get_property(self):
		return self.pix_property

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
	map.set_property(config)

	return World(map)
