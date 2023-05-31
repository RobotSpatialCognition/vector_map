#raster: Point cloud representation
#map: Vector representation

from enum import Enum
import numpy as np
import os
import cv2
import yaml

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
		self.rotation = 0.0
		self.cliped_origin_x = 0
		self.cliped_origin_y = 0
	#data:boolの配列にすべき
	
	def pix_to_coord(self, pix_x, pix_y):
		reverse_mat = [[np.cos(self.rotation),-1 * np.sin(self.rotation)],[np.sin(self.rotation),np.cos(self.rotation)]]
		pix_x,pix_y = np.dot(reverse_mat,np.array([pix_y,pix_x])) # ndarray(y, x)
		pix_x += self.cliped_origin_x
		pix_y += self.cliped_origin_y
#		coord_x = float(pix_x)*self.resolution + self.offset_x
		coord_x = float(pix_x)*self.resolution
		height = self.data.shape[0] * self.resolution
#		coord_y = -float(pix_y)*self.resolution + height + self.offset_y
		coord_y = -float(pix_y)*self.resolution + height
		return coord_x, coord_y
	
	def move(self, x, y):
		self.offset_x += x
		self.offset_y += y
	
	def rotate(self, angle):
		self.rotate = angle
		self.data = vectorize.rotation(self.data, self.rotate)

	def clip(self):
		self.data, (clipped_origin_x,clipped_origin_y) = vectorize.img_crop(self.data)
		self.offset_x += clipped_origin_x
		self.offset_y += clipped_origin_y
	
class PixType(Enum):
	INNER = 2
	OUTER = 1
	WALL   = 12
	CORNER = 15


class VectorMap:
	def __init__(self,raster,ksize=11, epsilon=3):
		self.raster = raster

		bin_raster = self.create_target_raster()
		self.bin_raster = bin_raster

		tmp_property, corner_list = vectorize.addition_property(bin_raster.data)
		_, clist, _ = vectorize.approximate_corner(tmp_property, corner_list)
		self.corners = np.empty((len(clist),2),dtype=np.int64)
		for n,c in enumerate(clist):
			self.corners[n][0] = c[0]
			self.corners[n][1] = c[1]

	def create_target_raster(self,ksize = 5):
		data = self.raster.data #ndarray
		resolution = self.raster.resolution
		center, r  = vectorize.make_mapbb(data)

		croped_raster,offset = vectorize.img_crop(data)
		denoised_raster = vectorize.gen_sk_map(croped_raster, ksize)
		self.denoised_raster = denoised_raster
		bin_raster_data = np.pad(denoised_raster, 10, constant_values=0)
		origin_x = self.raster.offset_x
		origin_y = self.raster.offset_y
		origin_x += 10 * self.raster.resolution 
		origin_y += 10 * self.raster.resolution 
		bin_raster = Raster(bin_raster_data, resolution)
		bin_raster.move(origin_x,origin_y)
		return bin_raster

	def get_denoised_raster(self):
		raster = Raster(self.denoised_raster, self.raster.resolution, self.raster.origin)
		return raster

	def get_corners(self):
		# デカルト座標でoriginを原点とした座標点のリスト
		points = []
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
