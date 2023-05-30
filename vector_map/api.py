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
	#data:boolの配列にすべき
	pass

class RasterProperty(Raster):
	pass
	
class PixType(Enum):
	INNER = 2
	OUTER = 1
	WALL   = 12
	CORNER = 15


class VectorMap:
	def __init__(self,raster,ksize=11, epsilon=3):
		self.raster = raster
		self.offset_x = 0
		self.offset_y = 0
		self.rotation = 0

		data = raster.data #ndarray
		center, r  = vectorize.make_mapbb(data)
		croped_raster,offset = vectorize.img_crop(data)
		self.denoised_raster = vectorize.gen_sk_map(croped_raster, ksize)
		self.bin_raster = self.denoised_raster/255
		self.shapeup_raster()
		tmp_property, corner_list = vectorize.addition_property(self.bin_raster)
		temp, clist, dlist = vectorize.approximate_corner(tmp_property, corner_list)


		self.corners = np.empty((len(clist),2),dtype=np.int64)
		for n,c in enumerate(clist):
			self.corners[n][0] = c[0]
			self.corners[n][1] = c[1]
		self.corners[:,0] += offset[0]
		self.corners[:,1] += offset[1]





	def get_denoised_raster(self):
		raster = Raster()
		raster.data = self.denoised_raster
		raster.scale = self.raster.scale
		return raster

	def get_coord(self,p):
		#clipの補正
		shape = self.raster.shape
		resolution = self.raster.resolution
		origin = self.raster.origin
		# base_x = -0.75 #pix * resolution
		# base_y = shape[1]*resolution - 1.1

		base_x = self.offset_x
		base_y = shape[1]*resolution + self.offset_y 

    	# further adjustment: move origin from (0, 0) to config.origin
		base_x += origin[0]
		base_y += origin[1]

		return float(p[1])*resolution+base_x, -float(p[0])*resolution+base_y 

	def get_corners(self):
		# デカルト座標でoriginを原点とした座標点のリスト
		points = []
		for p in self.corners:
			points.append(self.get_coord(p))
		print(points)
		print(len(points))
		return points



	def get_raster_property(self):
		prop = RasterProperty()
		prop.data = self.pix_property
		prop.scale = self.raster.scale
		return prop
	
	def shapeup_raster(self):
		self.bin_raster = np.pad(self.bin_raster, 10, constant_values=0)
		self.offset_x += 10 * self.raster.resolution 
		self.offset_y += 10 * self.raster.resolution 
		## clip 補正


	def shapeup_raster(self):
		self.bin_raster = np.pad(self.bin_raster, 10, constant_values=0)
		self.offset_x += 10 * self.raster.resolution 
		self.offset_y += 10 * self.raster.resolution 
		#clip
		






def get_map_ROS(dir):
	raster = Raster()
	if dir.startswith('~'):
		dir = os.path.expanduser(dir)
	map_file = dir + ".pgm"
	meta_file = dir + ".yaml"
	map_img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
	raster.data = map_img
	raster.shape = map_img.shape

    # read meta data
	with open(meta_file, 'r') as yml:
		config = yaml.safe_load(yml)
	raster.resolution = float(config['resolution'])
	raster.origin = config['origin']

	vector_map = VectorMap(raster)
	geometric_map = World(vector_map)
	return geometric_map
