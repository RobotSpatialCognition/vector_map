#raster: Point cloud representation
#map: Vector representation
import vectorize 
from enum import Enum
import numpy as np
import os
import cv2
import yaml
from .geometric_map import World
#@dataclass
class Raster:
	#data:boolの配列にすべき
	pass

class Property(Raster):
	pass
	
class PixType(Enum):
	INNER = 2
	OUTER = 1
	WALL   = 12
	CORNER = 15


class VectorMap:
	def __init__(self,raster,ksize=11, epsilon=3):
		self.raster = raster
		data = raster.data #ndarray
		img = self.make_mapbb(data)
		croped_raster,offset = vectorize.img_crop(img)
		self.denoised_raster = vectorize.gen_sk_map(croped_raster, ksize)
		bin_raster = self.denoised_raster/255
		# bin_raster = np.pad(bin_raster, 10, constant_values=0)
		# bin_raster = bin_raster.astype(np.int32)
	
		# p_map = vectorize.wall_detected1(bin_raster) ##壁とそれ以外を分類
		# p2_map = vectorize.wall_detected2(p_map,bin_raster) ##暫定コーナーの抽出
		# p3_map = vectorize.wall_detected3(p2_map, bin_raster) ##内部と外部を塗り分け
		# p = vectorize.renew_corner_func(p3_map, bin_raster)
		# corner_list = vectorize.get_corner_list_from_pm(p)
		tmp_property, corner_list = vectorize.addition_property(bin_raster)
		temp, clist, dlist = vectorize.approximate_corner(tmp_property, corner_list)


		self.corners = np.empty((len(clist),2),dtype=np.int64)
		for n,c in enumerate(clist):
			self.corners[n][0] = c[0]
			self.corners[n][1] = c[1]
		self.corners[:,0] += offset[0]
		self.corners[:,1] += offset[1]

		# nodes = vectorize.search_nearpoint(clist, dlist)




	def get_denoised_raster(self):
		raster = Raster()
		raster.data = self.denoised_raster
		# raster.center = self.raster ##offset分と計算した値
		raster.scale = self.raster.scale
		return raster

	def get_coord(self,p):
		shape = self.raster.shape
		resolution = self.raster.resolution
		origin = self.raster.origin
		base_x = -0.75 #pix * resolution
		base_y = shape[1]*resolution - 1.1
    	# further adjustment: move origin from (0, 0) to config.origin
		base_x += origin[0]
		base_y += origin[1]
		return float(p[1])*resolution+base_x, -float(p[0])*resolution+base_y

	def get_corners(self):
		# デカルト座標でoriginを原点とした座標点のリスト
		points = []
		for p in self.corners:
			points.append(self.get_coord(p))
		return points



	def get_raster_property(self):
		prop = Property()
		prop.data = self.pix_property
		#raster.center = self.raster ##offset分と計算した値
		prop.scale = self.raster.scale
		return prop


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
