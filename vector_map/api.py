#raster: Point cloud representation
#map: Vector representation

from enum import Enum
#@dataclass
class Raster:
	#data:boolの配列にすべき
	pass

class Property(Raster):
	pass
	
class PixType(Enum):
	INNER = 2
	OUTER = 1
	CORNER = 11
	WALL   = 12

class VectorMap:
	def __init__(self,raster,ksize=11, epsilon=3):
		self.raster = raster
		data = raster.data #ndarray
		self.make_mapbb(img)
		croped_raster,offset = img_crop(raster)
		self.denoised_raster =gen_sk_map(croped_raster, 11)

		pass
	def get_denoised_raster(self):
		raster = Raster()
		raster.data = self.denoised_raster
		# raster.center = self.raster ##offset分と計算した値
		raster.scale = self.raster.scale
		return raster 

	def get_corners(self):
		# デカルト座標でoriginを原点とした座標点のリスト
		pass

	def get_raster_property(self):
		prop = Property()
		prop.data = self.pix_property
		# raster.center = self.raster ##offset分と計算した値
		prop.scale = self.raster.scale
		return prop


def ROS_map_factory(dir):

	return geometric_map