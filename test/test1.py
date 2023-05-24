import sys
import cv2

from sympy import Point, Ray

sys.path.append('/home/tago/map/geometric_map')
from geometric_map.visualize import SimulationSpace
import demo_maplibrary.demo_maplibrary.demo_maplibrary as mlb
from geometric_map.api import World, Region, Object

class MapHandleUI:
    def __init__(self) -> None:
        map = cv2.imread('kenA_rote.pgm', cv2.IMREAD_GRAYSCALE)
        #map = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
        mapcenter , r = mlb.make_mapbb(map)
        img_org = map[mapcenter[1]-r:mapcenter[1]+r, mapcenter[0]-r:mapcenter[0]+r]

        a,b,c,map = mlb.getMovePoint(img_org)

        r = Region(map.tolist())
        self.region = r

        ss = SimulationSpace()
        self.simulation_space = ss
        ss.set_callback("set_object", self.set_object_handler)
        ss.start_mouse()
        ss.draw(r)
        ss.loop()
    
    def set_object_handler(self, x, y):
        self.simulation_space.create_circle(x, y, 10, color="red")
        obj = Object(x, y, 10)
        walls = self.region.get_near_walls(obj.center)
        points = []
        for w in walls:
            ir = w[1].intersect(obj)
            points += ir
        mid = points[0].midpoint(points[1])
        org = Point(x, y)
        ray = Ray(org, mid)
        self.simulation_space.draw_line(ray, arrow=True)

ui = MapHandleUI()
