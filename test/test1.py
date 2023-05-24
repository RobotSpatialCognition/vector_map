from sympy import Point, Ray
import sys
sys.path.append(".")
from vector_map import *
from vector_map.visualize import init_visualize

class MapHandleUI:
    def __init__(self) -> None:
        world = get_map_ROS("../resource/map")
        r = world.get_root_region()

        ss = SimulationSpace(r)
        self.simulation_space = ss
        ss.set_callback("set_object", self.set_object_handler)
        ss.start_mouse()
#        ss.draw(r)
#        ss.loop()
    
    def set_object_handler(self, x, y):
        self.simulation_space.create_circle(x, y, 10, color="red")
        # obj = Object(x, y, 10)
        
        # walls = self.region.get_near_walls(obj.center)
        points = []
        # for w in walls:
            # ir = w[1].intersect(obj)
            # points += ir
        mid = points[0].midpoint(points[1])
        org = Point(x, y)
        ray = Ray(org, mid)
        self.simulation_space.draw_line(ray, arrow=True)

init_visualize()
ui = MapHandleUI()
