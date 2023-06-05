import sys
sys.path.append(".")

from vector_map import *

init_visualize()
world = get_map_ROS("resource/map")
r = world.get_root_region()
ss = SimulationSpace(r)
ss.show_outer_boundary()
ss.show_subregions()
for sr in r.get_subregions():
    w = sr.get_weight_center()
    ss.create_mark(w.x, w.y)
