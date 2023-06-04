import sys
sys.path.append(".")

from vector_map import *

init_visualize()
world = get_map_ROS("resource/slamdata/matsuken_map6") 
#world = get_map_ROS("resource/map")
r = world.get_root_region()
ss = SimulationSpace(r)
ss.show_outer_boundary()
ss.show_subregions()