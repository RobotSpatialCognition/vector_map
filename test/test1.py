import sys
sys.path.append(".")

from vector_map import *

init_visualize()
world = get_map_ROS("resource/matsuken_map6") 
r = world.get_root_region()
ss = SimulationSpace(r)
ss.show_outer_boundary()
#ss.show_subregions()