import sys
sys.path.append(".")
from vector_map import *

init_visualize()
world = get_map_ROS("resource/map")
r = world.get_root_region()
ss = SimulationSpace(r)
