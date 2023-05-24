from enum import Enum

import math
from operator import itemgetter

from sympy.geometry import Polygon, Point, Segment

class BoundaryType(Enum):
    OUTER = 1
    INNER = 2
    VIRTUAL_OUTER = 3
    VIRTUAL_INNER = 4

class Boundary:
    def __init__(self, start:Point, end:Point, order:int, type:BoundaryType) -> None:
        self.start = start
        self.end = end
        self.segment = Segment(start, end)
        self.order = order
        self.type = type
    
    def distance(self, point:Point):
        return self.segment.distance(point)
    
    def perpendicular_line(self, point:Point):
        return self.segment.perpendicular_line(point)
    
    def slope(self):
        s = self.segment
        return math.atan2(s.direction.x, s.direction.y)
    
    def intersect(self, shape):
        return self.segment.intersect(shape)

class View:
    def __init__(self, belong) -> None:
        if belong:
            self.belong = belong
            self.boundaries = belong.boundaries
        self.subregions = []
        self.world = belong.world
    
    def get_subregions(self):
        return self.subregions
        
class World:
    def __init__(self, map) -> None:
        self.map = map
        corner_map = map.get_corners()
        last = corner_map.pop(0)
        first = last
        self.boundaries = []
        ord = 0
        outer = []
        for p in map:
            boundary = Boundary(last, p, ord, BoundaryType.OUTER)
            outer.append(boundary)
            last = p
            ord += 1
        boundary = Boundary(last, first, ord, BoundaryType.OUTER)
        outer.append(boundary)
        self.boundaries.append(outer)
        dummy = View(None)
        dummy.world = self
        self.root_region = Region(self.boundaries, dummy)
        self.regions = [self.root_region]
    
    def get_root_region(self):
        return self.root_region

    def get_regions(self):
        return self.regions
    
class Region:
    def __init__(self, outer:list, view, inner=[]) -> None:
        self.outer_boundaries = outer
        self.islands = inner
        self.outer_boundary = Polygon(outer)
        self.parent_view = view
        self.default_view = View(self)
        self.views = {}
        self.world = view.world

    def get_circumference(self):
        return self.outer_boundary

    def get_weight_center(self):
        return self.outer_boundary.center

    def get_corner_points(self):
        return self.outer_boundaries

    def is_inside(self, point):
        return self.outer_boundary.encloses_point(point)

    def get_near_boundaries(self, point:Point, thresh=1):
        cand = []
        for b in self.physical_boundaries:
            dl = float(b.distance(point))
            if dl > thresh: continue
            cand.append((dl, b))
        num_w = len(cand)
        if num_w == 0: return None
        return sorted(cand, key=itemgetter(0))

    def set_view(self, view_class, name):
        view = view_class(self.physical_boundaries)
        self.views[name] = view
        return view

    def get_view(self, name=None):
        if not name:
            return self.default_view
        else:
            return self.views.get(name)
    
    def get_subregions(self, view_name=None):
        if not view_name:
            return self.default_view.get_subregions()
        else:
            view = self.views.get(view_name)
            if not view: return None
            return view.get_subregions()
