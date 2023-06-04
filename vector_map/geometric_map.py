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
        self.belong = belong
        self.subregions = []
         
    def get_subregions(self):
        return self.subregions

class DefaultView(View):
    def __init__(self, belong) -> None:
        super().__init__(belong)
        s = belong.world.map.get_subregions()
        for sub_outer_coord in belong.world.map.get_subregions():
            sub_outer = []
            last = sub_outer_coord.pop(0)
            ord = 0
            first = last
            for p in sub_outer_coord:
                boundary = Boundary(last, p, ord, BoundaryType.OUTER)
                sub_outer.append(boundary)
                last = p
                ord += 1
            boundary = Boundary(last, first, ord, BoundaryType.OUTER)
            sub_outer.append(boundary)
            sr = Region(belong.world, sub_outer)
            self.subregions.append(sr)
        
class World:
    def __init__(self, map) -> None:
        self.map = map
        corner_map = map.get_corners()
        last = corner_map.pop(0)
        first = last
        self.boundaries = []
        ord = 0
        outer = []
        for p in corner_map:
            boundary = Boundary(last, p, ord, BoundaryType.OUTER)
            outer.append(boundary)
            last = p
            ord += 1
        boundary = Boundary(last, first, ord, BoundaryType.OUTER)
        outer.append(boundary)
        root_region = Region(self, outer)
        root_region.set_view(DefaultView)
        self.regions = [root_region]
        self.root_region = root_region
    
    def get_root_region(self):
        return self.root_region

    def get_regions(self):
        return self.regions
    
class Region:
    def __init__(self, world:World, outer:list, inner=[]) -> None:
        self.world = world
        self.outer_boundary = outer
        self.inner_boundaries = inner
        points = []
        for b in outer:
            points.append(b.start)
        self.corner_points = points
        self.outer_polygon = Polygon(*points)
        self.views = {}
        self.default_view = None

    def get_circumference(self):
        return self.outer_polygon

    def get_weight_center(self):
        return self.outer_polygon.center

    def get_corner_points(self):
        return self.corner_points

    def is_inside(self, point):
        return self.outer_polygon.encloses_point(point)

    def get_near_boundaries(self, point:Point, thresh=1):
        cand = []
        boundaries = [self.outer_boundary] + self.inner_boundaries
        for b in boundaries:
            if b.type != BoundaryType.INNER and b.type != BoundaryType.OUTER: continue
            dl = float(b.distance(point))
            if dl > thresh: continue
            cand.append((dl, b))
        num_w = len(cand)
        if num_w == 0: return None
        return sorted(cand, key=itemgetter(0))

    def set_view(self, view_class, name=None):
        view = view_class(self)
        if name:
            self.views[name] = view
        else:
            self.default_view = view
        return view

    def get_view(self, name=None):
        if not name:
            return self.default_view
        else:
            return self.views.get(name)
    
    def get_subregions(self, view_name=None):
        if not view_name:
            view = self.default_view
        else:
            view = self.views.get(view_name)
        if not view: return None
        return view.get_subregions()
