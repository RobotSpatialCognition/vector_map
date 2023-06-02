# vector_map api

vector_map has the function of reading an occupancy grid format SLAM map and converting it into an ordered connection of line segments which consist of straight lines and/or curved portions of constant curvature. This is called vectorization of the map, and the generated map is called a vector map.
vector_map further has ability to express this vector map in an object format and has the function of hierarchically decomposing it into multiple partial maps. Using a map represented by objects, various particles in the environment can also be added as properties in the form of objects. The ultimate goal is to use it as a basis for aggregating various semantic information about the environment.

vector_map expresses the target terrain with two types of objects, Boundary and Region. Boundary represents a mathematical line segment, and fall into a PhysicalBoundary type representing a physical boundary detected by a sensor or a VirtualBoundary created by an API. Region represents a physical area. A Region's perimeter is completely enclosed by an ordered chain of Boundaries.
The inside of a Region may contain an area that is not connected to the outer circumference. Such internal area is also represented by Region. Therefore, Regions have a nested structure, and the contained child Regions are called Subregions.

Subregions can also be created from a parent Region with applying a specific division algorithm to it. For example, it corresponds to dividing a Region representing an entire house into Regions of interior rooms and corridors. The object that holds this algorithm is called a View and also contains Subregions that it creates. Since a Region can have multiple Views, the division into Subregions is not limited to one specific type.

In vector_map the entire map is represented by a World object.

```
class vector_map.Raster(data, resolution)
   A map in the form of an occupancy grid
   Parameters:
      data: (Nx ⅹ Ny, d) array
         Nx ⅹ Ny occupancy grid data represented by ndarray

class vector_map.World(map)
   Map in vector format
   Parameters:
      map: vector_map.Raster
         Occupancy grid format map

   get_root_region():
      returns the Regions contained in the map that contain the coordinate origin.
      Returns:
         vector_map.Region

   get_regions():
      returns a list of all Regions contained in the map.
      Returns:
         list of vector_map.Region

class vector_map.Boundary(start, end, order, type)
   A geometric unit element representing the boundary of the Region.
   Parameters:
      start: sympy.geometry.Point
         Starting point of Boundary. Since Boundary is a geometric element,
         each element of sympy.geometry is used for the type of geometric
         parameters such as coordinate points.
      end: sympy.geometry.Point
         End point of Boundary.
      order: int
         Boundary number. A unique sequence number is assigned to the Boundary
         surrounding one Region. Note that the numbers are serial numbers for each
         Region, not globally unique.
      type: vector_map.BoundaryType
         Boundaries include those that are physically detected by sensors
         and those that are virtually created by views. 'type' indicates
         this distinction.

   distance(point)
      returns distance from external point to this Boundary.
      Returns:
         float
   
   perpendicular_line(self, point)
      returns a perpendicular to this Boundary through point.
      Returns:
         sympy.geometry.Line
      Parameters:
         point: sympy.geometry.Point

   slope()
      returns the angle between this Boundary and X-axis in radian.
      Returns:
         float

   intersect(shape)
      returns intersection with shape.
      Returns:
         list of sympy.geometry.Point
      Parameters:
         shape: sympy.geometry.Point,Line,Segment,Ray
   
   Properties:
      start, end: sympy.geometry.Point
      segment: sympy.geometry.Segment
         The shape of this Boundary represented by geometric line segment.
      order: int
      type: vector_map.BoundaryType

class vector_map.View(belong)
   A set of subregions contained in a Region. A subregion may be a physically independent region from the beginning, or it may be created by virtually dividing a parent region. Division is done by inserting a virtual Boundary into the parent Region. This operation is called "cut".

   The cut operation is not unique, so multiple views can be set for a single parent Region. Each View must divide the entire region of the parent Region into any subregion, and no partial region corresponding to the subregion must be generated. Also, Boundaries owned by the parent Region must be dispatched to one of subregions without duplication.

   The View class does not include the cut operation algorithm, so it must be created in a class that inherits the View class.
   Parameters:
      belong: vector_map.Region
         the parent Region to which this View belongs

   get_subregions()
      returns the subregions of this View in list format.
      Returns:
         list of vector_map.Region

class vector_map. Region(outer, view, inner)
   A closed area contained in the World.
   Parameters:
      outer: list of vector_map.Boundary
         a list of Boundaries representing the perimeter of this Region
      view: vector_map.View
         the default View that this Region has
      inner: list of vector_map.Region
         a list of independent Regions within this Region

   get_circumference():
      returns a polygon that encloses this Region.
      Returns:
         sympy.geometry.Polygon

   get_weight_center():
      returns the centroid point of the region represented by this Region.
      Returns:
         sympy.geometry.Point

   get_corner_points():
      returns a list of corner points of the region represented by this Region.
      Returns:
         list of sympy.geometry.Points

   is_inside(point):
      Determines if point is inside this Region.
      Parameters:
         point: sympy.geometry.Point
      Returns:
         boolean

   get_near_boundaries(point, thresh=1):
      returns a Boundary that is close to the point.
      Parameters:
         point: sympy.geometry.Point
         thresh: int
            Denote the detection range by [m]. Default value is 1m.
      Returns:
         list of vector_map.Boundary

   set_view(view_class, name):
       registers the View class by name.
       Parameters:
          view_class: View and its inherited class
             View class to be registered
          name: string
             the name given to the view
       Returns:
          vector_map.View
             created view instance

   get_view(name):
      get registered View instance.
      Parameters:
         name: string
            View name. Default View if omitted
      Returns:
         View instance

   get_subregions(view_name):
      get the list of subregions that the registered View has.
      If the name is omitted, the subregion of the default View.
      Parameters:
         view_name: string
      Returns:
         list of vector_map.Region
```