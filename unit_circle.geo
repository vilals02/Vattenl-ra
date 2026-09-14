// unit_circle.geo
//
// Geometry for the unit disk  { (x,y) : x^2 + y^2 <= 1 }
//
// Usage (command line):
//   gmsh unit_circle.geo -2 -o circle.msh
//
// Or use make_circle_mesh.py which runs gmsh programmatically.

lc = 0.05;   // characteristic mesh size (reduce for finer mesh)

// Centre and four equidistant points on the unit circle
Point(1) = { 0,  0, 0, lc};   // centre (not a mesh vertex, only for arcs)
Point(2) = { 1,  0, 0, lc};   // right
Point(3) = { 0,  1, 0, lc};   // top
Point(4) = {-1,  0, 0, lc};   // left
Point(5) = { 0, -1, 0, lc};   // bottom

// Four circular arcs forming the boundary
Circle(1) = {2, 1, 3};   // right  -> top
Circle(2) = {3, 1, 4};   // top    -> left
Circle(3) = {4, 1, 5};   // left   -> bottom
Circle(4) = {5, 1, 2};   // bottom -> right

// Closed boundary loop and surface
Curve Loop(1)    = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical groups (required by DOLFINx gmshio; also kept for meshio tagging)
Physical Surface("domain",   1) = {1};
Physical Curve("boundary",   2) = {1, 2, 3, 4};
