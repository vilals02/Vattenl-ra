// cylinder.geo
// Rectangular channel with a circular cylinder obstacle.
//
// Usage:
//   gmsh cylinder.geo -2 -o cylinder.msh
//   python convert_gmsh_xdmf.py cylinder.msh cylinder.xdmf

L   = 10.0;    // channel length
H   = 8.0;    // channel height
cx  = 3.0;    // cylinder centre x
cy  = 4.0;    // cylinder centre y
cr  = 1.0;   // cylinder radius

res_far = 0.5;   // mesh size far from cylinder
res_cyl = 0.25;   // mesh size on cylinder

// ── rectangle corners ────────────────────────────────────────────────────────
Point(1) = {0,  0,  0, res_far};
Point(2) = {L,  0,  0, res_far};
Point(3) = {L,  H,  0, res_far};
Point(4) = {0,  H,  0, res_far};

Line(1) = {1, 2};   // bottom wall
Line(2) = {2, 3};   // right  (outflow)
Line(3) = {3, 4};   // top wall
Line(4) = {4, 1};   // left   (inflow)

// ── cylinder: four quarter-circle arcs, counter-clockwise ────────────────────
Point(5) = {cx,    cy,    0, res_cyl};   // centre (not meshed)
Point(6) = {cx+cr, cy,    0, res_cyl};   // rightmost
Point(7) = {cx,    cy+cr, 0, res_cyl};   // top
Point(8) = {cx-cr, cy,    0, res_cyl};   // leftmost
Point(9) = {cx,    cy-cr, 0, res_cyl};   // bottom

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

// ── surface with hole ────────────────────────────────────────────────────────
Curve Loop(1) = {1, 2, 3, 4};      // outer boundary
Curve Loop(2) = {5, 6, 7, 8};      // cylinder (inner hole)
Plane Surface(1) = {1, 2};

// ── physical groups (written into the .msh for boundary tagging) ─────────────
Physical Curve("inflow")   = {4};
Physical Curve("outflow")  = {2};
Physical Curve("walls")    = {1, 3};
Physical Curve("cylinder") = {5, 6, 7, 8};
Physical Surface("domain") = {1};

// ── mesh refinement: fine near cylinder, coarse far away ─────────────────────
Field[1] = Distance;
Field[1].CurvesList = {5, 6, 7, 8};
Field[1].Sampling   = 200;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = res_cyl;
Field[2].SizeMax = res_far;
Field[2].DistMin = 0.0;
Field[2].DistMax = 0.5;

Background Field = 2;
