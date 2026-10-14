#!/bin/bash

python convert_gmsh_xdmf.py unit_circle.geo 
python simulate_init_A.py
python simulate_init_A_stabilized.py
python simulate_init_B.py
python simulate_init_B_stabilized.py
python compare_euler_rk4.py