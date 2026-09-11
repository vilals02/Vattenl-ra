#!/bin/bash

python convert_gmsh_xdmf.py unit_circle.geo 
mpiexec -n 1 python demo_assemblers.py