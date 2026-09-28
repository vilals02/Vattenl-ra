#!/bin/bash

python convert_gmsh_xdmf.py unit_circle.geo 
python NSCLStandardGFEM.py
