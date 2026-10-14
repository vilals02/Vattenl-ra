#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

import meshio


# --------------------------------------------------------------------
def convert_geo_to_xdmf(geo_file, output_name=None):

    geo_path = Path(geo_file)

    if not geo_path.exists():
        raise RuntimeError(f"File not found: {geo_file}")

    if output_name is None:
        stem = geo_path.stem
    else:
        stem = output_name

    msh_file = f"{stem}.msh"
    xdmf_file = f"{stem}.xdmf"

    print(f"[1/3] Generating {msh_file} with gmsh")

    subprocess.run(
        [
            "gmsh",
            "-2",
            str(geo_path),
            "-format",
            "msh2",
            "-o",
            msh_file,
        ],
        check=True,
    )

    print(f"[2/3] Reading {msh_file}")

    msh = meshio.read(msh_file)

    print("[3/3] Extracting triangle mesh")

    triangles = msh.get_cells_type("triangle")

    triangle_tags = None
    try:
        triangle_tags = msh.get_cell_data(
            "gmsh:physical",
            "triangle",
        )
    except Exception:
        print("Warning: no physical tags found")

    mesh = meshio.Mesh(
        points=msh.points[:, :2],
        cells=[("triangle", triangles)],
        cell_data=(
            {"name_to_read": [triangle_tags]}
            if triangle_tags is not None
            else {}
        ),
    )

    meshio.write(xdmf_file, mesh)

    print(f"Done: wrote {xdmf_file}")


# --------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_gmsh_xdmf.py mesh.geo [output_name]")
        sys.exit(1)

    geo_file = sys.argv[1]

    output_name = None
    if len(sys.argv) >= 3:
        output_name = sys.argv[2]

    convert_geo_to_xdmf(geo_file, output_name)
