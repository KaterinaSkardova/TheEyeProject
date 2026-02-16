import numpy as numpy
import os
import meshio

base = "/Users/skardova/Documents/MRI_data/2026-Eyes-project/Trial2/"
folder = "VTI_shared_view/"

mesh_filename = "2-views_segment.vtk"

matrix = numpy.loadtxt(base + folder + os.sep + "matrix_poz1.txt")

#############################################################################

mesh = mesh = meshio.read(base + folder +'segmentations/' + mesh_filename)


points = mesh.points
cells = mesh.cells
cell_data = mesh.cell_data


new_cells = []
new_cell_data = {}

# Loop over each cell block (e.g., triangle, tetra, etc.)
for i, cell_block in enumerate(cells):
    cell_type = cell_block.type
    cell_array = cell_block.data
    
    # Get corresponding cell data array
    # (assumes only one data name exists)
    data_name = list(cell_data.keys())[0]
    tags = cell_data[data_name][i]

    # Select only cells with tag == 2
    mask = tags == 2
    filtered_cells = cell_array[mask]
    filtered_tags = tags[mask]

    if len(filtered_cells) > 0:
        new_cells.append((cell_type, filtered_cells))
        
        if data_name not in new_cell_data:
            new_cell_data[data_name] = []
        new_cell_data[data_name].append(filtered_tags)

# Create new mesh
new_mesh = meshio.Mesh(
    points=points,
    cells=new_cells,
    cell_data=new_cell_data
)

# Write mesh
meshio.write(base + folder +'segmentations/' + mesh_filename[:-4] + "_filtered.vtk", new_mesh)