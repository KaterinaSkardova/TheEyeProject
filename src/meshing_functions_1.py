import vtk 
from vtk.util import numpy_support
import numpy
import gmsh
import meshio
from scipy.spatial import cKDTree
# from collections import defaultdict
import os

def mask_to_surfaces(mask, spacing=(1.0,1.0,1.0)):
    """Convert a labeled mask to vtkPolyData surfaces (ignoring label 0)."""
    surfaces = {}
    dims = mask.shape
    img = vtk.vtkImageData()
    img.SetDimensions(dims)
    img.SetSpacing(*spacing)
    img.SetOrigin(0,0,0)

    flat = mask.ravel(order='F')
    vtk_data = numpy_support.numpy_to_vtk(flat.astype(numpy.uint8), deep=True)
    img.GetPointData().SetScalars(vtk_data)

    labels = numpy.unique(mask)
    labels = labels[labels != 0]
    for label in labels:
        mc = vtk.vtkDiscreteMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, label)
        mc.Update()
        surfaces[label] = mc.GetOutput()
    return surfaces

def generate_eyeball_muscle_mesh(mask, spacing=(1.0,1.0,1.0), shell_thickness=1.0,
                                 element_size=1.0, output_file="eye_muscle.vtk"):

    surfaces = mask_to_surfaces(mask, spacing=spacing)
    
    # --- Write temporary STL surfaces ---
    tmp_dir = "tmp_gmsh_stl"
    os.makedirs(tmp_dir, exist_ok=True)
    label_to_stl = {}
    for label, polydata in surfaces.items():
        stl_path = os.path.join(tmp_dir, f"label{label}.stl")
        stl_writer = vtk.vtkSTLWriter()
        stl_writer.SetInputData(polydata)
        stl_writer.SetFileName(stl_path)
        stl_writer.Write()
        label_to_stl[label] = stl_path

    # --- Initialize Gmsh ---
    gmsh.initialize()
    gmsh.model.add("eye_muscle_model")
    surface_tags = {}

    # Import STL surfaces
    for label, stl_file in label_to_stl.items():
        gmsh.merge(stl_file)
        # Get newly added surfaces (last imported)
        entities = gmsh.model.getEntities(2)
        surface_tags[label] = [e[1] for e in entities]

    # --- Create shell volume for eyeball ---
    eyeball_label = 1
    # Use inner + outer surface (approximate shell thickness in mask)
    eyeball_surfaces = surface_tags[eyeball_label]
    sl = gmsh.model.geo.addSurfaceLoop(eyeball_surfaces)
    vol_tag_eyeball = gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # --- Muscle surfaces ---
    muscle_label = 2
    muscle_surfaces = surface_tags[muscle_label]
    sl_muscle = gmsh.model.geo.addSurfaceLoop(muscle_surfaces)
    vol_tag_muscle = gmsh.model.geo.addVolume([sl_muscle])
    gmsh.model.geo.synchronize()

    # --- Assign physical groups ---
    gmsh.model.addPhysicalGroup(3, [vol_tag_eyeball], eyeball_label)
    gmsh.model.addPhysicalGroup(3, [vol_tag_muscle], muscle_label)

    # --- Mesh parameters ---
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 20)

    # --- Generate tetrahedral mesh ---
    gmsh.model.mesh.generate(3)

    # --- Write mesh ---
    gmsh.write(output_file)
    gmsh.finalize()
    print(f"Tetrahedral mesh written to {output_file}")