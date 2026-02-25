import vtk 
from vtk.util import numpy_support
import numpy
import gmsh

def getTetraMesh(stack, filename):
    """
    Generates tetrahedral volume mesh from a 3D binary mask.
    
    stack: 3D numpy array (z,y,x or any ordering — assumed consistent)
    """

    d0, d1, d2 = stack.shape

    # ---------------------------
    # Create vtkImageData
    # ---------------------------
    img = vtk.vtkImageData()
    img.SetDimensions(d0, d1, d2)
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)

    # Fast scalar assignment (correct ordering)
    flat = stack.ravel(order='F')  # VTK expects Fortran order
    vtk_data = numpy_support.numpy_to_vtk(
        flat.astype(numpy.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    img.GetPointData().SetScalars(vtk_data)

    # ---------------------------
    # Extract surface
    # ---------------------------
    snets = vtk.vtkSurfaceNets3D()
    snets.SetInputData(img)
    snets.SetOutputMeshTypeToTriangles()
    snets.Update()

    surface = snets.GetOutput()


    # ---------------------------
    # Smoothen surface (optional)
    # ---------------------------

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(30)
    smoother.SetRelaxationFactor(0.1)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()

    # ---------------------------
    # Clean surface 
    # ---------------------------
    cleaner = vtk.vtkCleanPolyData()

    # if not smoothing
    # cleaner.SetInputData(surface)
    cleaner.SetInputData(smoother.GetOutput())

    cleaner.Update()

    # ---------------------------
    # Generate tetrahedral mesh
    # ---------------------------
    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(cleaner.GetOutput())
    delaunay.SetTolerance(0.01)
    delaunay.Update()

    tetra_mesh = delaunay.GetOutput()

    # ---------------------------
    # Write to VTU (recommended)
    # ---------------------------
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(tetra_mesh)
    writer.Write()

    return tetra_mesh



def getSurfaceMesh(stack, filename, voxel_size, shell_bool,
                   smoothing_iterations=20,
                   pass_band=0.1):
    """
    Generate smoothed surface mesh from 3D binary mask
    and export as STL for Gmsh.
    """

    d0, d1, d2 = stack.shape

    # -------------------------
    # Create vtkImageData
    # -------------------------
    img = vtk.vtkImageData()
    img.SetDimensions(d0, d1, d2)
    img.SetSpacing(voxel_size, voxel_size, voxel_size)
    img.SetOrigin(0.0, 0.0, 0.0)

    # Fast and correct scalar assignment (VTK uses Fortran order)
    flat = stack.ravel(order='F')
    vtk_data = numpy_support.numpy_to_vtk(
        flat.astype(numpy.uint8),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )
    img.GetPointData().SetScalars(vtk_data)

    # -------------------------
    # Surface extraction
    # -------------------------
    snets = vtk.vtkSurfaceNets3D()
    snets.SetInputData(img)
    snets.SetOutputMeshTypeToTriangles()
    snets.Update()

    surface = snets.GetOutput()

    # Clean duplicate points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface)
    cleaner.Update()

    # Triangles only
    triFilter = vtk.vtkTriangleFilter()
    triFilter.SetInputData(cleaner.GetOutput())
    triFilter.Update()

    # Normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(triFilter.GetOutput())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    # Fill holes
    holeFiller = vtk.vtkFillHolesFilter()
    holeFiller.SetInputData(normals.GetOutput())
    holeFiller.SetHoleSize(1e6)  # large enough to close all small gaps
    holeFiller.Update()

    # Keep only largest connected component
    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(holeFiller.GetOutput())
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()

    repaired_surface = connectivity.GetOutput()

    # Check non-manifold edges
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputData(repaired_surface)
    featureEdges.NonManifoldEdgesOn()
    featureEdges.BoundaryEdgesOn()
    featureEdges.FeatureEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.Update()
    print("Problem edges:", featureEdges.GetOutput().GetNumberOfCells())

    if shell_bool :
        # -----------------------------------
        # Compute point normals on repaired surface
        # -----------------------------------
        normalsFilter = vtk.vtkPolyDataNormals()
        normalsFilter.SetInputData(repaired_surface)
        normalsFilter.ComputePointNormalsOn()
        normalsFilter.SplittingOff()
        normalsFilter.ConsistencyOn()
        normalsFilter.AutoOrientNormalsOn()
        normalsFilter.Update()

        surface_with_normals = normalsFilter.GetOutput()

        # -----------------------------------
        # Create inner offset surface
        # -----------------------------------
        thickness = 1.0  # mm
        points = surface_with_normals.GetPoints()
        normals = surface_with_normals.GetPointData().GetNormals()

        newPoints = vtk.vtkPoints()
        newPoints.SetNumberOfPoints(points.GetNumberOfPoints())

        for i in range(points.GetNumberOfPoints()):
            p = numpy.array(points.GetPoint(i))
            n = numpy.array(normals.GetTuple(i))
            p_new = p - thickness * n  # move inward
            newPoints.SetPoint(i, p_new)

        inner_surface = vtk.vtkPolyData()
        inner_surface.DeepCopy(surface_with_normals)
        inner_surface.SetPoints(newPoints)

        # Write both outer and inner surfaces
        outerWriter = vtk.vtkSTLWriter()
        outerWriter.SetInputData(surface_with_normals)
        outerWriter.SetFileName(filename.replace(".stl", "_outer.stl"))
        outerWriter.Write()

        innerWriter = vtk.vtkSTLWriter()
        innerWriter.SetInputData(inner_surface)
        innerWriter.SetFileName(filename.replace(".stl", "_inner.stl"))
        innerWriter.Write()

        print("Outer and inner surfaces written.")
    else:
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetInputData(repaired_surface)
        stlWriter.SetFileName(filename)
        stlWriter.Write()

        print("written file: ", filename)


def tetra_mesh_from_stl(
        stl_file,
        output_file,
        element_size=2.0,
        surface_angle=40):

    gmsh.initialize()
    gmsh.model.add("model")

    # --------------------------------------------------
    # Import STL
    # --------------------------------------------------
    gmsh.merge(stl_file)

    # --------------------------------------------------
    # Convert STL mesh → geometry
    # --------------------------------------------------
    angle = surface_angle * numpy.pi / 180.0

    gmsh.model.mesh.classifySurfaces(
        angle,
        True,   # include boundary
        True    # force parametrizable patches
    )

    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # --------------------------------------------------
    # Create volume
    # --------------------------------------------------
    surfaces = gmsh.model.getEntities(2)
    if not surfaces:
        raise RuntimeError("No surfaces found after reconstruction.")

    surface_tags = [s[1] for s in surfaces]

    sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # --------------------------------------------------
    # Uniform mesh size (true coarsening)
    # --------------------------------------------------
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # --------------------------------------------------
    # Mesh settings
    # --------------------------------------------------
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 20)

    # --------------------------------------------------
    # Generate volume mesh
    # --------------------------------------------------
    gmsh.model.mesh.generate(3)

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    gmsh.write(output_file)

    gmsh.finalize()
    print("Tetra mesh written to:", output_file)


def tetra_shell_from_two_surfaces(
        outer_stl,
        inner_stl,
        output_file,
        element_size=2.0,
        surface_angle=40):

    gmsh.initialize()
    gmsh.model.add("shell_model")

    # Import surfaces
    gmsh.merge(outer_stl)
    gmsh.merge(inner_stl)

    # Convert STL → geometry
    angle = surface_angle * numpy.pi / 180.0

    gmsh.model.mesh.classifySurfaces(
        angle,
        True,
        True
    )

    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # Create shell volume
    surfaces = gmsh.model.getEntities(2)
    surface_tags = [s[1] for s in surfaces]

    sl = gmsh.model.geo.addSurfaceLoop(surface_tags)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # Mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    gmsh.option.setNumber("Mesh.Algorithm3D", 4)

    gmsh.model.mesh.generate(3)

    gmsh.write(output_file)
    gmsh.finalize()

    print("Shell tetra mesh written:", output_file)