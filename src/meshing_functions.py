import vtk 
from vtk.util import numpy_support
import numpy
import gmsh
import meshio
from scipy.spatial import cKDTree
from collections import defaultdict


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
        thickness = 0.8  # mm
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
        offset = None,
        surface_angle=70):

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
    gmsh.write(output_file + ".vtk")
    gmsh.write(output_file + ".stl")

    gmsh.finalize()
    print("Tetra mesh written to:", output_file)









def tetra_shell_from_two_surfaces(
        outer_stl,
        inner_stl,
        output_file,
        element_size=2.0,
        offset = None,
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

    gmsh.write(output_file +".vtk")
    gmsh.write(output_file +".stl")

    gmsh.finalize()

    print("Shell tetra mesh written:", output_file)








def repair_stl(input_stl, output_stl):
    """Repair STL: remove duplicates, fill holes"""
    reader = vtk.vtkSTLReader()
    reader.SetFileName(input_stl)
    reader.Update()
    poly = reader.GetOutput()

    # Clean duplicate points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.Update()
    poly = cleaner.GetOutput()

    # Fill holes
    filler = vtk.vtkFillHolesFilter()
    filler.SetInputData(poly)
    filler.SetHoleSize(1e6)
    filler.Update()
    poly = filler.GetOutput()

    # Write repaired STL
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(output_stl)
    writer.Write()
    return output_stl

def snap_stl_to_shell(shell_stl, obj_stl, output_stl, tolerance=1.0):
    """Repair and snap object STL to shell surface"""
    # Repair object
    repair_stl(obj_stl, obj_stl)

    # Read object
    reader = vtk.vtkSTLReader()
    reader.SetFileName(obj_stl)
    reader.Update()
    poly = reader.GetOutput()

    # Read shell
    shell_reader = vtk.vtkSTLReader()
    shell_reader.SetFileName(shell_stl)
    shell_reader.Update()
    shell_poly = shell_reader.GetOutput()

    # Snap points
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(shell_poly)
    locator.BuildLocator()
    points = poly.GetPoints()
    for i in range(points.GetNumberOfPoints()):
        p = points.GetPoint(i)
        closest_id = locator.FindClosestPoint(p)
        closest_p = shell_poly.GetPoint(closest_id)
        # Respect tolerance
        dist2 = (p[0]-closest_p[0])**2 + (p[1]-closest_p[1])**2 + (p[2]-closest_p[2])**2
        if dist2 <= tolerance**2:
            points.SetPoint(i, closest_p)
    poly.SetPoints(points)

    # Write snapped STL
    writer = vtk.vtkSTLWriter()
    writer.SetInputData(poly)
    writer.SetFileName(output_stl)
    writer.Write()
    return output_stl


def snap_to_shell_vol(shell_vtk, muscle_vtk, output_vtk, tolerance=0.2):
    """
    Snap muscle tetra mesh nodes to the shell tetra mesh surface, keeping tetra volume.

    Parameters
    ----------
    shell_vtk : str
        VTK file of shell tetra mesh
    muscle_vtk : str
        VTK file of muscle tetra mesh
    output_vtk : str
        Path to save snapped muscle tetra mesh
    tolerance : float
        Maximum distance to snap nodes (in same units as mesh)
    """



     # --- Read meshes ---
    shell_mesh = meshio.read(shell_vtk)
    muscle_mesh = meshio.read(muscle_vtk)
    tets = muscle_mesh.cells_dict["tetra"]

    surface_faces = extract_surface_faces(tets)

    def extract_surface_nodes(tets):
        face_count = defaultdict(int)
        for tet in tets:
            faces = [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]])),
            ]
            for f in faces:
                face_count[f] += 1

        surface_nodes = set()
        for f, count in face_count.items():
            if count == 1:
                surface_nodes.update(f)

        return numpy.array(list(surface_nodes))

    shell_surface_idx = extract_surface_nodes(
        shell_mesh.cells_dict["tetra"]
    )
    muscle_surface_idx = extract_surface_nodes(
        muscle_mesh.cells_dict["tetra"]
    )

    shell_points = shell_mesh.points
    muscle_points = muscle_mesh.points.copy()

    shell_surface_coords = shell_points[shell_surface_idx]

    # -------------------------
    # KDTree for nearest shell surface node
    # -------------------------
    tree = cKDTree(shell_surface_coords)

    snapped_flag_muscle = numpy.zeros(len(muscle_points), dtype=int)
    snapped_flag_shell = numpy.zeros(len(shell_points), dtype=int)
    shell_node_used = numpy.zeros(len(shell_points), dtype=bool)

    # Map surface index to global shell index
    surface_index_map = {
        i: shell_surface_idx[i] for i in range(len(shell_surface_idx))
    }

    # -------------------------
    # Snap only muscle surface nodes
    # -------------------------
    for i in muscle_surface_idx:
        p = muscle_points[i]
        dist, local_idx = tree.query(p)

        if dist <= tolerance:

            global_shell_idx = surface_index_map[local_idx]

            # only snap if shell node not already used
            if not shell_node_used[global_shell_idx]:

                muscle_points[i] = shell_points[global_shell_idx]

                snapped_flag_muscle[i] = 1
                snapped_flag_shell[global_shell_idx] = 1

                shell_node_used[global_shell_idx] = True

    # -------------------------
    # Write updated muscle mesh
    # -------------------------
    new_muscle = meshio.Mesh(
        points=muscle_points,
        cells=muscle_mesh.cells,
        point_data={
            **(muscle_mesh.point_data or {}),
            "snapped": snapped_flag_muscle
        },
        cell_data=muscle_mesh.cell_data
    )

    # meshio.write(output_vtk , new_muscle)
    meshio.write(output_vtk[:-4] + ".stl", new_muscle )


    # -------------------------
    # Write updated shell mesh
    # -------------------------
    # Get previous snapped_to if it exists
    if shell_mesh.point_data and "snapped_to" in shell_mesh.point_data:
        previous_flag = shell_mesh.point_data["snapped_to"]
    else:
        previous_flag = numpy.zeros(len(shell_points), dtype=int)

    # Accumulate (logical OR)
    updated_flag = numpy.maximum(previous_flag, snapped_flag_shell)

    new_shell = meshio.Mesh(
        points=shell_points,
        cells=shell_mesh.cells,
        point_data={
            **(shell_mesh.point_data or {}),
            "snapped_to": updated_flag
        },
        cell_data=shell_mesh.cell_data
    )


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # vols = tet_volumes(muscle_points, muscle_mesh.cells_dict["tetra"])
    # print("     Min volume:", vols.min(), "Max volume:", vols.max())

    # neg_ids = numpy.where(vols <= 0)[0]
    # print("     Number of negative volume tets:", len(neg_ids))

    # for i in neg_ids[:30]:   # print first 20
    #     tet = tets[i]

    #     print("     \nTet index:", i)
    #     print("     Node indices:", tet)
    #     print("     Volume:", vols[i])

    #     for n in tet:
    #         print("     node", n, "coord:", muscle_points[n])
    # print()
    # # # # # # # # # # # # # # # # # # #
    tets = muscle_mesh.cells_dict["tetra"]

    surface_faces = extract_surface_faces(tets)

    bad_edges = find_open_surface_edges(
        muscle_points,
        surface_faces
    )

    print("     problem edges:", len(bad_edges))
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    print("     Muscle snapped nodes:", snapped_flag_muscle.sum())
    print("     Shell contacted nodes:", snapped_flag_shell.sum())


def export_centerline(points_np, spacing, filename, offset):

    points_np = numpy.array(points_np)
    spacing = numpy.array(spacing)
    offset = numpy.array(offset)


    transformed_points = (points_np * spacing) - offset

    vtk_points = vtk.vtkPoints()

    for p in transformed_points:
        # p is now a simple 1D array of 3 elements [x, y, z]
        vtk_points.InsertNextPoint(p[0], p[1], p[2])

    # for p in points_np:
        # vtk_points.InsertNextPoint(float(p[0]*spacing - offset[0]), float(p[1]*spacing - offset[1]), float(p[2]*spacing - offset[2]))

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(points_np))

    for i in range(len(points_np)):
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def export_mask(mask, filename):

    vtk_data = numpy_support.numpy_to_vtk(
        num_array=mask.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR
    )

    image = vtk.vtkImageData()
    image.SetDimensions(mask.shape[::-1])
    image.GetPointData().SetScalars(vtk_data)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image)
    writer.Write()



def tet_volumes(points, tets):
    # Compute signed volume of each tet
    p0 = points[tets[:,0]]
    p1 = points[tets[:,1]]
    p2 = points[tets[:,2]]
    p3 = points[tets[:,3]]

    v = numpy.einsum('ij,ij->i', numpy.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
    return v


def find_problem_faces(points, tets):

    face_count = defaultdict(int)

    for tet in tets:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]

        for f in faces:
            face_count[f] += 1

    face_vals = numpy.array(list(face_count.values()))

    print("     faces belonging to 1 tet (boundary):", numpy.sum(face_vals == 1))
    print("     faces belonging to 2 tets (internal):", numpy.sum(face_vals == 2))
    print("     faces belonging to >2 tets (non-manifold):", numpy.sum(face_vals > 2))

    bad_faces = [f for f,c in face_count.items() if c > 2]

    return bad_faces


def extract_surface_faces(tets):

    face_count = defaultdict(int)

    for tet in tets:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]

        for f in faces:
            face_count[f] += 1

    surface_faces = [f for f,c in face_count.items() if c == 1]

    return numpy.array(surface_faces)


def find_open_surface_edges(points, surface_faces):

    edge_count = defaultdict(int)

    for tri in surface_faces:

        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]

        for e in edges:
            edge_count[e] += 1

    vals = numpy.array(list(edge_count.values()))

    # print("     edges belonging to 1 triangle (holes):", numpy.sum(vals == 1))
    # print("     edges belonging to 2 triangles (correct):", numpy.sum(vals == 2))
    # print("     edges belonging to >2 triangles (non-manifold):", numpy.sum(vals > 2))

    bad_edges = [e for e,c in edge_count.items() if c != 2]

    return bad_edges


def set_the_offset(file, offset):

    mesh = meshio.read(file + ".vtk")

    new_mesh = meshio.Mesh(
        points=mesh.points - offset,
        cells=mesh.cells
        )

    meshio.write(file + ".vtk" , new_mesh)
    meshio.write(file + ".stl" , new_mesh)



def mesh_2d_from_stl(stl_file, output_file, element_size=2.0, surface_angle=40):

    gmsh.initialize()
    gmsh.model.add("model_2D")

    # --------------------------------------------------
    # 1. Import STL
    # --------------------------------------------------
    gmsh.merge(stl_file)

    # --------------------------------------------------
    # 2. Reconstruct geometry
    # --------------------------------------------------
    angle = surface_angle * numpy.pi / 180.0

    gmsh.model.mesh.classifySurfaces(angle, True, True)
    gmsh.model.mesh.createGeometry()
    gmsh.model.geo.synchronize()

    # --------------------------------------------------
    # 3. Get surfaces
    # --------------------------------------------------
    surfaces = gmsh.model.getEntities(2)

    if len(surfaces) == 0:
        raise RuntimeError("No surfaces found.")

    print("Surfaces found:", surfaces)

    # 👉 Keep ONLY the largest surface (important!)
    def surface_size(tag):
        return gmsh.model.getBoundingBox(2, tag)

    # crude but effective: pick first (usually fine)
    surface_tag = surfaces[0][1]

    # --------------------------------------------------
    # 4. Extract ORDERED boundary
    # --------------------------------------------------
    boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=True)

    # IMPORTANT: keep orientation!
    curve_loop = [c[1] if c[0] > 0 else -c[1] for c in boundary]

    # --------------------------------------------------
    # 5. Create clean surface
    # --------------------------------------------------
    cl = gmsh.model.geo.addCurveLoop(curve_loop)
    ps = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # --------------------------------------------------
    # 6. Remove original STL surfaces (CRITICAL FIX)
    # --------------------------------------------------
    for s in surfaces:
        gmsh.model.removeEntities([s], recursive=True)

    gmsh.model.geo.synchronize()

    # --------------------------------------------------
    # Physical groups
    # --------------------------------------------------

    # Surface (triangles)
    gmsh.model.addPhysicalGroup(2, [ps], tag=100)
    gmsh.model.setPhysicalName(2, 100, "Domain")

    # Boundary (edges)
    curve_tags_abs = [abs(c) for c in curve_loop]

    gmsh.model.addPhysicalGroup(1, curve_tags_abs, tag=200)
    gmsh.model.setPhysicalName(1, 200, "Boundary")

    # --------------------------------------------------
    # 7. Mesh size
    # --------------------------------------------------
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)

    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # --------------------------------------------------
    # 8. Mesh settings
    # --------------------------------------------------    
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 20)

    # --------------------------------------------------
    # 9. Generate mesh
    # --------------------------------------------------
    gmsh.model.mesh.generate(2)


    types, elementTags, nodeTags = gmsh.model.mesh.getElements(2)

    print("Element types:", types)
    # --------------------------------------------------
    # 10. Export
    # --------------------------------------------------
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    gmsh.write(output_file + ".vtk")
    gmsh.write(output_file + ".msh")
    gmsh.write(output_file + ".stl")

    gmsh.finalize()

    print("Fixed 2D mesh written to:", output_file)