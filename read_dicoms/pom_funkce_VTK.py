import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os, glob
import numpy as numpy
import pydicom, pyvista
import myVTKPythonLibrary as myvtk
import myPythonLibrary    as mypy


from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)


def generate_image_grid_exp(dicom_SA_folder, out_folder, T, o, z_spacing_new):
    print("Generating grid for the image")

    fname = dicom_SA_folder + "time_00_slice_01.dcm"
    slice_SA = pydicom.dcmread(fname)
    n_times = slice_SA[0x2001,0x1017].value
    dcm_counter = len(glob.glob1(dicom_SA_folder,"*.dcm"))
    n_slices_SA = int(dcm_counter/n_times)

    ps_SA = slice_SA.PixelSpacing
    ss_SA = slice_SA.SpacingBetweenSlices

    FOV_SA = [int(slice_SA.pixel_array.shape[0]*ps_SA[0]), int(slice_SA.pixel_array.shape[1]*ps_SA[1]), int(n_slices_SA*ss_SA)]
    print("  initial FOV SA = ", FOV_SA)

    FOV_SA[2] = int(numpy.ceil(FOV_SA[2]/z_spacing_new)*z_spacing_new)

    print("  final FOV SA = ", FOV_SA)

    offset_SA  = slice_SA.ImagePositionPatient # (0020, 0032)
    vectors_SA = slice_SA.ImageOrientationPatient # (0020, 0037)

    ps = slice_SA.PixelSpacing
    ss = slice_SA.SpacingBetweenSlices
    st = slice_SA.SliceThickness

    spacing = numpy.zeros(3)
    spacing[0:2] = ps
    spacing[2]=z_spacing_new

    vect_row_SA = vectors_SA[0:3]
    vect_col_SA = vectors_SA[3:7]
    vect_z_SA = numpy.cross(vect_row_SA, vect_col_SA)
    vect_z_SA = vect_z_SA/ numpy.linalg.norm(vect_z_SA)

    matrix_SA = numpy.zeros((3,3))
    matrix_SA[1,:] = vect_row_SA
    matrix_SA[0,:] = vect_col_SA
    matrix_SA[2,:] = - vect_z_SA
    T_SA_inv = numpy.linalg.inv(matrix_SA)

    offset_new = numpy.zeros(3)

    offset_new[0] = offset_SA[0] - 0.5*spacing[0] +0.5
    offset_new[1] = offset_SA[1] + 0.5*spacing[1] -0.5
    offset_new[2] = offset_SA[2] + 0.5*spacing[2] -0.5

    x_coord = numpy.arange(0,FOV_SA[0])
    y_coord = numpy.arange(0,FOV_SA[1])
    z_coord = numpy.arange(0,FOV_SA[2])

    coords = [numpy.array((x0, y0, z0)) for x0 in x_coord for y0 in y_coord for z0 in z_coord]

    #mesh = pyvista.PolyData()
    #vertices = numpy.array(coords)
    #mesh = pyvista.PolyData(vertices)
    #myvtk.writePData(mesh, out_folder +"SA_grid_init.vtk", 1)

    coords = [ T.dot((T_SA_inv.dot(c) + offset_new) - o) for c in coords   ]

    vertices = numpy.array(coords)
    mesh = pyvista.PolyData(vertices)
    myvtk.writePData(mesh, out_folder +"SA_grid.vtk", 1)
    print("  Grid created")

    numpy.savetxt(out_folder + "FOV_SA.txt",  FOV_SA)
    numpy.savetxt(out_folder + "matrix.txt",  matrix_SA)
    numpy.savetxt(out_folder + "offset.txt",  offset_SA)
    numpy.savetxt(out_folder + "n_times.txt",  [n_times])
    numpy.savetxt(out_folder + "offset_fine.txt",  offset_new)
    numpy.savetxt(out_folder + "spacing.txt",  spacing)


    return mesh, FOV_SA


def generate_image_grid(dicom_SA_folder, out_folder, T, o):
    print("Generating grid for the image")

    fname = dicom_SA_folder + "time_00_slice_01.dcm"
    slice_SA = pydicom.dcmread(fname)
    n_times = slice_SA[0x2001,0x1017].value
    dcm_counter = len(glob.glob1(dicom_SA_folder,"*.dcm"))
    n_slices_SA = int(dcm_counter/n_times)

    ps_SA = slice_SA.PixelSpacing
    ss_SA = slice_SA.SpacingBetweenSlices

    FOV_SA = [int(slice_SA.pixel_array.shape[0]*ps_SA[0]), int(slice_SA.pixel_array.shape[1]*ps_SA[1]), int(n_slices_SA*ss_SA)]

    print("  FOV SA = ", FOV_SA)

    offset_SA  = slice_SA.ImagePositionPatient # (0020, 0032)
    vectors_SA = slice_SA.ImageOrientationPatient # (0020, 0037)

    ps = slice_SA.PixelSpacing
    ss = slice_SA.SpacingBetweenSlices
    st = slice_SA.SliceThickness

    spacing = numpy.zeros(3)
    spacing[0:2] = ps
    spacing[2]=ss

    vect_row_SA = vectors_SA[0:3]
    vect_col_SA = vectors_SA[3:7]
    vect_z_SA = numpy.cross(vect_row_SA, vect_col_SA)
    vect_z_SA = vect_z_SA/ numpy.linalg.norm(vect_z_SA)

    matrix_SA = numpy.zeros((3,3))
    matrix_SA[1,:] = vect_row_SA
    matrix_SA[0,:] = vect_col_SA
    matrix_SA[2,:] = - vect_z_SA
    T_SA_inv = numpy.linalg.inv(matrix_SA)

    offset_new = numpy.zeros(3)

    offset_new[0] = offset_SA[0] - 0.5*spacing[0] +0.5
    offset_new[1] = offset_SA[1] + 0.5*spacing[1] -0.5
    offset_new[2] = offset_SA[2] + 0.5*spacing[2] -0.5

    x_coord = numpy.arange(0,FOV_SA[0])
    y_coord = numpy.arange(0,FOV_SA[1])
    z_coord = numpy.arange(0,FOV_SA[2])

    coords = [numpy.array((x0, y0, z0)) for x0 in x_coord for y0 in y_coord for z0 in z_coord]

    #mesh = pyvista.PolyData()
    #vertices = numpy.array(coords)
    #mesh = pyvista.PolyData(vertices)
    #myvtk.writePData(mesh, out_folder +"SA_grid_init.vtk", 1)

    coords = [ T.dot((T_SA_inv.dot(c) + offset_new) - o) for c in coords   ]

    vertices = numpy.array(coords)
    mesh = pyvista.PolyData(vertices)
    myvtk.writePData(mesh, out_folder +"SA_grid.vtk", 1)
    print("  Grid created")

    numpy.savetxt(out_folder + "FOV_SA.txt",  FOV_SA)
    numpy.savetxt(out_folder + "matrix.txt",  matrix_SA)
    numpy.savetxt(out_folder + "offset.txt",  offset_SA)
    numpy.savetxt(out_folder + "n_times.txt",  [n_times])
    numpy.savetxt(out_folder + "offset_fine.txt",  offset_new)
    numpy.savetxt(out_folder + "spacing.txt",  spacing)


    return mesh, FOV_SA





def numpy2VTK(d, data_type=None):
    """
    Transform a 3d numpy array into a vtk image data object

    Args:
        d (numpy.ndarray) : Voxel data
        data_type (str) : Data type of voxel data, if None attempts to read from the array

    Returns:
        vtkImageData
    """
    array_data_type = d.dtype

    if data_type is not None:
        assert array_data_type == data_type

    importer = vtk.vtkImageImport()
    assert isinstance(d, numpy.ndarray)
    importer.SetDataScalarTypeToShort()  # default
    if array_data_type is None:
        array_data_type = d.type
    if array_data_type == numpy.float64:
        importer.SetDataScalarTypeToDouble()
    elif array_data_type == numpy.float32:
        importer.SetDataScalarTypeToFloat()
    elif array_data_type == numpy.int32:
        importer.SetDataScalarTypeToInt()
    elif array_data_type == numpy.int16:
        importer.SetDataScalarTypeToShort()
    elif array_data_type == numpy.uint8:
        importer.SetDataScalarTypeToUnsignedChar()
    else:
        log = logging.getLogger(__name__)
        log.warning("casting to float64")
        importer.SetDataScalarTypeToDouble()
        d = d.astype(numpy.float64)
        #======================================
    dstring = d.flatten(order='F').tostring()
    if array_data_type.byteorder == '>':
        # Fix byte order
        dflat_l = d.flatten(order='F').tolist()
        format_string = '<%id' % len(dflat_l)
        dstring = struct.pack(format_string, *dflat_l)
        # importer.SetDataScalarTypeToInt()
    importer.SetNumberOfScalarComponents(1)
    importer.CopyImportVoidPointer(dstring, len(dstring))
    dshape = d.shape

    if(len(dshape)>=3):
        importer.SetDataExtent(0, dshape[0] - 1, 0, dshape[1] - 1, 0, dshape[2] - 1)
        importer.SetWholeExtent(0, dshape[0] - 1, 0, dshape[1] - 1, 0, dshape[2] - 1)
    else:
        importer.SetDataExtent(0, dshape[0] - 1, 0, dshape[1] - 1, 0, 0)
        importer.SetWholeExtent(0, dshape[0] - 1, 0, dshape[1] - 1, 0, 0)

    importer.Update()
    imgData = importer.GetOutput()
    # return imgData
    out_img = vtk.vtkImageData()
    out_img.DeepCopy(imgData)
    return out_img


def read_VTI(name):
    #print("loading : ", name)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(name)
    reader.Update()
    image = reader.GetOutput()

    spacing = image.GetSpacing()

    dim1, dim2, dim3 = image.GetDimensions()

    sc = image.GetPointData().GetScalars()
    a = vtk_to_numpy(sc)
    a = a.reshape(dim1, dim2, dim3, order='F')

    return a, spacing

def write_centerline_with_data(all_points, all_radius, all_dist, all_integrals, all_valid, name, spacing):

    start = all_points[0][0]

    vtk_points = vtkPoints()
    cells = vtkCellArray()
    # Create a polydata to store everything in
    polyData = vtkPolyData()

    n_points = int(sum([len(i) for i in all_points]))
    #print("        n points = ", n_points)

    array_rad = vtk.vtkDoubleArray()
    array_rad.SetName("Radius")
    array_rad.SetNumberOfValues(n_points)
    R = sum(all_radius, [])
    #print("        len R = ", len(R))

    for x in zip(range(n_points), R):
        array_rad.SetValue(*x)

    array_id = vtk.vtkDoubleArray()
    array_id.SetName("SectionID")
    array_id.SetNumberOfValues(n_points)

    ID = []
    for j in range(len(all_points)):
        for i in range(len(all_points[j])):
            ID.append(j)
    for x in zip(range(n_points), ID):
        array_id.SetValue(*x)

    array_dist = vtk.vtkDoubleArray()
    array_dist.SetName("Distance")
    array_dist.SetNumberOfValues(n_points)

    D = sum(all_dist, [])
    #print("        len D = ", len(D))


    for x in zip(range(n_points), D):
        array_dist.SetValue(*x)

    F = []
    array_flux = vtk.vtkDoubleArray()
    array_flux.SetName("Flux")
    array_flux.SetNumberOfValues(n_points)

    k_prev = 0

    for j in range(len(all_points)):
        print("j : points = ", len(all_points[j]), "  val = ", len(all_valid[j]), "  val sum = ", sum(all_valid[j]) , "  int = ", len(all_integrals[j]))
        print(all_valid[j])
        print()

        for i in range(len(all_points[j])):
            print("   i = ", i)

            if all_valid[j][i]==1:
                k = int(sum(all_valid[j][0:i+1]))
                #print("k-1 = ", k-1)
                F.append(all_integrals[j][k-1]/100)
                k_prev = k
            else:
                F.append(all_integrals[j][k_prev-1]/100)

    for x in zip(range(n_points), F):
        array_flux.SetValue(*x)


    counter = 0
    for j in range(len(all_points)):

        for y in all_points[j]:
            vtk_points.InsertPoint(counter, y[0]*spacing[0],y[1]*spacing[1],y[2]*spacing[2])
            counter = counter +1

    prev = 0
    for j in range(len(all_points)):

        line = vtk.vtkIdList()
        for i in range(len(all_points[j])):
            line.InsertNextId(prev+i)

        prev = prev+ len(all_points[j])
        c_ind = cells.InsertNextCell(line)


    dataset = polyData.GetPointData()
    dataset.AddArray(array_rad)
    dataset.AddArray(array_id)
    dataset.AddArray(array_dist)
    dataset.AddArray(array_flux)


    polyData.SetLines(cells)

    polyData.SetPoints(vtk_points)
    writePolyData(polyData, name)

def write_line(points, radius, dist, name, spacing):
    vtk_points = vtkPoints()
    n_points = 0

    R = radius
    D = dist

    for y in points:
        vtk_points.InsertNextPoint([y[0]*spacing[0],y[1]*spacing[1],y[2]*spacing[2]])
        n_points = n_points +1


    print("  n centerline points = ", n_points)
    polyLine = vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(n_points)
    for i in range(n_points):
            polyLine.GetPointIds().SetId(i, i)

    # Create a cell array to store the lines in and add the lines to it
    cells = vtkCellArray()
    cells.InsertNextCell(polyLine)

    # Create a polydata to store everything in
    polyData = vtkPolyData()

    # Add the points to the dataset
    polyData.SetPoints(vtk_points)

    array_rad = vtk.vtkDoubleArray()
    array_rad.SetName("Radius")
    array_rad.SetNumberOfValues(n_points)
    for x in zip(range(n_points), R):
        array_rad.SetValue(*x)

    array_dist = vtk.vtkDoubleArray()
    array_dist.SetName("Distance")
    array_dist.SetNumberOfValues(n_points)
    for x in zip(range(n_points), D):
        array_dist.SetValue(*x)

    dataset = polyData.GetPointData()
    dataset.AddArray(array_rad)
    dataset.AddArray(array_dist)


    # Add the lines to the dataset
    polyData.SetLines(cells)

    writePolyData(polyData, name)




def compute_upsampled_image(
        filename,
        downsampling_factors,
        write_temp_images=0,
        suffix="",
        verbose=0):


    image = myvtk.readImage(filename,0)
    images_ndim = myvtk.getImageDimensionality(
        image=image,
        verbose=0)
    mypy.my_print(verbose, " images_ndim = "+str(images_ndim))
    images_dimensions = image.GetDimensions()
    mypy.my_print(verbose, " images_dimensions = "+str(images_dimensions))
    images_npoints = numpy.prod(images_dimensions)
    mypy.my_print(verbose, " images_npoints = "+str(images_npoints))
    images_origin = image.GetOrigin()
    mypy.my_print(verbose, "images_origin = "+str(images_origin))
    images_spacing = image.GetSpacing()
    mypy.my_print(verbose, " images_spacing = "+str(images_spacing))

    mypy.my_print(verbose, " sampling_factors = "+str(downsampling_factors))
    downsampling_factors = downsampling_factors+[1]*(3-images_ndim)
    mypy.my_print(verbose, " sampling_factors = "+str(downsampling_factors))

    images_new_dimensions = numpy.divide(images_dimensions, downsampling_factors)
    images_new_dimensions = numpy.ceil(images_new_dimensions)
    images_new_dimensions = [int(n) for n in images_new_dimensions]
    mypy.my_print(verbose, " images_new_dimensions = "+str(images_new_dimensions))
    mypy.my_print(verbose, "________________________________________")


    reader_type = vtk.vtkXMLImageDataReader
    writer_type = vtk.vtkXMLImageDataWriter

    reader = reader_type()
    reader.UpdateDataObject()

    fft = vtk.vtkImageFFT()
    fft.SetDimensionality(images_ndim)
    fft.SetInputData(reader.GetOutput())
    fft.UpdateDataObject()


    images_downsampled_npoints = numpy.prod(images_new_dimensions)
    mypy.my_print(verbose, " images_new_npoints = "+str(images_new_dimensions))
    downsampling_factors = list(numpy.divide(images_dimensions, images_new_dimensions))
    mypy.my_print(verbose, " sampling_factors = "+str(downsampling_factors))
    downsampling_factor = numpy.prod(downsampling_factors)
    mypy.my_print(verbose, " product sampling_factor = "+str(downsampling_factor))
    images_downsampled_origin = images_origin
    mypy.my_print(verbose, " images_downsampled_origin = "+str(images_downsampled_origin))
    images_downsampled_spacing = list(numpy.multiply(images_spacing, downsampling_factors))
    mypy.my_print(verbose, " images_downsampled_spacing = "+str(images_downsampled_spacing))
    mypy.my_print(verbose, "________________________________________")


    image_new = vtk.vtkImageData()                              # create container for new image
    image_new.SetDimensions(images_new_dimensions)
    image_new.SetOrigin(images_downsampled_origin)
    image_new.SetSpacing(images_downsampled_spacing)

    image_new_scalars = myvtk.createDoubleArray(                # create array for values
        name="ImageScalars",
        n_components=2,
        n_tuples=images_downsampled_npoints,
        verbose=0)
    image_new.GetPointData().SetScalars(image_new_scalars)
    I = numpy.empty(2)                                                  # container for interpolated intensity

    rfft = vtk.vtkImageRFFT()
    rfft.SetDimensionality(images_ndim)
    rfft.SetInputData(image_new)
    rfft.UpdateDataObject()

    extract = vtk.vtkImageExtractComponents()
    extract.SetInputData(rfft.GetOutput())
    extract.SetComponents(0)
    extract.UpdateDataObject()

    writer = writer_type()
    writer.SetInputData(extract.GetOutput())


    ##################### do the thing: ####################

    reader.SetFileName(filename)        # read input (coarse) image
    reader.Update()
    fft.Update()                        # apply FFT to the input image


    image_scalars = fft.GetOutput().GetPointData().GetScalars()                 # get values at the FFT points
    image_new_scalars = image_new.GetPointData().GetScalars()
    for k_z_downsampled in range(images_dimensions[2]):
        k_z = k_z_downsampled if (k_z_downsampled <= images_dimensions[2]//2) else k_z_downsampled+(images_new_dimensions[2]-images_dimensions[2])
        for k_y_downsampled in range(images_dimensions[1]):
            k_y = k_y_downsampled if (k_y_downsampled <= images_dimensions[1]//2) else k_y_downsampled+(images_new_dimensions[1]-images_dimensions[1])
            for k_x_downsampled in range(images_dimensions[0]):
                k_x = k_x_downsampled if (k_x_downsampled <= images_dimensions[0]//2) else k_x_downsampled+(images_new_dimensions[0]-images_dimensions[0])
                k_point_downsampled = k_z_downsampled*images_dimensions[1]*images_dimensions[0] + k_y_downsampled*images_dimensions[0] + k_x_downsampled
                k_point             = k_z            *images_new_dimensions[1]            *images_new_dimensions[0]             + k_y            *images_new_dimensions[0]             + k_x

                image_scalars.GetTuple(k_point_downsampled, I)
                I /= downsampling_factor
                image_new_scalars.SetTuple(k_point, I)
    image_new.Modified()


    rfft.Update()

    extract.Update()

    writer.SetFileName(filename)
    writer.Write()
