import nibabel as nib
import numpy as numpy
import vtk
import tomllib
from vtk.util import numpy_support
from vtk.util.numpy_support import vtk_to_numpy

import os


def load_nifti(input_path, write = False):
    """
    Crop a 3D NIfTI image and save it with updated affine.

    Parameters:
        input_path (str): input .nii/.nii.gz
        output_path (str): output file
        x_range, y_range, z_range: tuples (start, end)
    """

    nii = nib.load(input_path)
    data = numpy.array(nii.dataobj)  # efficient loading
    affine = nii.affine.copy()
    header = nii.header.copy()
    offset = affine[:3, 3] 

    pixel_spacing = header.get_zooms()[:3] 

    if write:
        spacing_file = input_path[:-4] + "_pixel_spacing.txt"

        data_offset_file = input_path[:-4] + "_offset.txt"
        
        with open(spacing_file, 'w') as f:
            f.write(f"{pixel_spacing[0]} {pixel_spacing[1]} {pixel_spacing[2]}\n")
        print(f"Original pixel spacing saved to: {spacing_file} -> {pixel_spacing}")

        with open(data_offset_file, 'w') as f:
            f.write(f"{offset[0]} {offset[1]} {offset[2]}\n")
        print(f"Original pixel spacing saved to: {data_offset_file} -> {affine}")

    print("Original matrix shape = ", data.shape)

    return data, pixel_spacing, offset, affine


def write_vti(data, pixel_spacing, offset, output_file):
    data = numpy.asarray(data)

    # Older readers can choke on bool or 64-bit integer arrays
    if data.dtype == bool:
        data = data.astype(numpy.uint8)
    elif data.dtype == numpy.int64:
        data = data.astype(numpy.int32)

    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(data.shape)
    vtk_image.SetSpacing(pixel_spacing)
    vtk_image.SetOrigin(offset[0], offset[1], offset[2])

    # Flatten in Fortran order to match VTK's x-fastest convention
    flat_data = numpy.ascontiguousarray(data.flatten(order="F"))
    vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True)
    vtk_array.SetName("intensity")
    vtk_image.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file)       # should end in .vti
    writer.SetInputData(vtk_image)
    writer.SetDataModeToBinary()          # inline base64 instead of appended raw data
    writer.SetCompressorTypeToNone()      # no zlib compression
    writer.SetHeaderTypeToUInt32()        # 32-bit headers for older readers
    writer.Write()
    print(f"Saved VTI file to: {output_file}")


def export_prep(data):
    if data.dtype == bool:
        data = data.astype(numpy.uint8)
    elif data.dtype == numpy.int64:
        data = data.astype(numpy.int32)
    elif data.dtype == numpy.float64:
        data = data.astype(numpy.float32)      # GPU-friendly

    return data


def load_vti(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    img = reader.GetOutput()

    nx, ny, nz = img.GetDimensions()
    scalars = img.GetPointData().GetScalars() or img.GetPointData().GetArray(0)
    flat = vtk_to_numpy(scalars)

    data = flat.reshape(nz, ny, nx, -1).transpose(2, 1, 0, 3)
    if data.shape[-1] == 1:
        data = data[..., 0]          # single component -> plain 3D array

    pixel_spacing = numpy.array(img.GetSpacing())
    origin = numpy.array(img.GetOrigin())
    return data, pixel_spacing, origin