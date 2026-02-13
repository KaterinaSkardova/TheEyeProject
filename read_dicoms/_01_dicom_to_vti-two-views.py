"""Loads dicom files sorted with respect to time and slice
position by a preceding script. Than converts them to several
3D .vti files, one for each time frame. This is done for
one subtype of 4D flow sequence (e.g. THR, inplane1, inplane2,
magnitude). Velicity images are scaled to cm/s, magnitude images
ale left unscaled. XYZ-scanner-space to image-space
transformation matrix and offset are stored to .txt files.
"""

import pydicom
import numpy as np
import glob
import numpy
import os
#import scipy.ndimage       as nd
#from scipy.ndimage import zoom, generic_gradient_magnitude
from pathlib import Path
import myVTKPythonLibrary as myvtk
from pom_funkce_VTK import numpy2VTK
# from scipy.ndimage import zoom
from scipy.ndimage import affine_transform


def rotate_volume_to_match(volume, v1_a, v2_a, v1_b, v2_b, order=1):
    """
    Rotates 3D volume so that frame (v1_a, v2_a) aligns with (v1_b, v2_b).

    Parameters
    ----------
    volume : 3D numpy array
    v1_a, v2_a : edge vectors of volume A
    v1_b, v2_b : edge vectors of target orientation
    order : interpolation order (0=nearest,1=linear,3=cubic)

    Returns
    -------
    rotated_volume : 3D numpy array
    """

    # Build orthonormal frames
    R1 = make_frame(v1_a, v2_a)
    R2 = make_frame(v1_b, v2_b)

    # Rotation matrix mapping A -> B
    R = R2 @ R1.T

    # affine_transform applies the inverse matrix
    R_inv = np.linalg.inv(R)

    # Rotate around center instead of origin
    center = np.array(volume.shape) / 2.0
    offset = center - R_inv @ center

    rotated = affine_transform(
        volume,
        R_inv,
        offset=offset,
        order=order,
        mode='constant',
        cval=0.0
    )

    return rotated

def make_frame(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return np.column_stack((v1, v2, v3))

def read_dicom(src_folder, id):

    listcount = 0
    filename_list = []

    for filename in glob.glob(base + src_folder + os.sep    + "*.DCM", recursive=True):

        f = pydicom.dcmread(filename)
        if hasattr(f,"SliceLocation"):
            filename_list.append(filename)
            slices_list.append(float(f.SliceLocation))
            listcount = listcount + 1
        else:
            skipcount = skipcount + 1

    series_file_list = sorted(filename_list, key = lambda s: pydicom.dcmread(s).SliceLocation)

    f0 = pydicom.dcmread(series_file_list[-1])
    ps = f0.PixelSpacing
    ss = f0.SpacingBetweenSlices
    st = f0.SliceThickness
    bs = f0.BitsStored
    offset = numpy.array(f0.ImagePositionPatient) # (0020, 0032)
    oriantation = f0.ImageOrientationPatient # (0020, 0037)

    vect_row = oriantation[0:3]
    vect_col = oriantation[3:7]
    vect_z = numpy.cross(vect_row, vect_col) / numpy.linalg.norm(numpy.cross(vect_row, vect_col))

    matrix = numpy.zeros((3,3))
    matrix[1,:] = vect_row
    matrix[0,:] = vect_col
    matrix[2,:] = vect_z


    numpy.savetxt(base + out_folder + os.sep + "offset_"+str(id)+".txt",  offset)
    numpy.savetxt(base + out_folder + os.sep + "matrix_poz"+str(id)+".txt",  matrix)
    matrix[2,:] = -vect_z
    numpy.savetxt(base + out_folder + os.sep + "matrix_neg_"+str(id)+".txt",  matrix)
    numpy.savetxt(base + out_folder + os.sep + "pixel_spacing_"+str(id)+".txt",  (ps[0], ps[1], ss))

    ###################################################################


    # create 3D array
    img_shape = list(f0.pixel_array.shape)
    img_shape.append(len(series_file_list))
    img3d = np.zeros(img_shape)

    print("PixelSpacing = ", ps)
    print("SpacingBetweenSlices = ", ss)
    print("SliceThickness = ", st)
    print("BitsStored = ", bs)
    print("Image shape = ", img_shape)
    print("Offset = ", offset)
    print()

    for i, filename in zip(range(len(series_file_list)), series_file_list):
        f = pydicom.dcmread(filename)
        img3d[:, :, i] = f.pixel_array

    return img3d, img_shape, [ps[0], ps[1], ss], offset, matrix

def normalize(img):
    img_norm = img/numpy.max(img)*255
    img_norm = img_norm.astype('uint8')

    return img_norm 


base = "/Users/skardova/Documents/MRI_data/2026-Eyes-project/Trial2/"

src_folder_1 = "DelRec - PDT1 0.5 Fat AIMax"
out_folder_1 = "VTI_0.5_fat_AIMax"

src_folder_2 = "PDwT2 Y Rap shim1"
out_folder_2 = "VTI_PDwT2_shim1"

out_folder = "VTI_shared_view"
###################################################################

if not os.path.isdir(base + out_folder):
    os.mkdir(base + out_folder)

###################################################################

filename_list = []
slices_list = []
listcount = 0
skipcount = 0


image_1, img_shape_1, pixel_spacing_1, offset_1, matrix_1 = read_dicom(src_folder_1, 1)
image_2, img_shape_2, pixel_spacing_2, offset_2, matrix_2 = read_dicom(src_folder_2, 2)


print("image_1 = ", image_1.shape)
print("image_2 = ", image_2.shape)

image_1 = normalize(image_1)
image_2 = normalize(image_2)
###################################################################

v1_1 = matrix_1[0,:]
v1_2 = matrix_1[1,:]
v1_3 = matrix_1[2,:]

v2_1 = matrix_2[0,:]
v2_2 = matrix_2[1,:]
v2_3 = matrix_2[2,:]

print("v3 vs v3 = ", numpy.dot(v1_3, v2_3))

angle = numpy.arccos(numpy.dot(v1_1, v2_1)/(numpy.sqrt(numpy.dot(v1_1, v1_1))*numpy.sqrt(numpy.dot(v2_1, v2_1))))

print("angle = ", angle/numpy.pi*180)
print()


image_2 = rotate_volume_to_match(image_2, v2_1, v2_2, v1_1, v1_2, order=1)

print("FOV 1 = ", img_shape_1*numpy.array((pixel_spacing_1[0], pixel_spacing_1[1], pixel_spacing_1[2])))
print("FOV 2 = ", img_shape_2*numpy.array((pixel_spacing_2[0], pixel_spacing_2[1], pixel_spacing_2[2])))

################################################


vtk_image = numpy2VTK(image_1)
vtk_image.SetOrigin( offset_1)
vtk_image.SetSpacing(pixel_spacing_1)

myvtk.writeImage(vtk_image, base + out_folder + os.sep + "view-1.vti")
numpy.save(base + out_folder + os.sep + "view_1.npy", image_1)


vtk_image = numpy2VTK(image_2)
vtk_image.SetOrigin( offset_2)
vtk_image.SetSpacing(pixel_spacing_2)

myvtk.writeImage(vtk_image, base + out_folder + os.sep + "rot-view-2.vti")
numpy.save(base + out_folder + os.sep + "rot-view_2.npy", image_2)
#################################################


