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



base = "/Users/skardova/Documents/MRI_data/2026-Eyes-project/Trial2/"

src_folder = "DelRec - PDT1 0.5 Fat AIMax"
out_folder = "VTI_0.5_fat_AIMax"

# src_folder = "PDwT2 Y Rap shim1"
# out_folder = "VTI_PDwT2_shim1"

###################################################################

if not os.path.isdir(base + out_folder):
    os.mkdir(base + out_folder)

###################################################################

filename_list = []
slices_list = []
listcount = 0
skipcount = 0

for filename in glob.glob(base + src_folder + os.sep    + "*.DCM", recursive=True):

    f = pydicom.dcmread(filename)
    if hasattr(f,"SliceLocation"):
        filename_list.append(filename)
        slices_list.append(float(f.SliceLocation))
        listcount = listcount + 1
    else:
        skipcount = skipcount + 1

# slices_list.sort()
# slices_dict = { item : i for i, item in enumerate(slices_list) 

series_file_list = sorted(filename_list, key = lambda s: pydicom.dcmread(s).SliceLocation)

f0 = pydicom.dcmread(series_file_list[-1])
ps = f0.PixelSpacing
ss = f0.SpacingBetweenSlices
st = f0.SliceThickness
bs = f0.BitsStored
offset = f0.ImagePositionPatient # (0020, 0032)
oriantation = f0.ImageOrientationPatient # (0020, 0037)

vect_row = oriantation[0:3]
vect_col = oriantation[3:7]
vect_z = numpy.cross(vect_row, vect_col)
vect_z = vect_z/ numpy.linalg.norm(vect_z)

matrix = numpy.zeros((3,3))
matrix[1,:] = vect_row
matrix[0,:] = vect_col
matrix[2,:] = vect_z


numpy.savetxt(base + out_folder + os.sep + "offset.txt",  offset)
numpy.savetxt(base + out_folder + os.sep + "matrix_poz.txt",  matrix)
matrix[2,:] = -vect_z
numpy.savetxt(base + out_folder + os.sep + "matrix_neg.txt",  matrix)
numpy.savetxt(base + out_folder + os.sep + "pixel_spacing.txt",  (ps[0], ps[1], ss))

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

###################################################################

img3d_cut = img3d[100:300,170:340,30:150]

print("FOV = ", img_shape*numpy.array((ps[0], ps[1], ss)))

#################################################
#img3d_norm = img3d + abs( numpy.min(img3d))
#img3d_norm = img3d_norm/numpy.max(img3d_norm)*255
#img3d_norm = img3d_norm.astype('uint8')
#img3d_norm = img3d.astype('float64')
# img3d_norm = img3d/numpy.max(img3d)*pow(2,8)
# img3d_norm = img3d_norm.astype('uint8')
#################################################


# vtk_image = numpy2VTK(img3d_cut)
# vtk_image.SetOrigin( offset)
# vtk_image.SetSpacing((ps[0], ps[1], ss))


vtk_image = numpy2VTK(img3d_cut)
vtk_image.SetOrigin( (0,0,0))
vtk_image.SetSpacing((1,1,1))

myvtk.writeImage(vtk_image, base + out_folder + os.sep + "cut.vti")

numpy.save(base + out_folder + os.sep + "cut.npy", img3d_cut)
#################################################


