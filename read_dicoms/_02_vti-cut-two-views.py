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


def new_dimensions(range_x, range_y, range_z, pixel_spacing):
    
    dim_x = int(numpy.ceil((range_x[1]-range_x[0])/pixel_spacing[0]))
    dim_y = int(numpy.ceil((range_y[1]-range_y[0])/pixel_spacing[1]))
    dim_z = int(numpy.ceil((range_z[1]-range_z[0])/pixel_spacing[2]))

    return numpy.array((dim_x, dim_y, dim_z))

def new_offset(ideal_offset_coord, offset, pixel_spacing):

    pixel_shift = numpy.floor((ideal_offset_coord - offset)/pixel_spacing)
    pixel_shift = [int(x) for x in pixel_shift]
    new_offset = offset + pixel_shift*pixel_spacing

    return new_offset, pixel_shift


################################################################################################


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

image_1 = numpy.load(base + out_folder + os.sep + "view_1.npy")
image_2 = numpy.load(base + out_folder + os.sep + "rot-view_2.npy")

offset_1 = numpy.loadtxt(base + out_folder + os.sep + "offset_1.txt")
pixel_spacing_1 = numpy.loadtxt(base + out_folder + os.sep + "pixel_spacing_1.txt")

offset_2 = numpy.loadtxt(base + out_folder + os.sep + "offset_2.txt")
pixel_spacing_2 = numpy.loadtxt(base + out_folder + os.sep + "pixel_spacing_2.txt")

###################################################################


range_x = [-81, -27]
range_y = [-61, -10]
range_z = [30, 72]

new_offset_1, pixel_shift_1 = new_offset([range_x[0], range_y[0], range_z[0]], offset_1, pixel_spacing_1)
new_offset_2, pixel_shift_2 = new_offset([range_x[0], range_y[0], range_z[0]], offset_2, pixel_spacing_2)

new_shape_1 = new_dimensions(range_x, range_y, range_z, pixel_spacing_1)
new_shape_2 = new_dimensions(range_x, range_y, range_z, pixel_spacing_2)


print(pixel_shift_1[0], pixel_shift_1[0]+new_shape_1[0])
print(pixel_shift_1[1], pixel_shift_1[1]+new_shape_1[1])
print(pixel_shift_1[2], pixel_shift_1[2]+new_shape_1[2])

image_1_cut = image_1[pixel_shift_1[0]:pixel_shift_1[0]+new_shape_1[0], pixel_shift_1[1]:pixel_shift_1[1]+new_shape_1[1], pixel_shift_1[2]:pixel_shift_1[2]+new_shape_1[2]]

image_2_cut = image_2[pixel_shift_2[0]:pixel_shift_2[0]+new_shape_2[0], pixel_shift_2[1]:pixel_shift_2[1]+new_shape_2[1], pixel_shift_2[2]:pixel_shift_2[2]+new_shape_2[2]]



vtk_image = numpy2VTK(image_1_cut)
vtk_image.SetOrigin( new_offset_1)
vtk_image.SetSpacing(pixel_spacing_1)

myvtk.writeImage(vtk_image, base + out_folder + os.sep + "view-1-cut.vti")
numpy.save(base + out_folder + os.sep + "view_1_cut.npy", image_1)


vtk_image = numpy2VTK(image_2_cut)
vtk_image.SetOrigin( new_offset_2)
vtk_image.SetSpacing(pixel_spacing_2)

myvtk.writeImage(vtk_image, base + out_folder + os.sep + "rot-view-2-cut.vti")
numpy.save(base + out_folder + os.sep + "rot-view_2_cut.npy", image_2)
#################################################


