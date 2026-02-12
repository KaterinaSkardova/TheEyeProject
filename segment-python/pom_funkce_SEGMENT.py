import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as numpy
import myVTKPythonLibrary as myvtk
import matplotlib.pyplot as plt

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkPolyData,
    vtkPolyLine
)
#from scipy.ndimage import zoom, generic_gradient_magnitude
#from scipy.ndimage import zoom
from scipy import ndimage, misc

from pom_funkce_VTK import numpy2VTK, writeImage, read_VTI, write_line
from matplotlib.backend_bases import MouseButton
from skimage.morphology import skeletonize, opening




def get_ball(c, r, dim):

    c = [ int(coord) for coord in c]

    set = []
    for i in range(c[0]-r, c[0]+r+1):
        for j in range(c[1]-r, c[1]+r+1):
            for k in range(c[2]-r, c[2]+r+1):
                if  ( i < dim[0]-1 and j < dim[1]-1 and k < dim[2]-1 and i >0 and j >0   and k >0):
                    if (c[0]-i)*(c[0]-i) + (c[1]-j)*(c[1]-j) + (c[2]-k)*(c[2]-k) < r*r :
                        set.append((i,j,k))
    #else:
        #print("Seed not in frame")
    return set


def get_hollow_data_ball(c, r, dim, data, cell):

    c = [ int(coord) for coord in c]

    set = []
    values = []

    for i in range(c[0]-r, c[0]+r+1):
        for j in range(c[1]-r, c[1]+r+1):
            for k in range(c[2]-r, c[2]+r+1):
                if  ( i < dim[0]-1 and j < dim[1]-1 and k < dim[2]-1 and i >0 and j >0   and k >0):
                    if (c[0]-i)*(c[0]-i) + (c[1]-j)*(c[1]-j) + (c[2]-k)*(c[2]-k) < r*r :
                        if (cell[i,j,k]==1) and not(i == c[0] and j == c[1] and k ==c[2]):
                            set.append((i,j,k))
                            values.append(data[i,j,k])

    #else:
        #print("Seed not in frame")

    return set, values

def get_neighbours(y, dim):
    set = []

    # 6 zakladnih smeru

    if (y[0]<dim[0]-1):
        set.append((y[0]+1, y[1], y[2]))
    if (y[0]>0):
        set.append((y[0]-1, y[1], y[2]))
    if (y[1]<dim[1]-1):
        set.append((y[0], y[1]+1, y[2]))
    if (y[1]>0):
        set.append((y[0], y[1]-1, y[2]))
    if (y[2]<dim[2]-1):
        set.append((y[0], y[1], y[2]+1))
    if (y[2]>0):
        set.append((y[0], y[1], y[2]-1))
    return set



def submatrix(arr):
    x, y, z = numpy.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min()-1:x.max()+2, y.min()-1:y.max()+2, z.min()-1:z.max()+2], (x.min()-1, y.min()-1, z.min()-1)
