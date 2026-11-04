import numpy
# import skimage
import meshio
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA




def fit_target_constrained_spline(binned_centers, target_point, order, extension_factor = 0.05, smoothing_factor=3.5):
    """
    binned_centers: (N, 3) array of centers from the mask
    target_point: (3,) array [z, y, x] where the object should end
    """
    # 1. Append the target point to the end of your centers
    # We ensure target_point is shaped correctly for concatenation
    target_point = numpy.array(target_point).reshape(1, 3)
    constrained_points = numpy.concatenate([binned_centers, target_point], axis=0)
    
    # 2. Assign Weights (w)
    # We give the target point a very high weight (e.g., 10) 
    # and the noisy data points a lower weight (e.g., 1)
    weights = numpy.ones(len(constrained_points))
    weights[-1] = 20.0  # Force the spline to prioritize hitting the target
    
    # print("constrained_points = ", constrained_points)

    # 3. Fit the Spline
    # We use the transpose .T because splprep expects (3, N)
    try:
        # Note: 'w' is the weights, 's' is the overall smoothing
        tck, u = splprep(constrained_points.T, w=weights, s=smoothing_factor, k=order)
        
        # 4. Evaluate from 0 to 1
        # Since the target is now the last point in the data, 
        # u=1.0 will correspond to the target point.
        u_new = numpy.linspace(-extension_factor, 1.0, 160)
        smooth_centerline = numpy.array(splev(u_new, tck)).T
        
        return smooth_centerline
    except Exception as e:
        print(f"Constrained spline fit failed: {e}")
        return None
    

def fit_3_constrained_spline(binned_centers, target_1, target_2, target_3, order, extension_factor = 0.05, smoothing_factor=3.5):
    """
    Fits a spline by truncating binned centers at target_1 to avoid over-fitting,
    then sequentially appends target_2 and target_3.
    
    Ordering: [Body Part 1 (Up to Target 1)] -> [Target 1] -> [Target 2] -> [Target 3]
    """
    # 1. Standardize shapes
    target_1 = numpy.array(target_1).reshape(1, 3) # Intersects mid-body (Truncation point)
    target_2 = numpy.array(target_2).reshape(1, 3) # Intersects near end
    target_3 = numpy.array(target_3).reshape(1, 3) # Eye axis anchor (Absolute End)

    # 2. Find where target_1 belongs along the binned centers
    distances = numpy.linalg.norm(binned_centers - target_1, axis=1)
    split_idx = numpy.argmin(distances)

    # 3. Keep ONLY the binned centers that come strictly before target_1
    # We discard binned_centers[split_idx:] to prevent the spline from fight-fitting noise
    binned_truncated = binned_centers[:split_idx]

    # 4. Assemble the final sequence sequentially
    constrained_points = numpy.concatenate([
        binned_truncated,
        target_1,
        target_2,
        target_3
    ], axis=0)

    # 5. Assign precise weights based on the new truncated layout
    weights = numpy.ones(len(constrained_points))
    
    target_1_final_idx = len(binned_truncated)
    target_2_final_idx = len(constrained_points) - 2
    target_3_final_idx = len(constrained_points) - 1

    weights[target_1_final_idx] = 10.0  # Attract to target
    weights[target_2_final_idx] = 10.0  # Attract to target
    weights[target_3_final_idx] = 20.0  # Attract to target

    print(f"Truncated muscle body from {len(binned_centers)} down to {len(binned_truncated)} centers.")

    # 6. Fit the Spline
    try:
        tck, u = splprep(constrained_points.T, w=weights, s=smoothing_factor, k=order)
        
        u_new = numpy.linspace(-extension_factor, 1.0, 160)
        smooth_centerline = numpy.array(splev(u_new, tck)).T
        
        return smooth_centerline
    except Exception as e:
        print(f"Truncated 3-point constrained spline fit failed: {e}")
        return None


def fit_dual_constrained_spline(binned_centers, target_1, target_2, order, extension_factor = 0.05, smoothing_factor=3.5):
    """
    binned_centers: (N, 3) array of centers from the mask
    target_1: (3,) array where the object should begin
    target_2: (3,) array where the object should end
    """
    # 1. Prepare the sequence: [Start, Binned Data, End]
    points = [
        binned_centers,
        numpy.array(target_1).reshape(1, 3),
        numpy.array(target_2).reshape(1, 3)
    ]
    constrained_points = numpy.concatenate(points, axis=0)
    
    # 2. Assign Weights
    # Weights[0] is Start, Weights[-1] is End. 
    # High values force the spline to hit those exact coordinates.
    weights = numpy.ones(len(constrained_points))
    weights[0] = 50.0  # Force start
    weights[-1] = 50.0 # Force end
    
    # 3. Fit the Spline
    # We pass the weights array to 'w'
    try:
        tck, u = splprep(constrained_points.T, w=weights, s=smoothing_factor, k=order)
        
        # 4. Evaluate
        # u=0.0 will be target_start, u=1.0 will be target_end
        u_new = numpy.linspace(-extension_factor, 1.0, 160)
        smooth_centerline = numpy.array(splev(u_new, tck)).T
        
        return smooth_centerline
    except Exception as e:
        print(f"Dual constrained spline fit failed: {e}")
        return None

def fit_analytical_centerline(mask, target_point, ignored_nodes, order, n_bins=20, extension_factor=0.2, both_sides = False):
        # 1. Get all voxel coordinates (Z, Y, X)
        points = numpy.argwhere(mask == 1).astype(float)
        
        if len(points) < 10:
            return None

        # 2. Find the main orientation using PCA
        pca = PCA(n_components=1)
        # Project points onto the first principal component 
        projection = pca.fit_transform(points)


        # =========================================================================
        # 2.5 FORCE ORIENTATION TOWARD EYE 
        # =========================================================================
        if len(target_point) > 0:
            # The direction of the PCA axis vector in 3D space
            pca_axis = pca.components_[0] 
            
            # Vector pointing from the muscle center of mass to the target point
            center_of_mass = points.mean(axis=0)
            direction_to_target = numpy.array(target_point[-1]) - center_of_mass
            
            # If the dot product is negative, the PCA axis points AWAY from the target.
            if numpy.dot(pca_axis, direction_to_target) < 0:
                print("reverse")
                projection = -projection

        
        # 3. Sort points based on their position along the principal axis
        sort_idx = numpy.argsort(projection.flatten())

        sort_idx = sort_idx[int(len(sort_idx)*ignored_nodes):int(len(sort_idx)*(1-ignored_nodes))]

        sorted_points = points[sort_idx]
        sorted_projection = projection[sort_idx].flatten()

        # 4. Create "Binned Centers" to handle the noise
        # Fit to the average center of 'n' slices
        bins = numpy.linspace(sorted_projection.min(), sorted_projection.max(), n_bins)
        binned_centers = []
        
        for i in range(len(bins)-1):
            # Find points that fall within this slice of the object
            mask_slice = (sorted_projection >= bins[i]) & (sorted_projection < bins[i+1])
            if numpy.any(mask_slice):
                center = sorted_points[mask_slice].mean(axis=0)
                binned_centers.append(center)
        
        binned_centers = numpy.array(binned_centers)

        # # 5. Fit the Spline to the centers
        tck, u = splprep(binned_centers.T, s=3.5, k=2)
        
        # # 6. Extend the object
        if both_sides:
            smooth_centerline = numpy.array(splev(numpy.linspace(-extension_factor, 1.0 + extension_factor , 100), tck)).T
        else:
            # # Evaluating from 0.0 to 1+extension extends the centerline
            smooth_centerline = numpy.array(splev(numpy.linspace(-extension_factor, 1.0 , 100), tck)).T

        if len(target_point)==0:
            return smooth_centerline
        
        if len(target_point)==1:
            smooth_centerline_extended = fit_target_constrained_spline(binned_centers, target_point[0], order, extension_factor)
            return smooth_centerline, smooth_centerline_extended
        
        if len(target_point)==2:
            smooth_centerline_extended = fit_dual_constrained_spline(binned_centers, target_point[0], target_point[1], order, extension_factor )
            return smooth_centerline, smooth_centerline_extended

        if len(target_point)==3:
            smooth_centerline_extended = fit_3_constrained_spline(binned_centers, target_point[0], target_point[1], target_point[2], order, extension_factor )
            return smooth_centerline, smooth_centerline_extended    