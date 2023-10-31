import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

def voxel_traversal(np.ndarray[double, ndim=1] ray_start, 
                    np.ndarray[double, ndim=1] ray_vec,
                    np.ndarray[double, ndim=1] grid_element_size,
                    np.ndarray[int, ndim=3] terrain_array,
                    double max_elevation=0):

    cdef double x0, y0, z0
    cdef double A, B, C
    cdef double t
    cdef double x, y
    cdef np.ndarray[int, ndim=1] current_voxel
    cdef np.ndarray[int, ndim=2] visited_voxels
    cdef np.ndarray[int, ndim=1] grid_max_boundary
    cdef np.ndarray[int, ndim=1] step
    cdef np.ndarray[double, ndim=1] next_voxel_boundary
    cdef np.ndarray[double, ndim=1] tMax
    cdef np.ndarray[double, ndim=1] tDelta
    cdef np.ndarray[int, ndim=1] diff
    cdef bint neg_ray
    cdef int i

    if max_elevation != 0:
        x0 = <double>ray_start[0]
        y0 = <double>ray_start[1] 
        z0 = <double>ray_start[2]
        A = ray_vec[0]
        B = ray_vec[1]
        C = ray_vec[2]

        t = (max_elevation - z0) / C
        x = x0 + A * t
        y = y0 + B * t
        
        ray_start[0] = x
        ray_start[1] = y
        ray_start[2] = max_elevation

    current_voxel = <int[:ray_start.shape[0]]>np.floor(ray_start / grid_element_size).astype(int)
    visited_voxels = np.reshape(current_voxel.copy(), (1, 3))
    grid_max_boundary = <int[:terrain_array.shape[0]]>np.array(terrain_array.shape)

    step = np.sign(ray_vec)
    next_voxel_boundary = (current_voxel + step) * grid_element_size
    tMax = (next_voxel_boundary - ray_start) / ray_vec
    tDelta = grid_element_size / ray_vec * step

    diff = np.zeros(3, dtype=int)
    neg_ray = any(ray < 0 for ray in ray_vec)
    diff -= np.less(ray_vec, 0).astype(int)
    if neg_ray:
        current_voxel += diff
        visited_voxels = np.row_stack((visited_voxels, current_voxel))
    
    i = 0
    while i < 10000:
        if tMax[0] < tMax[1]:
            if tMax[0] < tMax[2]:
                current_voxel[0] += step[0]
                if (current_voxel[0] == grid_max_boundary[0] or 
                    current_voxel[0] == 0 or
                    terrain_array[current_voxel[0], current_voxel[1], current_voxel[2]]):
                    return np.row_stack((visited_voxels, current_voxel))
                tMax[0] += tDelta[0]
            else:
                current_voxel[2] += step[2]
                if current_voxel[2] == grid_max_boundary[2] or current_voxel[2] == 0 or terrain_array[current_voxel[0], current_voxel[1], current_voxel[2]]:
                    return np.row_stack((visited_voxels, current_voxel))
                tMax[2] += tDelta[2]
        else:
            if tMax[1] < tMax[2]:
                current_voxel[1] += step[1]
                if current_voxel[1] == grid_max_boundary[1] or current_voxel[1] == 0 or terrain_array[current_voxel[0], current_voxel[1], current_voxel[2]]:
                    return np.row_stack((visited_voxels, current_voxel))
                tMax[1] += tDelta[1]
            else:
                current_voxel[2] += step[2]
                if current_voxel[2] == grid_max_boundary[2] or current_voxel[2] == 0 or terrain_array[current_voxel[0], current_voxel[1], current_voxel[2]]:
                    return np.row_stack((visited_voxels, current_voxel))
                tMax[2] += tDelta[2]
        
        visited_voxels = np.row_stack((visited_voxels, current_voxel))
        i += 1
        
    return visited_voxels