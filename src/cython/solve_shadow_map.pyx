import cython
cimport cython
cimport numpy as cnp
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_shadow_map_cy(cnp.ndarray[cnp.float32_t, ndim = 2] ray_point, 
                        cnp.ndarray[cnp.float32_t, ndim = 2] ray_vec,
                        cnp.ndarray[cnp.float32_t, ndim = 1] minBound, 
                        cnp.ndarray[cnp.float32_t, ndim = 1] maxBound, 
                        cnp.ndarray[cnp.uint8_t, ndim = 3] terrain_voxel_map):

    cdef cnp.ndarray[cnp.int32_t, ndim = 1] cur_vox
    cdef cnp.ndarray[cnp.int32_t, ndim = 1] terrain_voxel_map_shape = np.array([terrain_voxel_map.shape[0], terrain_voxel_map.shape[1], terrain_voxel_map.shape[2]], dtype = np.int32)
    cdef cnp.ndarray[cnp.int32_t, ndim = 1] step
    cdef cnp.ndarray[cnp.int32_t, ndim = 2] cum_shadow_map = np.zeros([terrain_voxel_map.shape[0], terrain_voxel_map.shape[1]], dtype = np.int32)

    cdef cnp.ndarray[cnp.uint8_t, ndim = 2] temp_shadow_map = np.zeros([terrain_voxel_map_shape[0], terrain_voxel_map_shape[1]], dtype = np.uint8)

    cdef cnp.ndarray[cnp.float32_t, ndim = 1] boxSize
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] tVoxel
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] tMax
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] tDelta
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] ray
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] point
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] voxelMax
    cdef cnp.ndarray[cnp.float32_t, ndim = 1] voxelSize

    cdef cnp.float32_t x0, y0, z0, A, B, C, t, x, y
    cdef cnp.int32_t i, j, k, step0, step1, step2
    cdef cnp.float32_t terrain_max_elevation = maxBound[2] - 1

    for i in range(ray_vec.shape[0]):
        ray = ray_vec[i, :]

        for j in range(ray_point.shape[0]):
            point = ray_point[j, :]
            if terrain_max_elevation < point[2]:
                x0 = point[0]
                y0 = point[1]
                z0 = point[2]
                A = ray[0]
                B = ray[1]
                C = ray[2]
                t = (terrain_max_elevation - z0) / C
                x = x0 + A * t
                y = y0 + B * t
                point = np.array([x, y, terrain_max_elevation], dtype = np.float32)

            boxSize = maxBound - minBound
            cur_vox = np.floor(((point - minBound) / boxSize) * terrain_voxel_map_shape).astype(np.int32)

            if cur_vox[0] >= terrain_voxel_map_shape[0] or cur_vox[0] < 0:
                continue
            elif cur_vox[1] >= terrain_voxel_map_shape[1] or cur_vox[1] < 0:
                continue
            elif cur_vox[2] >= terrain_voxel_map_shape[2] or cur_vox[2] < 0:
                continue
            
            step = np.ones(3, dtype=np.int32)
            tVoxel = np.empty(3, dtype=np.float32)

            if ray[0] >= 0:
                tVoxel[0] = (cur_vox[0] + 1) / terrain_voxel_map_shape[0]
            else:
                tVoxel[0] = cur_vox[0] / terrain_voxel_map_shape[0]
                step[0] = -1

            if ray[1] >= 0:
                tVoxel[1] = (cur_vox[1] + 1) / terrain_voxel_map_shape[1]
            else:
                tVoxel[1] = cur_vox[1] / terrain_voxel_map_shape[1]
                step[1] = -1

            if ray[2] >= 0:
                tVoxel[2] = (cur_vox[2] + 1) / terrain_voxel_map_shape[2]
            else:
                tVoxel[2] = cur_vox[2] / terrain_voxel_map_shape[2]
                step[2] = -1

            voxelMax = minBound + tVoxel * boxSize
            tMax = (voxelMax - point) / ray
            voxelSize = (boxSize / terrain_voxel_map_shape).astype(np.float32)
            tDelta = voxelSize / np.abs(ray)

            while True:
                if tMax[0] < tMax[1]:
                    if tMax[0] < tMax[2]:
                        cur_vox[0] += step[0]
                        if (cur_vox[0] >= terrain_voxel_map_shape[0]) or (cur_vox[0] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]] == 1:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = 1
                            break
                        tMax[0] += tDelta[0]
                    else:
                        cur_vox[2] += step[2]
                        if cur_vox[2] >= terrain_voxel_map_shape[2] or (cur_vox[2] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]] == 1:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = 1
                            break
                        tMax[2] += tDelta[2]
                else:
                    if tMax[1] < tMax[2]:
                        cur_vox[1] += step[1]
                        if cur_vox[1] >= terrain_voxel_map_shape[1] or (cur_vox[1] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]] == 1:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = 1
                            break
                        tMax[1] += tDelta[1]
                    else:
                        cur_vox[2] += step[2]
                        if cur_vox[2] >= terrain_voxel_map_shape[2] or (cur_vox[2] < 0):
                            break    
                        if terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]] == 1:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = 1
                            break
                        tMax[2] += tDelta[2]
                        
            cum_shadow_map[temp_shadow_map == 1] += 1
            temp_shadow_map = np.zeros([terrain_voxel_map_shape[0], terrain_voxel_map_shape[1]], dtype = np.uint8)
    return cum_shadow_map