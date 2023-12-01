from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import pyproj
from pyproj import Geod
import os
import matplotlib.pyplot as plt
import requests
import urllib.request
from io import BytesIO
from subprocess import check_output
import inspect
from IPython.display import display
from datetime import datetime
from scipy.interpolate import RectBivariateSpline, griddata
import gzip
import pathlib
from pathlib import Path
import json
import pandas as pd
import cartopy.geodesic as cgeo
import multiprocessing as mp
import PIL
import importlib.util as _importlib_util 

np.seterr(divide='ignore', invalid='ignore')
ROOT_DIR = Path(os.path.abspath(__file__)).parent.parent

def load_dynamic(name, module_path): 
    spec = _importlib_util.spec_from_file_location(name, module_path) 
    module = _importlib_util.module_from_spec(spec) 
    spec.loader.exec_module(module) 
    return module 
solve_shadow_map = load_dynamic("solve_shadow_map", Path.joinpath(ROOT_DIR, "src/cython/solve_shadow_map.cpython-311-x86_64-linux-gnu.so"))

def get_api_key():
    api_key_file_path = Path.joinpath(ROOT_DIR, "assets/api_key.txt")

    # Check if the file exists
    if os.path.exists(api_key_file_path):
        # If the file exists, read the API key from the file
        with open(api_key_file_path, 'r') as file:
            api_key = file.read().strip()
    else:
        # If the file does not exist, prompt the user for an API key
        api_key = input("Enter API key: ")

        # Save the API key to the file
        with open(api_key_file_path, 'w') as file:
            file.write(api_key)
    return api_key

def voxel_traversal(origin, direction, grid3D, verbose=False):
    boxSize = grid3D['maxBound'] - grid3D['minBound']

    cur_vox = np.floor(((origin - grid3D['minBound']) / boxSize) * grid3D["n"])
    visited_vox = []
    step = np.ones(3)
    tVoxel = np.empty(3)

    if direction[0] >= 0:
      tVoxel[0] = (cur_vox[0] + 1) / grid3D['n'][0]
    else:
      tVoxel[0] = cur_vox[0]/ grid3D['n'][0]
      step[0] = -1
    
    if direction[1] >= 0:
      tVoxel[1]= (cur_vox[1] + 1) / grid3D['n'][1]
    else:
      tVoxel[1] = cur_vox[1] / grid3D['n'][1]
      step[1] = -1
    
    if direction[2] >= 0:
      tVoxel[2] = (cur_vox[2] + 1) / grid3D['n'][2]
    else:
      tVoxel[2] = cur_vox[2] / grid3D['n'][2]
      step[2] = -1

    voxelMax = grid3D['minBound'] + tVoxel*boxSize
    tMax = (voxelMax - origin) / direction
    voxelSize = boxSize / grid3D['n']
    tDelta = voxelSize / abs(direction)
    visited_vox.append(cur_vox.copy())
    while (cur_vox[0] < grid3D['n'][0]) and (cur_vox[0] >= 0) and (cur_vox[1] < grid3D['n'][1]) and (cur_vox[1] >= 0) and (cur_vox[2] < grid3D['n'][2]) and (cur_vox[2] >= 0):
      if verbose:
        print(f'Intersection: voxel = ({cur_vox[0]}, {cur_vox[1]}, {cur_vox[2]})')

      if tMax[0] < tMax[1]:
        if tMax[0] < tMax[2]:
          cur_vox[0] += step[0]
          tMax[0] += tDelta[0]
        else:
          cur_vox[2] += step[2]
          tMax[2] += tDelta[2]
      else:
        if tMax[1] < tMax[2]:
          cur_vox[1] += step[1]
          tMax[1] += tDelta[1]
        else:
          cur_vox[2] += step[2]
          tMax[2] += tDelta[2]
      visited_vox.append(cur_vox.copy())
    return visited_vox

def import_point_source_data(file_name):
    input_folder = Path.joinpath(ROOT_DIR, "inputs")
    file_path = Path.joinpath(input_folder, file_name)
    with open(file_path, 'r') as file:
        point_source_data = json.load(file)
    df = pd.DataFrame(point_source_data)
    return df

def intrinsic_parameters(f, shape, fov):
    """
    INPUT:
        f : float
            focal length
        shape : array_like
            shape of image [pixels]
        fov : array_like
            horizontal and vertical field-of-view
    OUTPUT:
        K : ndarray (3, 3)
                camera calibration matrix
    """
    hfov = np.deg2rad(fov[0]) # horizontal field of view
    vfov = np.deg2rad(fov[1]) # vertical field of view
    m_x = 1/((np.tan(hfov/2)*f)/(shape[1]/2))
    m_y = 1/((np.tan(vfov/2)*f)/(shape[0]/2))
    a_x = f * m_x; a_y = f * m_y

    K = np.array([[a_x, 0, shape[1]/2], 
                [0, a_y, shape[0]/2], 
                [0, 0, 1]])
    
    return K

def rotation_matrix(theta, order = "ZYX"):
    """
    INPUT:
        theta : array_like (3, 1)
            Euler angles [rad]
    OUTPUT:
        R : ndarray (3, 3)
            rotation matrix
    """
    R_x = np.array([[1,0,0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0,np.sin(theta[0]),np.cos(theta[0])]])
    R_y = np.array([[np.cos(theta[1]),0,np.sin(theta[1])],
                    [0,1,0],
                    [-np.sin(theta[1]),0,np.cos(theta[1])]])
    R_z = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                    [np.sin(theta[2]), np.cos(theta[2]),0],
                    [0,0,1]])
    
    # if order.upper() == "ZYX":
    #     R = R_z @ R_y @ R_x
    # elif order.upper() == "XYZ":
    #     R = R_x @ R_y @ R_z

    R = np.eye(3) # init R

    for axis in order.upper():
        if axis == 'X':
            R = R @ R_x
        elif axis == 'Y':
            R = R @ R_y
        elif axis == 'Z':
            R = R @ R_z

    return R

def extrinsic_parameters(R, n):
    """
    INPUT:
        R : array_like (3, 3)
            rotation matrix
        n : array_like
            camera origin
    OUTPUT:
        C_N : ndarray (3, 4)
            normalized camera matrix 
    """
    C_N = R @ np.column_stack((np.identity(3),-n)) # normalized camera matrix 
    return C_N

def camera_matrix(K, C_N):
    P = K @ C_N
    return P

def image_plane(P, point_coordinate):
    point_coordinate = np.append(point_coordinate, 1)
    res = P @ point_coordinate
    u, v = (res/res[2])[:2]
    return u, v

def object_frame_boundaries(object_position, scale_factor : float):
    perp_vector = np.array([-object_position[1], object_position[0]])
    perp_vector = perp_vector / (np.linalg.norm(perp_vector) / scale_factor)
    return object_position + perp_vector, object_position - perp_vector

def find_coeffs(pa, pb): # https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def generate_visual_impact(pic_class, turb_class, debug = True):

    shape = pic_class.shape
    f = pic_class.focal_length
    fov = pic_class.fov
    radius = turb_class.radius
    turbine_origin = turb_class.location
    camera_origin = pic_class.location
    height = turb_class.height
    theta = pic_class.theta

    K = intrinsic_parameters(f, shape, fov)
    R = rotation_matrix(theta, order = "xzy")
    C_N = extrinsic_parameters(R, camera_origin)
    P = camera_matrix(K, C_N)

    perp1, perp2 = object_frame_boundaries(turbine_origin[:-1]-camera_origin[:-1], radius)
    point1 = np.append(perp1, 0) + np.append(camera_origin[:-1], 0) + np.array([0,0,turbine_origin[2]])
    point2 = np.append(perp2, 0) + np.append(camera_origin[:-1], 0) + np.array([0,0,turbine_origin[2]])
    point3= np.append(perp1, height) + np.append(camera_origin[:-1], 0) + np.array([0,0,turbine_origin[2]])
    point4 = np.append(perp2, height) + np.append(camera_origin[:-1], 0) + np.array([0,0,turbine_origin[2]])

    p_list = [point1, point2, point3, point4]

    draw = ImageDraw.Draw(pic_class.im)
    pa = []
    for i, p in enumerate(p_list):
        u, v = image_plane(P, p)
        colors = ["red", "green", "blue", "yellow"]
        if debug:
            draw.rectangle(((u,v),(u+5,v+5)),fill = colors[i])
        pa.append([u,v])

    pc = np.array(pa).reshape(4,2)
    pc[:,0] -= np.amin(pc[:,0])
    pc[:,1] -= np.amin(pc[:,1])

    # pb = [(turb_class.shape[1], turb_class.shape[0]), (0, turb_class.shape[0]),  (turb_class.shape[1], 0), (0, 0)]
    pb = [(0, turb_class.shape[0]), (turb_class.shape[1], turb_class.shape[0]),  (0, 0), (turb_class.shape[1], 0)]
    pd = list(pa)
    for i in range(len(pa)):
        pa[i] = [pc[i,0],pc[i,1]]
        
    coeffs = find_coeffs(pa, pb)

    turb_class.im = turb_class.im.transform((int(np.amax(np.array(pa).reshape(4,2)[:,0])),int(np.amax(np.array(pa).reshape(4,2)[:,1]))), method=Image.Transform.PERSPECTIVE,data=coeffs)

    pic_class.im.paste(turb_class.im, box=[np.amin(np.array(pd).reshape(4,2)[:,0]).astype(int),np.amin(np.array(pd).reshape(4,2)[:,1]).astype(int)], mask = turb_class.im)
    pic_class.im.save(Path.joinpath(ROOT_DIR, "temp/site_img.png"))
    return pic_class.im

def calc_angle_dist(location1, location2):
    angle,angle2,distance = Geod(ellps='WGS84').inv(location1[0], location1[1], location2[0] ,location2[1]) # N = 0, E = 90, W = -90, S = 180/-180
    if angle < 0:
        angle = angle + 360
    return angle, distance

def is_in_frame(angle, hfov_range):
    if angle >= hfov_range[0] and angle <= hfov_range[1]:
        return True
    else:
        return False

def pull_street_view_image(api_key, longitude, latitude, fov = 90, heading = 0, pitch = 90, width = 800, height = 800):
# URL of the image you want to load
    pitch = 90 - pitch # correct for reference frame
    image_url = f"https://maps.googleapis.com/maps/api/streetview?size=800x800&location={latitude},{longitude}&fov={fov}&heading={heading}&pitch={pitch}&key={api_key}"
    try:
        # Send an HTTP GET request to fetch the image
        response = requests.get(image_url)
        
        # Check if the request was successful (HTTP status code 200)
        if response.status_code == 200:
            # Get the image data as bytes
            image_data = response.content
            
            # Create a Pillow Image object from the image data
            img = Image.open(BytesIO(image_data))
            
        else:
            print(f"Failed to retrieve the image. Status code: {response.status_code}")

    except Exception as e:
        print(f"An errorr occurred: {str(e)}")

    # img.save(f"../../temp/site_img.png")
    img.save(Path.joinpath(ROOT_DIR, "temp/site_img.png"))

    return img

def adjust_image(image_path, brightness = 1, contrast = 1, display_image = True):
    # Load the image using PIL.
    if isinstance(image_path, (pathlib.PosixPath, str)):
        image = Image.open(image_path)
    elif isinstance(image_path, PIL.Image.Image):
        image = image_path
    else:
        print(f"DataTypeError: {type(image_path)} is an unsupported DataType.")

    # Apply the brightness and contrast adjustments to the image.
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Save or display the adjusted image.
    image.save(Path.joinpath(ROOT_DIR, "temp/obj2png.png"))
    if display_image:
        display(image)
    return image

def print_code(source_code):
        source_code = inspect.getsource(source_code)
        command = ["pygmentize", "-f", "html", "-O", "noclasses, lineanchors,style=native", "-l", "python"]

        # Get the HTML output using check_output
        output = check_output(command, input=source_code, encoding="ascii")

        # Display the HTML using IPython.display
        return output

def load_and_normalize_obj(file_path, total_height = 1):
    V, F = [], []
    with open(file_path) as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                V.append([float(x) for x in values[1:4]])
            elif values[0] == 'f' :
                F.append([int(x.split("/", 1)[0]) for x in values[1:4]])
    V, F = np.array(V), np.array(F)-1
    V = (V - np.min(V[:,2])) / (np.max(V[:,2]) - np.min(V[:,2]))*total_height
    return V, F

def plot_trisurface(file_path, elevation = 0, azimuth = 0, view_height = 0, total_height = 1, show_plot = True):
    
    V, F = load_and_normalize_obj(file_path, total_height = total_height)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")

    plt.axis('off')
    plt.grid(visible=None)
    ax.view_init(elev=elevation, azim=azimuth)

    tri = ax.plot_trisurf(V[:, 0], V[:,1], F, V[:, 2], linewidth=0.1, antialiased=True, closed=True)
    tri.set(facecolor = "white", edgecolor = "gray")

    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    lower_bound = np.minimum(0, 2*view_height - total_height)
    upper_bound = np.maximum(total_height, 2*view_height)
    limits[2,:] = [lower_bound, upper_bound]
    ax.set(zlim=limits[2,:], aspect="equal")

    plt.savefig(Path.joinpath(ROOT_DIR, "temp/obj2png.png"), dpi=600, transparent=True)
    if show_plot:
        plt.show()
    else:
        plt.close() # prevent plot from showing

def crop_image(file_path, display_image = True):
    pil_image = Image.open(file_path)
    pil_image = pil_image.crop((5, 5, pil_image.size[0]-5, pil_image.size[1]-5))
    np_array = np.array(pil_image)
    blank_px = [255, 255, 255, 0]
    mask = np_array != blank_px
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    cropped_box = np_array[x0:x1, y0:y1, 0:4]
    pil_image = Image.fromarray(cropped_box, 'RGBA')
    pil_image.save(file_path)

    if display_image:
        display(pil_image)

    return pil_image

def object_to_image(file_path, elevation = 0, azimuth = 0, view_height = 0, total_height = 0, debug = False):
    plot_trisurface(file_path, elevation = elevation, azimuth = azimuth, view_height = view_height, total_height = total_height, show_plot = debug)
    pil_image = crop_image(file_path = Path.joinpath(ROOT_DIR, "temp/obj2png.png"), display_image = debug)
    pil_image = adjust_image(image_path = pil_image, brightness = 1, contrast = 1, display_image = debug)
    return pil_image

def solar_angles_to_vector(azimuth, altitude):
    """
    Convert solar azimuth and altitude angles to a 3D vector

    Args:
        azimuth (float): Solar azimuth angle in radians
        altitude (float): Solar altitude angle in radians

    Returns:
        vec (np.ndarray): Normalized 3D vector representing sun direction from origin
    """
    azimuth = azimuth+np.pi/2

    # Calculate 3D vector components
    x = -np.cos(azimuth) * np.cos(altitude)
    y = np.sin(azimuth) * np.cos(altitude)
    z = np.sin(altitude)

    # Normalize to unit vector
    vec = np.array([x, y, z])
    vec /= np.linalg.norm(vec)

    return vec

def angle_to_unit_vector(angle):
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Calculate the components of the unit vector
    x_component = np.sin(angle_rad)
    y_component = np.cos(angle_rad)
    
    return np.array([x_component, y_component])

def add_solar_axis(fig, ax):
        """
        Generates a double twin axis and automatically hides to ticks. Automatically centers to plot in the frame.
        """
        ax2 = ax.twinx().twiny()
        ax2.set(aspect="equal")
        if fig.get_size_inches()[0] > fig.get_size_inches()[1]:
                ax2.set(xlim = 1.1*np.array([-1,1])*fig.get_size_inches()[0]/fig.get_size_inches()[1])
        ax2.set_xticks([])
        ax2.set_yticks([])
        return ax2

def solar_position(date, lat, lng):
    """
    Calculate the azimuth and altitude of the sun for a given date, latitude and longitude.

    Parameters:
    date (datetime): Date to calculate for
    lat (float): Latitude in degrees
    lng (float): Longitude in degrees
    
    Returns:
    dict: Azimuth and altitude in radians
    """

    # Constants
    rad = np.pi / 180
    epochStart = datetime(1970, 1, 1) 
    J1970 = 2440588
    J2000 = 2451545
    dayMs = 24 * 60 * 60 * 1000
    e = rad * 23.4397 # obliquity of the Earth

    # Convert date to required formats
    ms = (date - epochStart).total_seconds() * 1000
    julian = ms / dayMs - 0.5 + J1970 
    days = julian - J2000

    # Calculate right ascension and declination
    M = rad * (357.5291 + 0.98560028 * days) # Solar mean anomaly
    C = rad * (1.9148 * np.sin(M) + 0.02 * np.sin(2 * M) + 0.0003 * np.sin(3 * M)) # Equation of center
    P = rad * 102.9372 # Perihelion of Earth
    L = M + C + P + np.pi # Ecliptic longitude
    dec = np.arcsin(np.sin(0) * np.cos(e) + np.cos(0) * np.sin(e) * np.sin(L)) 
    ra = np.arctan2(np.sin(L) * np.cos(e), np.cos(L))

    # Calculate sidereal time
    lw = rad * -lng
    st = rad * (280.16 + 360.9856235 * days) - lw

    # Calculate azimuth and altitude
    H = st - ra
    az = np.radians(180) + np.arctan2(np.sin(H), np.cos(H) * np.sin(rad * lat) - np.tan(dec) * np.cos(rad * lat))
    alt = np.arcsin(np.sin(rad * lat) * np.sin(dec) + np.cos(rad * lat) * np.cos(dec) * np.cos(H))
    return az, alt

def download_elevation(map_boundaries):

    long_min = np.minimum(map_boundaries[0], map_boundaries[1])
    long_max = np.maximum(map_boundaries[0], map_boundaries[1])
    lat_min = np.minimum(map_boundaries[2], map_boundaries[3])
    lat_max = np.maximum(map_boundaries[2], map_boundaries[3])

    long_range = np.arange(np.floor(long_min), np.ceil(long_max), 1)
    lat_range = np.arange(np.floor(lat_min), np.ceil(lat_max), 1)

    merged_map = np.zeros([len(lat_range)*3601, len(long_range)*3601])

    for i, latitude in enumerate(lat_range):
        for j, longitude in enumerate(long_range):
            if latitude < 0:
                lat_str = "S"+str(int(np.floor(-latitude))).zfill(2)
            else:
                lat_str = "N"+str(int(np.floor(latitude))).zfill(2)
                
            if longitude < 0:
                long_str = "W"+str(int(np.floor(-longitude))).zfill(3)
            else:
                long_str = "E"+str(int(np.floor(longitude))).zfill(3)

            output_name = f"{lat_str}{long_str}"
            # hgt_gz_file = "../../temp/"+output_name+".hgt.gz"
            # hgt_file = '../../temp/'+ output_name+ '.hgt'

            hgt_gz_file = Path.joinpath(ROOT_DIR, "temp/"+output_name+".hgt.gz")
            hgt_file = Path.joinpath(ROOT_DIR, "temp/"+output_name+".hgt")

            if os.path.exists(hgt_file):
                # print("File exists!")
                pass
            else:
                print("File does not exist.")

                url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_str}/{output_name}"+".hgt.gz"
                
                urllib.request.urlretrieve(url, hgt_gz_file)

                with gzip.open(hgt_gz_file, 'rb') as f_in:
                    with open(hgt_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                os.remove(hgt_gz_file)
            
            with open(hgt_file, 'rb') as f:
                data = np.frombuffer(f.read(), np.dtype('>i2')).reshape((3601, 3601))
                data = np.flip(data, axis=0)

            merged_map[i*3601:(i+1)*3601, j*3601:(j+1)*3601] = data

    srtm_latitude = np.linspace(np.floor(lat_min), np.ceil(lat_max), merged_map.shape[0])
    srtm_longitude = np.linspace(np.floor(long_min), np.ceil(long_max), merged_map.shape[1])

    return srtm_longitude, srtm_latitude, merged_map

def scale_array_func(array, long_list, lat_list, new_shape = None, scaling_factor = None):
        """
        Input:
            array : np.array (2D)
            long_list : np.array (1D)
            lat_list : np.array (1D)
            new_shape : list or tuple or np.array
            scaling_factor : float

        Output:
            scaled_array : np.ndarray (2D)
            scaled_long : np.ndarray (1D)
            scaled_lat : np.ndarray (1D)

        """
        old_shape = np.asarray(np.shape(array))
        
        if (new_shape is not None) or (scaling_factor is not None):

            scaled_shape = old_shape
            new_shape = np.asarray(new_shape)

            if new_shape is not None:
                scaled_shape = new_shape

            if scaling_factor is not None:
                scaled_shape = np.round(scaled_shape * scaling_factor).astype(int)
        else:
            raise ValueError("To scale an array, either the new shape or scaling factor is needed.")
        
        # make points from 0 to 1 for the length of the shape
        x = np.linspace(0, 1, old_shape[0])
        y = np.linspace(0, 1, old_shape[1])

        interp_spline = RectBivariateSpline(x, y, array)
            
        x2 = np.linspace(0, 1, scaled_shape[0])   
        y2 = np.linspace(0, 1, scaled_shape[1])
        
        scaled_array = interp_spline(x2, y2)

        scaled_long = np.interp(np.linspace(0, len(long_list) - 1, scaled_shape[0]), 
                                np.arange(len(long_list)), 
                                long_list)
        
        scaled_lat = np.interp(np.linspace(0, len(lat_list) - 1, scaled_shape[1]), 
                        np.arange(len(lat_list)), 
                        lat_list)


        return scaled_array, scaled_long, scaled_lat

def generate_subarray(original_array, long_points, col_points, subarray_bounds):

    start_long = np.argmin(np.abs(long_points - subarray_bounds[0]))
    end_long = np.argmin(np.abs(long_points - subarray_bounds[1])) 
    start_lat = np.argmin(np.abs(col_points - subarray_bounds[2]))
    end_lat = np.argmin(np.abs(col_points - subarray_bounds[3])) 
    
    subarray = original_array[start_lat:end_lat+1, start_long:end_long+1]
    subarray_bounds = [long_points[start_long], long_points[end_long], col_points[start_lat], col_points[end_lat]]
    sublong_points = long_points[start_long:end_long+1]
    sublat_points = col_points[start_lat:end_lat+1]
    
    return subarray, sublong_points, sublat_points, subarray_bounds

def transform_coordinates(long_list, lat_list, input_crs_str = "EPSG:4326", output_crs_str = "EPSG:3035"):
    
    input_crs = pyproj.CRS(input_crs_str)  # WGS84
    output_crs = pyproj.CRS(output_crs_str)

    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

    if hasattr(long_list, "__len__"):
        trans_cords = np.empty((len(lat_list), len(long_list), 2))
    else:
        trans_cords = np.empty((1, 1, 2))
        long_list = [long_list]
        lat_list = [lat_list]

    for i, lon in enumerate(long_list):
        for j, lat in enumerate(lat_list):
            x, y = transformer.transform(lon, lat)
            trans_cords[j, i, 0] = x
            trans_cords[j, i, 1] = y

    return trans_cords

def rotor_point_spacing(diameter, grid_element_size, angle):
    grid_element_size[2] = grid_element_size[2] * np.tan(angle)

    grid_resolution = min(grid_element_size[0], grid_element_size[0]**2*np.abs(np.tan(angle)))

    n_radius = np.ceil(diameter/(grid_resolution)).astype(int)
    r_list = np.linspace(0, diameter/2, n_radius)
    
    n_list = np.ones(r_list.shape)
    
    for i in np.arange(1, len(n_list)):
        points_per_radius = np.ceil(2*r_list[i]*np.pi/grid_resolution).astype(int)
        n_list[i] = points_per_radius
    

    return r_list, n_list

def generate_turbine(r_list, n_angle, n_vector, turbine_cord):
    
    iteration = 0
    rotor_angle = np.arctan(n_vector[0]/n_vector[1])+np.pi/2
    
    points = np.zeros([sum(n_angle).astype(int),3]) # initiate result point (1 extra for center point)
    
    for i, r in enumerate(r_list):
        angle_list = np.linspace(0, 2*np.pi*(1-1/n_angle[i]), n_angle[i].astype(int))
        for j, angle in enumerate(angle_list):
            x_rel = r * np.cos(angle) * np.cos(rotor_angle)
            y_rel = r * np.cos(angle) * np.sin(rotor_angle)
            z_rel = r * np.sin(angle)
            
            points[iteration,:] = np.array([x_rel, y_rel, z_rel])
            
            iteration += 1
    points += turbine_cord

    return points

def resample_to_straight_axis(trans_cords, map_array, shape):

    x_min = np.max(trans_cords[:,0,0])
    x_max = np.min(trans_cords[:,-1,0])
    y_min = np.max(trans_cords[0,:,1])
    y_max = np.min(trans_cords[-1,:,1])

    X, Y = np.meshgrid(np.linspace(x_min, x_max, shape[0]), np.linspace(y_min, y_max, shape[1]))
    Z = griddata((trans_cords[:,:,0].flatten(), trans_cords[:,:,1].flatten()), map_array.flatten(), (X, Y), method='cubic')
    return X, Y, Z

def generate_voxel_map(map_boundaries, shape):
    srtm_longitude, srtm_latitude, map_array = download_elevation(map_boundaries)
    map_array, sublong_points, sublat_points, map_boundaries = generate_subarray(map_array, srtm_longitude, srtm_latitude, map_boundaries)
    map_array, sublong_points, sublat_points = scale_array_func(map_array, sublong_points, sublat_points, new_shape = shape)
    trans_cords = transform_coordinates(sublong_points, sublat_points, input_crs_str = "EPSG:4326", output_crs_str = "EPSG:3035")
    X, Y, map_array = resample_to_straight_axis(trans_cords, map_array, shape)
    
    map_array_min = np.floor(np.min(map_array)).astype(int)
    map_array_max = np.ceil(np.max(map_array)).astype(int)

    elevation_range = np.arange(map_array_min, map_array_max, 1)
    voxel_map = np.zeros((map_array.shape[0], map_array.shape[1], len(elevation_range)), dtype=np.uint8)
    for i, elev in enumerate(elevation_range):
        voxel_map[:, :, i][map_array > elev] = 1
    return X, Y, voxel_map, map_array

def visual_impact_assesment(elevation_handler, point_source_data, camera_coord, theta, fov = [90, 90]):
    api_key = get_api_key()
    camera_origin = transform_coordinates(camera_coord[0], camera_coord[1], input_crs_str = "EPSG:4326", output_crs_str = "EPSG:3035")[0,0]
    map_array, long_range, lat_range = elevation_handler.map_array, elevation_handler.long_range, elevation_handler.lat_range
    X, Y = np.meshgrid(long_range, lat_range)
    camera_ground_elevation = griddata((X.flatten(), Y.flatten()), map_array.flatten(), (camera_coord[0], camera_coord[1]), method='linear').max() 
    camera_elevation = 2 # Google Street View picture elevation above ground
    camera_origin = np.append(camera_origin, camera_elevation + camera_ground_elevation)
    roll, tilt, yaw = theta
    pull_street_view_image(api_key, camera_coord[0], camera_coord[1], fov = fov[0], heading = yaw, pitch = roll, width = 800, height = 800)
    
    camera_angles, camera_distances = [], []
    for i, point_source in point_source_data.iterrows():
        camera_angle, camera_distance = calc_angle_dist(camera_coord, [point_source.longitude, point_source.latitude])
        camera_angles.append(camera_angle)
        camera_distances.append(camera_distance)
    point_source_data = point_source_data.assign(camera_angle=camera_angles, camera_distance=camera_distances)
    point_source_data.sort_values(by="camera_distance", ascending=False, inplace=True, ignore_index=True)

    for i, point_source in point_source_data.iterrows(): 
        turbine_coord = [point_source.longitude, point_source.latitude]
        turbine_origin = transform_coordinates(*turbine_coord, input_crs_str = "EPSG:4326", output_crs_str = "EPSG:3035")[0,0]
        turbine_ground_elevation = griddata((X.flatten(), Y.flatten()), map_array.flatten(), turbine_coord, method='linear').max() 
        turbine_elevation = 0 # turbine elevation above ground (on ground preferably)
        turbine_height = point_source.h + point_source.d/2
        turbine_origin = np.append(turbine_origin, turbine_elevation + turbine_ground_elevation)

        relative_elevation_diff = camera_ground_elevation + camera_elevation - turbine_ground_elevation - turbine_elevation

        object_to_image(Path.joinpath(ROOT_DIR, "assets/windmill.obj"), elevation = 0, azimuth = point_source.camera_angle +  point_source.wind_dir - 90, view_height = relative_elevation_diff, total_height = turbine_height, debug = False)
        adjust_image(Path.joinpath(ROOT_DIR, "temp/obj2png.png"), brightness = 0.5, contrast = 0.5, display_image = False)
        pic_class = Photo(Path.joinpath(ROOT_DIR, "temp/site_img.png"), fov, np.deg2rad(theta), camera_coord, camera_origin, focal_length = 1)
        turb_class = Turbine(Path.joinpath(ROOT_DIR, "temp/obj2png.png"), fov, turbine_height, turbine_coord, turbine_origin)
        res_pic = generate_visual_impact(pic_class, turb_class, debug = False)
    return res_pic

def calc_wavelength(f, temp):
    c = 331.5 * np.sqrt(1 + temp / 273.15) # speed of sound in air [m/s]
    wavelength = c/f # [m]
    return wavelength

def calcAgr(f, dp, G, hs, hr):
    """
    Input:
        f : float
            frequency [Hz]
        dp : numpy.ndarray
            source-to-receiver distance array, in metres, projected onto the ground planes [m]
        G : float
            ground factor for source / receiver
        hs : float
            height of source above ground [m]
        hr : float
            height of receiver above ground [m]
    Output:
        A : numpy.ndarray
            ground attenuation contributions array [dB]
    """
    def calcA(f, dp, G, h):
        if f == 63:
            A = -1.5
        elif f == 125:
            aPrime = 1.5 + 3.0 * np.exp(-0.12*(h - 5)**2) * (1 - np.exp(-dp/50)) + 5.7 * np.exp(-0.09 * h**2) * (1 - np.exp(-2.8 * 10**(-6) * dp**2))
            A = -1.5 + G * aPrime
        elif f == 250:
            bPrime = 1.5 + 8.6 * np.exp(-0.09*h**2) * (1 - np.exp(-dp/50))
            A = -1.5 + G * bPrime
        elif f == 500:
            cPrime = 1.5 + 14.0 * np.exp(-0.46*h**2) * (1 - np.exp(-dp/50))
            A = -1.5 + G * cPrime
        elif f == 1000:
            dPrime = 1.5 + 5.0 * np.exp(-0.9*h**2) * (1 - np.exp(-dp/50))
            A = -1.5 + G * dPrime
        elif f in (2000, 4000, 8000):
            A = -1.5*(1 - G)
        else:
            print("Invalid nominal midband frequency!")
            A = np.zeros(dp)
        return A

    def calcAm(f, dp, G, hs, hr):
        """
        Input:
            f : float
                frequency [Hz]
            dp : numpy.ndarray
                source-to-receiver distance array, in metres, projected onto the ground planes [m]
            hs : float
                height of source above ground [m]
            hr: float
                height of receiver above ground [m]
        Output:
            Am : numpy.ndarray
                ground attenuation contributions for middle region array [dB]
        """
        q = np.where(dp <= 30 * (hs + hr), 0, 1 - (30 * (hs + hr))/dp)

        if f == 63:
            Am = -3 * q
        elif f in (125, 250, 500, 1000, 2000, 4000, 8000):
            Am = -3 * q * (1 - G)
        else:
            print("Invalid nominal midband frequency!")
            Am = np.zeros(dp)
        return Am

    As = calcA(f, dp, G, hs)
    Ar = calcA(f, dp, G, hr)
    Am = calcAm(f, dp, G, hs, hr)

    return As + Ar + Am

def solve_noise_map(elevation_handler, point_source_data, ground_factor = 0, temp = 15, rh = 70, receiver_height = 0):

    LDW = np.full_like(elevation_handler.map_array, -np.inf) # init res

    for i, point_source in point_source_data.iterrows(): # iterate over each point source

        # Lf = np.ones_like(elevation_handler.map_array)*-9999999 # init res
        Lf = np.full_like(elevation_handler.map_array, -np.inf)

        dp, d, dss, dsr, e = calc_diffraction_path(elevation_handler, point_source, receiver_height=receiver_height)
        
        z = (dss + e + dsr) - d

        Kmet = np.ones_like(z) # init Kmet
        mask_direct = z > 0 # mask direct path (True for diffraction)
        mask_multiple = e > 0 # mask multiple diffractions (True for multiple diffractions)
        Kmet[mask_direct] = np.exp(-(1/2000) * np.sqrt(dss[mask_direct] * dsr[mask_direct] * d[mask_direct] / (2 * z[mask_direct]))) # Calculate Kmet only where the mask is True

        hs = point_source.h
        hr = receiver_height

        for f, LW in point_source.octave_band.items(): # iterate over each octave-band sound power level
            f = int(f)
            wavelength = calc_wavelength(f, temp)

            C2 = 20
            C3 = np.ones_like(z) # init Kmet
            C3[mask_multiple] = (1 + (5*wavelength/e[mask_multiple])**2) / (1/3 + (5*wavelength/e[mask_multiple])**2)
            Dz = np.zeros_like(z)
            Dz[mask_direct] = np.minimum(10 * np.log10(3 + (C2 / wavelength) * C3[mask_direct] * z[mask_direct] * Kmet[mask_direct]), 25) # limit of 25 dB
            Dz[mask_direct & ~mask_multiple] = 20 # limit 20 dB for single diffraction

            Agr = calcAgr(f, dp, ground_factor, hs, hr)

            alpha = atmospheric_absorption(f, temp, rh, ps=1.01325e5)
            Aatm = alpha * d

            Adiv = 20 * np.log10(d) + 11

            Abar = Dz.copy()
            mask_Agr = Agr > 0
            Abar[mask_Agr] = Dz[mask_Agr] - Agr[mask_Agr]
            Abar = np.maximum(0, Abar)
            
            A = Adiv + Aatm  + Abar + Agr
            
            Lf = 10*np.log10(10**(0.1 * Lf) + 10**(0.1 * (LW - A + A_weighting(f))))
        LDW = 10*np.log10(10**(0.1 * LDW) + 10**(0.1 * Lf))
    return LDW

def calc_diffraction(source_height, receiver_height, terrain_x, terrain_y):
    def diffraction_recursion(terrain_x, terrain_y, xt_new, yt_new, start_i, ylp, diffraction_index):
        def calculate_line(xt, ylp):
            slope = (ylp[-1] - ylp[0]) / (xt[-1] - xt[0])
            intercept = ylp[0] - slope * xt[0]
            yl = slope * xt + intercept
            return yl

        # Calculate line profile
        yl = calculate_line(xt_new, ylp)
        diff = (yl - yt_new)

        # Check if the line collides with terrain
        if np.any(diff < -1e-5):
            yi_peak = np.argmin(diff) + start_i
            yimax = np.argmin(diff)  # Splitting index
            diffraction_index.append([terrain_x[yi_peak], terrain_y[yi_peak]])

            # Recursively process the left side of the splitting point
            xt0, yt0 = xt_new[:yimax], yt_new[:yimax]
            if len(xt0) > 2:
                ylp0 = calculate_line(xt0, [yl[0], yt_new[yimax]])
                diffraction_recursion(terrain_x, terrain_y, xt0, yt0, start_i, ylp0, diffraction_index)

            # Recursively process the right side of the splitting point
            xt1, yt1 = xt_new[yimax + 1:], yt_new[yimax + 1:]
            if len(xt1) > 2:
                start_i += yimax + 1
                ylp1 = calculate_line(xt1, [yt_new[yimax], yl[-1]])
                diffraction_recursion(terrain_x, terrain_y, xt1, yt1, start_i, ylp1, diffraction_index)

    # Initialize variables
    diffraction_index = [[0, terrain_y[0] + source_height]]
    ylp = [terrain_y[0] + source_height, terrain_y[-1] + receiver_height]
    xt_new, yt_new = terrain_x, terrain_y
    start_i = 0

    # Perform recursion to identify diffraction points
    diffraction_recursion(terrain_x, terrain_y, xt_new, yt_new, start_i, ylp, diffraction_index)

    # Sort and append the final point
    diffraction_index = np.array(diffraction_index)
    diffraction_index = diffraction_index[np.argsort(diffraction_index[:, 0])]
    diffraction_index = np.vstack((diffraction_index, np.array([terrain_x[-1], terrain_y[-1] + receiver_height])))

    return diffraction_index

def get_linecut(map_array, X, Y, start_point, end_point):
    def get_row_col(point, meshgrid_X, meshgrid_Y):
        if meshgrid_X.min() <= point[0] <= meshgrid_X.max() and meshgrid_Y.min() <= point[1] <= meshgrid_Y.max():
            pass
        else:
            raise ValueError('The input center is not within the given scope.')
        center_coord_row_col = np.unravel_index(np.argmin(np.abs(meshgrid_Y - point[1]) + np.abs(meshgrid_X - point[0])), meshgrid_Y.shape)
        return center_coord_row_col

    # Calculate row and column indices for start and end points
    start_row_col, end_row_col = get_row_col(start_point, X, Y), get_row_col(end_point, X, Y)
    start_row, start_col = np.asarray(start_row_col).astype(float)
    end_row, end_col = np.asarray(end_row_col).astype(float)

    # Calculate Euclidean distance between start and end points
    distance = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)

    # Determine the number of points for linecut
    num_points = int(np.sqrt((start_row_col[0] - end_row_col[0])**2 + (start_row_col[1] - end_row_col[1])**2))
    if num_points < 2:
        num_points = 2

    # Generate interpolated row and column indices
    interpolated_row = (np.linspace(start_row, end_row, num_points)).astype(int)
    interpolated_col = (np.linspace(start_col, end_col, num_points)).astype(int)

    # Generate distances along the linecut
    distances = np.linspace(0, distance, num_points)

    return distances, map_array[interpolated_row, interpolated_col]

def calc_extent(point_source_data, dist):
    '''This function calculates extent of map
    Inputs:
        point_source_data: point source data
        dist: dist to edge from centre
    '''
    lon_list, lat_list = point_source_data.longitude, point_source_data.latitude

    # boundary of wind farm
    bot_left_bound = np.array([np.amin(lon_list),np.amin(lat_list)]) 
    top_right_bound = np.array([np.amax(lon_list),np.amax(lat_list)])

    
    dist_cnr = np.sqrt(2*dist**2)
    bot_left = cgeo.Geodesic().direct(points=bot_left_bound,azimuths=225,distances=dist_cnr)[:,0:2][0]
    top_right = cgeo.Geodesic().direct(points=top_right_bound,azimuths=45,distances=dist_cnr)[:,0:2][0]
    
    extent = [bot_left[0], top_right[0], bot_left[1], top_right[1]]
    
    return extent

def calc_diffraction_path(elevation_handler, point_source_data, receiver_height = 0):
    """
    Input:
    - elevation_handler (ElevationHandler): An object containing elevation map and coordinate ranges.
    - point_source_data (PointSourceData): Information about the source point, including longitude, latitude, and height.

    Output:
    - dss (numpy.ndarray): Path length from the source to the first diffraction point.
    - dsr (numpy.ndarray): Path length from the receiver to the last diffraction point.
    - e (numpy.ndarray): Total path length from the first to the last diffraction point.
    """

    # Extract necessary data from elevation_handler and point_source_data
    map_array, long_range, lat_range = elevation_handler.map_array, elevation_handler.long_range, elevation_handler.lat_range
    longitude, latitude, source_height = point_source_data.longitude, point_source_data.latitude, point_source_data.h

    # Create meshgrid for coordinates
    X, Y = np.meshgrid(long_range, lat_range)

    # Transform coordinates to target CRS
    trans_cords = transform_coordinates(long_range, lat_range, input_crs_str=elevation_handler.crs, output_crs_str="EPSG:3035")
    # trans_cords = transform_coordinates(long_range, lat_range, input_crs_str="EPSG:4326", output_crs_str="EPSG:4326")

    # Flatten arrays for easy iteration
    map_array_flat = map_array.flatten()
    trans_long_flat = trans_cords[:, :, 0].flatten()
    trans_lat_flat = trans_cords[:, :, 1].flatten()

    # Initialize arrays for path lengths
    dp = np.empty(map_array_flat.shape)  # path length from source to first diffraction point
    d = np.empty(map_array_flat.shape)  # path length from source to first diffraction point
    dss = np.empty(map_array_flat.shape)  # path length from source to first diffraction point
    dsr = np.empty(map_array_flat.shape)  # path length from receiver to last diffraction point
    e = np.empty(map_array_flat.shape)    # path length from first to last diffraction

    # Initialize array for transformed source coordinates
    trans_source = np.empty(3)

    # Get elevation of the source point
    elevation_source = griddata((X.flatten(), Y.flatten()), map_array_flat, (longitude, latitude), method='linear')

    # Transform source coordinates
    trans_source[:2] = transform_coordinates(longitude, latitude, input_crs_str=elevation_handler.crs, output_crs_str="EPSG:3035")
    # trans_source[:2] = transform_coordinates(longitude, latitude, input_crs_str="EPSG:4326", output_crs_str="EPSG:4326")
    trans_source[2] = source_height + elevation_source

    # Loop over each point in the flattened coordinates
    for i in range(len(trans_long_flat)):
        # Define start and end points for the linecut
        start_point = trans_source[:2]
        end_point = [trans_long_flat[i], trans_lat_flat[i]]

        # Get linecut along the terrain
        dist_list, terrain_elevation = get_linecut(map_array, trans_cords[:, :, 0], trans_cords[:, :, 1], start_point, end_point)
        dp[i] = dist_list[-1]
        # Set source height and receiver height
        hs = source_height
        hr = receiver_height

        # Calculate diffraction path indices
        diffraction_index = calc_diffraction(hs, hr, dist_list, terrain_elevation)
        d[i] =  np.hypot(*(diffraction_index[0, :] - diffraction_index[-1, :]))

        # Calculate path lengths based on diffraction indices
        if len(diffraction_index) == 2:  # no diffraction
            dss[i] = 0
            dsr[i] = 0
            e[i] = 0
        elif len(diffraction_index) == 3:  # single diffraction
            dss[i] = np.hypot(*(diffraction_index[1, :] - diffraction_index[0, :]))
            dsr[i] = np.hypot(*(diffraction_index[-2, :] - diffraction_index[-1, :]))
            e[i] = 0
        else:  # multiple diffraction
            dss[i] = np.hypot(*(diffraction_index[1, :] - diffraction_index[0, :]))
            dsr[i] = np.hypot(*(diffraction_index[-2, :] - diffraction_index[-1, :]))
            e_elev = np.sum(np.abs(np.diff(diffraction_index[1:-1, 1])))  # change in diffraction elevation
            e_dist = diffraction_index[-2, 0] - diffraction_index[1, 0]
            e[i] = np.hypot(e_dist, e_elev) + e_elev


    # Reshape the arrays to the original shape
    dp = dp.reshape(map_array.shape)
    d = d.reshape(map_array.shape)
    dss = dss.reshape(map_array.shape)
    dsr = dsr.reshape(map_array.shape)
    e = e.reshape(map_array.shape)

    return dp, d, dss, dsr, e

def A_weighting(f):
       RAf = (12194**2 * f**4)/ ((f**2 + 20.6**2) * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))* (f**2 + 12194**2))
       Af = 20 * np.log10(RAf) + 2.00
       return Af

def atmospheric_absorption(f=1000, t=10, rh=80, ps=1.01325e5):
    """ 
    Calculate the attenuation coefficient for a given frequency, temperature, relative humidity, and atmospheric pressure.
    
    Input:
    f: frequency in Hz
    t: temperature in C
    rh: relative humidity in %
    ps: atmospheric pressure in Pa
    
    Output:
    alpha: attenuation coefficient
    """
    
    # Convert atmospheric pressure to a variable that won't be modified
    ps0 = 1.01325e5
    
    # Convert temperature from Celsius to Kelvin
    T = t + 273.15
    # Reference temperature in Kelvin
    T0 = 293.15
    
    # Reference temperature in Kelvin for saturation vapor pressure
    T01 = 273.16
    
    # Calculate saturation vapor pressure constant
    Csat = -6.8346 * (T01/T)**1.261 + 4.6151
    
    # Calculate saturation vapor pressure
    rhosat = 10**Csat
    
    # Calculate relative humidity ratio
    H = rhosat * rh * ps0 / ps
    
    # Calculate frequency-dependent term for attenuation due to water vapor
    frn = (ps / ps0) * (T0/T)**0.5 * (9 + 280 * H * np.exp(-4.17 * ((T0/T)**(1/3) - 1)))
    
    # Calculate frequency-dependent term for attenuation due to oxygen
    fro = (ps / ps0) * (24.0 + 4.04e4 * H * (0.02 + H) / (0.391 + H))
    
    # Calculate the attenuation coefficient
    alpha = 20/np.log(10) * f**2 * (1.84e-11 / ( (T0/T)**0.5 * ps / ps0 )+ (T/T0)**(-2.5)* (0.10680 * np.exp(-3352 / T) * frn / (f**2 + frn * frn)+ 0.01278 * np.exp(-2239.1 / T) * fro / (f**2 + fro * fro)))
    
    return alpha

def multiprocessing(func, points, sun_vec, minBound, maxBound, voxel_map, processes):
    pool = mp.Pool(processes=processes)
    results = []
    
    for process in range(processes):
        rays = sun_vec[process::processes]
        result = pool.apply_async(func, args=(points, rays, minBound, maxBound, voxel_map))
        results.append(result)
    
    pool.close()
    pool.join()
    
    shadow_map = np.zeros(voxel_map.shape[:2])
    
    for p, result in enumerate(results):
        try:
            temp_array = result.get()
            shadow_map += temp_array
        except Exception as e:
            print(f"Error occurred for process {p}: {e}")
    
    np.savetxt(Path.joinpath(ROOT_DIR, "temp/shadow_map.txt"), shadow_map)
    return shadow_map

def shadow_map_solver(elevation_handler, point_source_data, start_date = '2023-01-01 00:00:00', end_date = '2023-12-30 23:59:59', freq="10min", altitude_limit = np.deg2rad(5), processes = 1):
    
    # Generate solar vector #
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    sun_pos = np.zeros([len(date_range), 2])

    mean_longitude = np.mean(point_source_data.longitude)
    mean_latitude = np.mean(point_source_data.latitude)
    
    for i, date in enumerate(date_range):
        az, alt = solar_position(date, mean_latitude, mean_longitude)
        sun_pos[i,:] = az, alt

    sun_pos = sun_pos[sun_pos[:,1] > altitude_limit]
    sun_vec = np.zeros([len(sun_pos), 3]).astype(np.float32)

    for i, pos in enumerate(sun_pos):
        sun_vec[i,:] = -solar_angles_to_vector(*pos)
    ##################################################

    # Generate voxel map #
    X, Y, voxel_map, map_array = generate_voxel_map(elevation_handler.map_boundaries, elevation_handler.map_shape)

    grid_element_size = np.array([np.max(X) - np.min(X), np.max(Y) - np.min(Y), np.ceil(np.max(map_array)) - np.floor(np.min(map_array))]) / np.array(voxel_map.shape)
    # Generate turbine points #
    points = []
    for i, point_source in point_source_data.iterrows(): 
        diameter = point_source.d
        n_vector = angle_to_unit_vector(point_source.wind_dir)
        longitude, latitude, source_height = point_source.longitude, point_source.latitude, point_source.h

        trans_long, trans_lat = transform_coordinates(longitude, latitude, input_crs_str=elevation_handler.crs, output_crs_str="EPSG:3035")[0][0]

        elevation_source = griddata((X.flatten(), Y.flatten()),  map_array.flatten(), (trans_long, trans_lat), method='linear')

        turbine_cord = np.array([trans_long, trans_lat, elevation_source + source_height])
        angle = altitude_limit
        r_list, n_angle = rotor_point_spacing(diameter, grid_element_size, angle)
        points.append(generate_turbine(r_list, n_angle, n_vector, turbine_cord))
    points = np.concatenate(points).astype(np.float32)
    
    # SOLVE #
    minBound = np.array([np.min(X), np.min(Y), np.floor(np.min(map_array))], dtype = np.float32)
    maxBound = np.array([np.max(X), np.max(Y), np.ceil(np.max(map_array))], dtype = np.float32)
    cum_shadow_map = multiprocessing(solve_shadow_map.solve_shadow_map_cy, points, sun_vec, minBound, maxBound, voxel_map, processes)

    # FACTOR IN SAMPLING INTERVAL #
    date1 = date_range[0]
    date2 = date_range[1]
    time_difference = (date2 - date1).seconds

    cum_shadow_map *= time_difference/3600

    trans_cords = transform_coordinates(X[0,:], Y[:,0], input_crs_str="EPSG:3035", output_crs_str=elevation_handler.crs)
    cum_shadow_map[cum_shadow_map == 0] = np.nan
    return cum_shadow_map, trans_cords[:,:,0], trans_cords[:,:,1]

class Photo():
    def __init__(self, file, fov, theta, coord, location, focal_length):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.hfov = fov[0]
        self.theta = theta
        self.tilt = theta[0]
        self.direction = theta[2]
        self.coord = coord
        self.location = location
        self.shape = np.shape(self.im)
        self.vfov = fov[1]
        self.fov = fov
        self.hfov_range = [self.direction-self.hfov/2, self.direction+self.hfov/2]
        self.vfov_range = [self.tilt-self.vfov/2, self.tilt+self.vfov//2]
        self.focal_length = focal_length
        
class Turbine():
    def __init__(self, file, fov, height, coord, location):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.height = height
        self.coord = coord
        self.location = location
        self.shape = np.shape(self.im)
        self.width = fov[0]/self.shape[0] * self.shape[1]
        self.radius = height/(2*self.shape[0]/self.shape[1])

class ElevationHandlerTest:
    def __init__(self, map_array, map_boundaries, crs = "EPSG:3035"):
        self.map_array = map_array
        self.crs = crs
        self.map_shape = self.map_array.shape
        # self.map_boundaries = [0, self.map_array.shape[1], 0, self.map_array.shape[0]]
        self.map_boundaries = map_boundaries
        self.long_min = np.minimum(self.map_boundaries[0], self.map_boundaries[1])
        self.long_max = np.maximum(self.map_boundaries[0], self.map_boundaries[1])
        self.lat_min = np.minimum(self.map_boundaries[2], self.map_boundaries[3])
        self.lat_max = np.maximum(self.map_boundaries[2], self.map_boundaries[3])
        self.long_range = np.linspace(self.long_min, self.long_max, self.map_shape[1])
        self.lat_range = np.linspace(self.lat_min, self.lat_max, self.map_shape[0])

class ElevationHandler:
    def __init__(self, map_boundaries, map_shape, crs = "EPSG:4326"):
        """
        Input:
            map_boudaries : array_like
                boundaries of map using WGS84 coordinates. 
                    Example: [longitude_min, longitude_max, latitude_min, latitude_max]
            map_size : array_like
                size of output map in pixels. 
                    Example: [ncol, nrow]
        """
        self.map_boundaries = map_boundaries
        self.crs = crs
        self.map_shape = map_shape
        self.long_min = np.minimum(map_boundaries[0], map_boundaries[1])
        self.long_max = np.maximum(map_boundaries[0], map_boundaries[1])
        self.lat_min = np.minimum(map_boundaries[2], map_boundaries[3])
        self.lat_max = np.maximum(map_boundaries[2], map_boundaries[3])
        self.full_map_long_range = np.arange(np.floor(self.long_min), np.ceil(self.long_max), 1)
        self.full_map_lat_range = np.arange(np.floor(self.lat_min), np.ceil(self.lat_max), 1)
        self.full_map_boundaries = [np.floor(self.long_min), np.ceil(self.long_max), np.floor(self.lat_min), np.ceil(self.lat_max)]
        self.full_map = self.download_elevation()
        self.map_array = self.generate_scaled_subarray()
        self.long_range = np.linspace(self.long_min, self.long_max, self.map_shape[1])
        self.lat_range = np.linspace(self.lat_min, self.lat_max, self.map_shape[0])
        self.long_length = Geod(ellps='WGS84').inv(self.long_min, self.lat_min, self.long_max, self.lat_min)[2]
        self.lat_length = Geod(ellps='WGS84').inv(self.long_min, self.lat_min, self.long_min, self.lat_max)[2]

    def download_elevation(self):
        self.full_map = np.zeros([len(self.full_map_lat_range)*3601, len(self.full_map_long_range)*3601]) # init full_map
        for i, latitude in enumerate(self.full_map_lat_range):
            for j, longitude in enumerate(self.full_map_long_range):
                if latitude < 0:
                    lat_str = "S"+str(int(np.floor(-latitude))).zfill(2)
                else:
                    lat_str = "N"+str(int(np.floor(latitude))).zfill(2)
                    
                if longitude < 0:
                    long_str = "W"+str(int(np.floor(-longitude))).zfill(3)
                else:
                    long_str = "E"+str(int(np.floor(longitude))).zfill(3)

                output_name = f"{lat_str}{long_str}"
                hgt_gz_file = Path.joinpath(ROOT_DIR, "temp/"+output_name+".hgt.gz")
                hgt_file = Path.joinpath(ROOT_DIR, "temp/"+output_name+".hgt")

                if os.path.exists(hgt_file):
                    # print("File exists!")
                    pass
                else:
                    print("File does not exist.")

                    url = f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{lat_str}/{output_name}"+".hgt.gz"
                    
                    urllib.request.urlretrieve(url, hgt_gz_file)

                    with gzip.open(hgt_gz_file, 'rb') as f_in:
                        with open(hgt_file, 'wb') as f_out:
                            f_out.write(f_in.read())
                    os.remove(hgt_gz_file)
                
                with open(hgt_file, 'rb') as f:
                    data = np.frombuffer(f.read(), np.dtype('>i2')).reshape((3601, 3601))
                    data = np.flip(data, axis=0)

                self.full_map[i*3601:(i+1)*3601, j*3601:(j+1)*3601] = data
        return self.full_map
    
    def generate_scaled_subarray(self):
        x_old = np.linspace(self.full_map_boundaries[0], self.full_map_boundaries[1], self.full_map.shape[1])
        y_old = np.linspace(self.full_map_boundaries[2], self.full_map_boundaries[3], self.full_map.shape[0])

        interp_spline = RectBivariateSpline(y_old, x_old, self.full_map)
            
        x_new = np.linspace(self.map_boundaries[0], self.map_boundaries[1], self.map_shape[1])
        y_new = np.linspace(self.map_boundaries[2], self.map_boundaries[3], self.map_shape[0])

        self.scaled_subarray = interp_spline(y_new, x_new)
        return self.scaled_subarray
    
class Line():
    def __init__(self, startPoint, endPoint):
        self.startPoint = startPoint
        self.endPoint = endPoint

    def length_3d(self):
            """
            Calculate the 3D length of the line.
            """
            x1, y1, z1 = self.startPoint
            x2, y2, z2 = self.endPoint

            distance_3d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            return distance_3d

    def length_2d(self):
        """
        Calculate the projected 2D length of the line.
        """
        x1, y1, _ = self.startPoint
        x2, y2, _ = self.endPoint

        distance_2d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance_2d
    
def test_dir():
    return os.path.dirname(os.path.abspath(__file__))