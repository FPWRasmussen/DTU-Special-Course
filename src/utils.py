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

    # calculate turbine direction
    # e_y = np.array([0, 1])
    # angle_vector = (turbine_origin - camera_origin)[:2]
    # direction = np.arccos((e_y @ angle_vector)/(np.linalg.norm(angle_vector)))
    # print(np.rad2deg(direction))

    # point1 = np.array([-np.sin(direction)*radius, np.cos(direction)*radius, 0]) + turbine_origin
    # point2 = np.array([np.sin(direction)*radius, -np.cos(direction)*radius, 0]) + turbine_origin
    # point3= np.array([-np.sin(direction)*radius, np.cos(direction)*radius, height+radius]) + turbine_origin
    # point4 = np.array([np.sin(direction)*radius, -np.cos(direction)*radius, height+radius]) + turbine_origin

    # point1 = np.array([-np.cos(direction)*radius, -np.sin(direction)*radius, 0]) + turbine_origin
    # point2 = np.array([np.cos(direction)*radius, np.sin(direction)*radius, 0]) + turbine_origin
    # point3= np.array([-np.cos(direction)*radius, -np.sin(direction)*radius, height+radius]) + turbine_origin
    # point4 = np.array([np.cos(direction)*radius, np.sin(direction)*radius, height+radius]) + turbine_origin

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
    print(pc)
    pc[:,0] -= np.amin(pc[:,0])
    pc[:,1] -= np.amin(pc[:,1])

    pb = [(turb_class.shape[1], turb_class.shape[0]), (0, turb_class.shape[0]),  (turb_class.shape[1], 0), (0, 0)]

    pd = list(pa)
    for i in range(len(pa)):
        pa[i] = [pc[i,0],pc[i,1]]
        
    coeffs = find_coeffs(pa, pb)

    turb_class.im = turb_class.im.transform((int(np.amax(np.array(pa).reshape(4,2)[:,0])),int(np.amax(np.array(pa).reshape(4,2)[:,1]))), method=Image.Transform.PERSPECTIVE,data=coeffs)
    pic_class.im.paste(turb_class.im, box=[np.amin(np.array(pd).reshape(4,2)[:,0]).astype(int),np.amin(np.array(pd).reshape(4,2)[:,1]).astype(int)], mask = turb_class.im)

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

def transform_coordinates(long_list, lat_list, input_crs_str = "EPSG:4326", output_crs_str = "EPSG:3035"):
    
    input_crs = pyproj.CRS(input_crs_str)  # WGS84
    output_crs = pyproj.CRS(output_crs_str)

    transformer = pyproj.Transformer.from_crs(input_crs, output_crs, always_xy=True)

    if isinstance(lat_list, int) or isinstance(lat_list, float):
        lat_list, long_list = [lat_list], [long_list]

    trans_cords = np.empty([len(lat_list),len(long_list), 2])

    for i, lon in enumerate(long_list):
        for j, lat in enumerate(lat_list):
            x, y = transformer.transform(lon, lat)
            trans_cords[j, i, 0] = x
            trans_cords[j, i, 1] = y

    return trans_cords

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

    img.save(f"../temp/site_img.png")

    return img

def adjust_image(image_path = "../temp/obj2png.png", brightness = 1, contrast = 1, display_image = True):
    # Load the image using PIL.
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path

    # Apply the brightness and contrast adjustments to the image.
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    # Save or display the adjusted image.
    image.save("../temp/obj2png.png")
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
    lower_bound = min(0, 2*view_height - total_height)
    upper_bound = max(total_height, 2*view_height)
    limits[2,:] = [lower_bound, upper_bound]
    ax.set(zlim=limits[2,:], aspect="equal")

    plt.savefig(f"../temp/obj2png.png", dpi=600, transparent=True)
    if show_plot:
        plt.show()
    else:
        plt.close() # prevent plot from showing

def crop_image(file_path = "../temp/obj2png.png", display_image = True):
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
    pil_image = crop_image(file_path = "../temp/obj2png.png", display_image = debug)
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
    # Convert to radians
    azimuth = azimuth+np.pi/2

    # Calculate 3D vector components
    x = -np.cos(azimuth) * np.cos(altitude)
    y = np.sin(azimuth) * np.cos(altitude)
    z = np.sin(altitude)

    # Normalize to unit vector
    vec = np.array([x, y, z])
    vec /= np.linalg.norm(vec)

    return vec

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
    # if az < 0:
    #     az += 2 * np.pi 
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
            hgt_gz_file = "../../temp/"+output_name+".hgt.gz"
            hgt_file = '../../temp/'+ output_name+ '.hgt'

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

    trans_cords = np.empty([len(lat_list),len(long_list), 2])

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
    voxel_map = np.zeros((map_array.shape[0], map_array.shape[1], len(elevation_range)), dtype=bool)
    for i, elev in enumerate(elevation_range):
        voxel_map[:, :, i] = map_array > elev
    return X, Y, voxel_map, map_array

def solve_shadow_map_old(ray_point, ray_vec, grid_element_size, terrain_voxel_map, verbose=True):
    terrain_voxel_map_shape = terrain_voxel_map.shape
    temp_shadow_map = np.zeros(terrain_voxel_map_shape[0:2], dtype=bool)
    cum_shadow_map = np.zeros(terrain_voxel_map_shape[0:2], dtype=int)

    terrain_max_elevation = terrain_voxel_map_shape[2] - 1

    # for z in range(terrain_voxel_map_shape[2]):
    #     if np.any(terrain_voxel_map[:, :, z]):
    #         terrain_max_elevation = z
    # terrain_max_elevation *= grid_element_size[2]
    print("lol")
    for i in range(ray_vec.shape[0]):
        ray = ray_vec[i, :]

        for j in range(ray_point.shape[0]):
            point = ray_point[j, :]
            if terrain_max_elevation < point[2]:  # optimize for elevation
                x0, y0, z0 = point
                A, B, C = ray
                t = (terrain_max_elevation - z0) / C
                x = x0 + A * t
                y = y0 + B * t
                point = np.array([x, y, terrain_max_elevation])

            cur_vox = np.floor(point / grid_element_size).astype(int)

            # if new start is out of bounds
            if cur_vox[0] >= terrain_voxel_map_shape[0] or cur_vox[0] < 0:
                continue
            elif cur_vox[1] >= terrain_voxel_map_shape[1] or cur_vox[1] < 0:
                continue
            elif cur_vox[2] >= terrain_voxel_map_shape[2] or cur_vox[2] < 0:
                continue
            
            boxSize = terrain_voxel_map_shape * grid_element_size

            step = np.ones(3)
            tVoxel = np.empty(3)

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

            voxelMax = tVoxel * boxSize
            tMax = (voxelMax - point) / ray
            voxelSize = boxSize / terrain_voxel_map_shape
            tDelta = voxelSize / abs(ray)

            while True:
                if verbose:
                    print(f"Intersection: voxel = ({cur_vox[0]}, {cur_vox[1]}, {cur_vox[2]})")
                if tMax[0] < tMax[1]:
                    if tMax[0] < tMax[2]:
                        cur_vox[0] += step[0]
                        if (cur_vox[0] >= terrain_voxel_map_shape[0]) or (cur_vox[0] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[0] += tDelta[0]
                    else:
                        cur_vox[2] += step[2]
                        if cur_vox[2] >= terrain_voxel_map_shape[2] or (cur_vox[2] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[2] += tDelta[2]
                else:
                    if tMax[1] < tMax[2]:
                        cur_vox[1] += step[1]
                        if cur_vox[1] >= terrain_voxel_map_shape[1] or (cur_vox[1] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[1] += tDelta[1]
                    else:
                        cur_vox[2] += step[2]
                        if cur_vox[2] >= terrain_voxel_map_shape[2] or (cur_vox[2] < 0):
                            break
                        if terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[2] += tDelta[2]
            cum_shadow_map[temp_shadow_map] += 1
            temp_shadow_map = np.zeros(terrain_voxel_map_shape[0:2], dtype=bool)
    return cum_shadow_map

def solve_shadow_map(ray_point, ray_vec, grid3D, terrain_voxel_map):
    terrain_voxel_map_shape = terrain_voxel_map.shape
    temp_shadow_map = np.zeros(terrain_voxel_map_shape[0:2], dtype=bool)
    cum_shadow_map = np.zeros(terrain_voxel_map_shape[0:2], dtype=int)

    terrain_max_elevation = grid3D['maxBound'][2] - 1

    for i in range(ray_vec.shape[0]):
        ray = ray_vec[i, :]

        for j in range(ray_point.shape[0]):
            point = ray_point[j, :]
            if terrain_max_elevation < point[2]:  # optimize for elevation
                x0, y0, z0 = point
                A, B, C = ray
                t = (terrain_max_elevation - z0) / C
                x = x0 + A * t
                y = y0 + B * t
                point = np.array([x, y, terrain_max_elevation])
            boxSize = grid3D['maxBound'] - grid3D['minBound']
            cur_vox = np.floor(((point - grid3D['minBound']) / boxSize) * terrain_voxel_map_shape).astype(int)

            if cur_vox[0] >= terrain_voxel_map_shape[0] or cur_vox[0] < 0:
                continue
            elif cur_vox[1] >= terrain_voxel_map_shape[1] or cur_vox[1] < 0:
                continue
            elif cur_vox[2] >= terrain_voxel_map_shape[2] or cur_vox[2] < 0:
                continue
            
            step = np.ones(3)
            tVoxel = np.empty(3)

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

            voxelMax = grid3D['minBound']+ tVoxel*boxSize

            tMax     = (voxelMax - point) / ray
            voxelSize = boxSize/ terrain_voxel_map_shape
            tDelta = voxelSize / abs(ray)

            while True:
                if tMax[0] < tMax[1]:
                    if tMax[0] < tMax[2]:
                        cur_vox[0] += step[0]
                        if (cur_vox[0] >= terrain_voxel_map_shape[0]) or (cur_vox[0] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[0] += tDelta[0]
                    else:
                        cur_vox[2] += step[2]
                        if cur_vox[2] >= terrain_voxel_map_shape[2] or (cur_vox[2] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[2] += tDelta[2]
                else:
                    if tMax[1] < tMax[2]:
                        cur_vox[1] += step[1]
                        if cur_vox[1] >= terrain_voxel_map_shape[1] or (cur_vox[1] < 0):
                            break
                        elif terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[1] += tDelta[1]
                    else:
                        cur_vox[2] += step[2]
                        if cur_vox[2] >= terrain_voxel_map_shape[2] or (cur_vox[2] < 0):
                            break
                        if terrain_voxel_map[cur_vox[0], cur_vox[1], cur_vox[2]]:
                            temp_shadow_map[cur_vox[1], cur_vox[0]] = True
                            break
                        tMax[2] += tDelta[2]
            cum_shadow_map[temp_shadow_map] += 1
            temp_shadow_map = np.zeros(terrain_voxel_map_shape[0:2], dtype=bool)
    return cum_shadow_map


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
    def __init__(self, file, fov, height, coord, location, wind_dir):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.height = height
        self.coord = coord
        self.location = location
        self.shape = np.shape(self.im)
        self.width = fov[0]/self.shape[0] * self.shape[1]
        self.radius = height/(2*self.shape[0]/self.shape[1])
        self.wind_dir = wind_dir