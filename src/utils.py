from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import pyproj
from pyproj import Geod
import os
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from subprocess import check_output
import inspect
from IPython.display import display

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
        azimuth (float): Solar azimuth angle in degrees
        altitude (float): Solar altitude angle in degrees

    Returns:
        vec (np.ndarray): Normalized 3D vector representing sun direction from origin
    """
    # Convert to radians
    azimuth_rad = np.radians(azimuth+90)
    altitude_rad = np.radians(altitude)

    # Calculate 3D vector components
    x = -np.cos(azimuth_rad) * np.cos(altitude_rad)
    y = np.sin(azimuth_rad) * np.cos(altitude_rad)
    z = np.sin(altitude_rad)

    # Normalize to unit vector
    vec = np.array([x, y, z])
    vec /= np.linalg.norm(vec)

    return vec


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