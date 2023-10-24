
def camera_matrix_old(photo, point):
    """
    INPUT:
        photo : background photo class
        point : 3D point is space
    """
    theta = np.array([np.pi, np.deg2rad(photo.tilt), -np.deg2rad(photo.direction)]) # the orientation of the camera [pitch, yaw, roll]. default look in z-direction 
    c = photo.location
    
    f = photo.focal_length # focal length [m]
    shape = photo.shape
    hfov = np.deg2rad(photo.hfov)
    vfov = np.deg2rad(photo.vfov)

    m_x = 1/((np.tan(hfov/2)*f)/(shape[1]/2))
    m_y = 1/((np.tan(vfov/2)*f)/(shape[0]/2))
    
    a_x = f * m_x; a_y = f * m_y

    R_x = np.array([[1,0,0],
                [0, np.cos(theta[0]), -np.sin(theta[0])],
                [0,np.sin(theta[0]),np.cos(theta[0])]])
    
    R_y = np.array([[np.cos(theta[1]),0,np.sin(theta[1])],
                    [0,1,0],
                    [-np.sin(theta[1]),0,np.cos(theta[1])]])
    
    R_z = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                      [np.sin(theta[2]), np.cos(theta[2]),0],
                      [0,0,1]])
    
    R = R_x @ R_y @ R_z 
    
    C_N  = R @ np.column_stack((np.identity(3),-c)) # camera matrix 
    
    
    K = np.array([[a_x, 0, shape[1]/2],
                  [0, a_y, shape[0]/2],
                  [0, 0, 1]])
    # print(f, m_x, m_y, a_x, a_y, K)
    P = np.hstack((point,1))
    
    p = K @ C_N @ P
    p = p / p[2]

    return p

def find_coeffs(pa, pb): # https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def generate_visual_impact_old(pic, turb):
    radius = turb.radius
    height = turb.height
    wind_dir = turb.wind_dir
    loc_turb = turb.location
    direction = pic.direction

    angle, distance = calc_angle_dist(pic.coord, turb.coord)

    # point1 = np.array([np.cos(wind_dir)*radius, height+radius, np.sin(wind_dir)*radius])+loc_turb
    # point2 = -np.array([np.cos(wind_dir)*radius, -height-radius, np.sin(wind_dir)*radius])+loc_turb
    # point3 = -np.array([np.cos(wind_dir)*radius, 0, np.sin(wind_dir)*radius])+loc_turb
    # point4 = np.array([np.cos(wind_dir)*radius, 0, np.sin(wind_dir)*radius])+loc_turb

    angle_diff = np.deg2rad(direction)

    point1 = np.array([np.cos(angle_diff)*radius, np.sin(angle_diff)*radius, height+radius])+loc_turb
    point2 = -np.array([np.cos(angle_diff)*radius, np.sin(angle_diff)*radius, -height-radius])+loc_turb
    point3 = -np.array([np.cos(angle_diff)*radius, np.sin(angle_diff)*radius, 0])+loc_turb
    point4 = np.array([np.cos(angle_diff)*radius, np.sin(angle_diff)*radius, 0])+loc_turb

    p_list = [point1, point2, point3, point4]
    pa = []
    for p in p_list:
        q = camera_matrix(pic, p, angle)
        if True:
            draw = ImageDraw.Draw(pic.im)
            draw.rectangle(((q[0],q[1]),(q[0]+5,q[1]+5)),fill = "red")
        pa.append([q[0],q[1]])

    pc = np.array(pa).reshape(4,2)
    pc[:,0] -= np.amin(pc[:,0])
    pc[:,1] -= np.amin(pc[:,1])

    pb = [(0, 0), (turb.shape[1], 0), (turb.shape[1], turb.shape[0]), (0, turb.shape[0])]

    pd = list(pa)
    for i in range(len(pa)):
        pa[i] = [pc[i,0],pc[i,1]]
        
    print(pa, pb)
    coeffs = find_coeffs(pa, pb)


    width, height = turb.im.size 
    turb.im = turb.im.transform((int(np.amax(np.array(pa).reshape(4,2)[:,0])),int(np.amax(np.array(pa).reshape(4,2)[:,1]))), method=Image.Transform.PERSPECTIVE,data=coeffs)
    vis_impact_im = pic.im.copy() 
    vis_impact_im.paste(turb.im, box=[np.amin(np.array(pd).reshape(4,2)[:,0]).astype(int),np.amin(np.array(pd).reshape(4,2)[:,1]).astype(int)], mask = turb.im)

    return vis_impact_im

def camera_matrix(photo, point): # spyder
    """
    INPUT:
        photo : background photo class
        point : 3D point is space
    """
    theta = np.array([np.deg2rad(photo.tilt),-np.deg2rad(photo.direction), np.pi]) # the orientation of the camera [pitch, yaw, roll]. default look in z-direction 
    c = photo.location # camera location
    a = point# the 3D position of a point A that is to be projected
    
    f = photo.focal_length # focal length [m]
    shape = photo.shape
    hfov = np.deg2rad(photo.hfov)
    vfov = np.deg2rad(photo.vfov)

    m_x = 1/((np.tan(hfov/2)*f)/(shape[1]/2))
    m_y = 1/((np.tan(vfov/2)*f)/(shape[0]/2))
    
    
    a_x = f * m_x; a_y = f * m_y
    
    R_z = np.array([[np.cos(theta[2]),-np.sin(theta[2]),0],
                      [np.sin(theta[2]), np.cos(theta[2]),0],
                      [0,0,1]])
    
    R_y = np.array([[np.cos(theta[1]),0,np.sin(theta[1])],
                      [0,1,0],
                      [-np.sin(theta[1]),0,np.cos(theta[1])]])
    R_x = np.array([[1,0,0],
                  [0, np.cos(theta[0]), -np.sin(theta[0])],
                  [0,np.sin(theta[0]),np.cos(theta[0])]])
    
    R = R_z @ R_y @ R_x
    
    # C_N  = np.column_stack((R,c)) # camera matrix
    C_N  = R @ np.column_stack((np.identity(3),-c)) # camera matrix
    
    K = np.array([[a_x, 0, shape[1]/2],
                  [0, a_y, shape[0]/2],
                  [0, 0, 1]])
    # print(f, m_x, m_y, a_x, a_y, K)
    P = np.hstack((point,1))
    
    p = K @ C_N @ P
    p = p / p[2]

    return p

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


def object_to_image(file_path, elevation = 0, azimuth = 0):
    # OLD CODE
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
    V = (V-(V.max(0)+V.min(0))/2) / max(V.max(0)-V.min(0))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0,0,1,1], xlim=[-1,+1], ylim=[-1,+1], aspect=1, frameon=False)
    ax = fig.add_subplot(111, projection="3d")
    plt.axis('off')
    ax.view_init(elev=elevation, azim=azimuth)
    plt.grid(visible=None)
    ax.plot_trisurf(V[:, 0], V[:,1], F, V[:, 2], linewidth=0, antialiased=True, closed=True, color = "w")
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    plt.savefig(f"../temp/obj2png.png", dpi=100, transparent=True)
    plt.close() # prevent plot from showing

    pil_image = Image.open(f"../temp/obj2png.png")
    pil_image = pil_image.crop((5, 5, pil_image.size[0]-5, pil_image.size[1]-5))
    np_array = np.array(pil_image)
    blank_px = [255, 255, 255, 0]
    mask = np_array != blank_px
    coords = np.argwhere(mask)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1
    cropped_box = np_array[x0:x1, y0:y1, z0:z1]
    pil_image = Image.fromarray(cropped_box, 'RGBA')
    pil_image.save(f"../temp/obj2png.png")
    return pil_image

def pull_street_view_image(api_key, longitude, latitude, fov = 90, heading = 0, pitch = 0, width = 800, height = 800):
# URL of the image you want to load
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

    tri = ax.plot_trisurf(V[:, 0], V[:,1], F, V[:, 2], linewidth=0, antialiased=False, closed=True)
    tri.set(facecolor = "grey", edgecolor = "none")

    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    lower_bound = min(0, 2*view_height - total_height)
    upper_bound = max(total_height, 2*view_height)
    limits[2,:] = [lower_bound, upper_bound]
    ax.set(zlim=limits[2,:], aspect="equal")

    plt.savefig(f"../temp/obj2png.png", dpi=300, transparent=True)
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


class Photo():
    def __init__(self, file, hfov, tilt, direction, coord, location, focal_length):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.hfov = hfov
        self.tilt = tilt
        self.direction = direction
        self.coord = coord
        self.location = location
        self.shape = np.shape(self.im)
        self.vfov = hfov/self.shape[1] * self.shape[0]
        self.hfov_range = [direction-hfov/2, direction+hfov/2]
        self.vfov_range = [tilt-self.vfov/2, tilt+self.vfov//2]
        self.focal_length = focal_length
        
class Turbine():
    def __init__(self, file, fov, height, coord, location, wind_dir):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.height = height
        self.coord = coord
        self.location = location
        self.shape = np.shape(self.im)
        self.width = fov/self.shape[0] * self.shape[1]
        self.radius = height/(2*self.shape[0]/self.shape[1]-1)
        self.wind_dir = wind_dir