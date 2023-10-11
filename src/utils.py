
def camera_matrix(photo, point):
    """
    INPUT:
        photo : background photo class
        point : 3D point is space
    """
    theta = np.array([np.deg2rad(photo.tilt),-np.deg2rad(photo.direction),np.pi]) # the orientation of the camera [pitch, yaw, roll]. default look in z-direction 
    c = photo.loc_pic
    a = point # the 3D position of a point A that is to be projected
    
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
    t = c
    
    C_N  = np.column_stack((R,t)) # camera matrix
    
    
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

from PIL import Image
import numpy as np
import pyproj
from pyproj import Geod


class Photo():
    def __init__(self, file, hfov, tilt, direction, loc_pic, focal_length):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.hfov = hfov
        self.tilt = tilt
        self.direction = direction
        self.loc_pic = loc_pic
        self.shape = np.shape(self.im)
        self.vfov = hfov/self.shape[1] * self.shape[0]
        self.hfov_range = [direction-hfov/2, direction+hfov/2]
        self.vfov_range = [tilt-self.vfov/2, tilt+self.vfov//2]
        self.focal_length = focal_length
        
class Turbine():
    def __init__(self, file, height, loc_turb, wind_dir):
        self.im = Image.open(file, mode = "r").convert('RGBA')
        self.height = height
        self.loc_turb = loc_turb
        self.shape = np.shape(self.im)
        self.width = 90/self.shape[0] * self.shape[1] # hard code 90 deg. Fix later
        self.radius = height/(2*self.shape[0]/self.shape[1]-1)
        self.wind_dir = wind_dir
 