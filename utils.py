import numpy as np
import cv2
import argparse
import glob
from scipy.spatial.transform import Rotation as scipyRot
from scipy.optimize import least_squares

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def convert_A_matrix_to_vector(A):
    return np.array([A[0,0], A[0,1], A[1,1], A[0,2], A[1,2]])

def convert_A_vector_to_matrix(a):
    alpha, gamma, beta, u0, v0 = a
    A1 = [alpha, gamma, u0]
    A2 = [0, beta, v0]
    A3 = [0, 0, 1]

    A = np.vstack((A1, A2, A3))
    return A

def dissect_x_vector(x, n_imgs):
    A = convert_A_matrix_to_vector(x[0:5])
    k1 = x[5]
    k2 = x[6]
    transformations = x[7:].reshape(n_imgs, 6)
    return A, k1, k2, transformations

def package_x_vector(A, k1, k2, transformations):
    a = convert_A_matrix_to_vector(A)
    x0 = np.concatenate((a, k1, k2, transformations))
    return x0

def draw_circles(img, base_path, name, m_new, m_old):
    for x,y,_ in m_new.T:
        cv2.circle(img, (int(x), int(y)), 0, (255,0,0), 30)
    for x,y in m_old:
        cv2.circle(img,(int(x), int(y)), 0, (0,255,0), 15)
    cv2.imwrite(f"{base_path}/rectified/{name}.png", img)

def get_images(base_path, input_extn):
    img_files = glob.glob(f"{base_path}/*{input_extn}", recursive = False)
    img_names = [img_file.replace(f"{base_path}/", '').replace(f"{input_extn}",'')for img_file in img_files]
    imgs = [cv2.imread(img_files) for img_file in img_files]
    return imgs, img_names

def get_chessboard_corners(img_color, pattern_size, name, args):
    if args.debug:
        cv2.imshow(name, img_color)

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    if args.debug:
        cv2.imshow(f"{name}_gray", img_gray)

    chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH  \
                        + cv2.CALIB_CB_NORMALIZE_IMAGE \
                        + cv2.CALIB_CB_FAST_CHECK
    
    ret, corners = cv2.findChessboardCorners(img_gray, pattern_size, flags=chessboard_flags)

    if not ret:
        print(f"something went wrong while provessing{name}")
        exit(1)
    if args.display:
        chessboard_img = cv2.drawChessboardCorners(img_color, pattern_size, corners, ret)
        cv2.imshow(f"{name}_chessboard", chessboard_img)

    corners = corners.reshape((corners.shape[0], -1))
    return corners

def get_world_corners(pattern_size, square_size):
    """
    description:
        returns world corners for a given pattern size and square size(mm)
    input:
        pattern_size - tuple (2)
        square_size - scalar (mm)
    output:
        world_corners - pattern_size[0]*pattern_size[1] x 2
    """
    x_lin = np.arange(0, pattern_size[0],1)
    y_lin =  np.arange(0,pattern_size[1],1)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    world_corners = np.vstack((x_grid, y_grid)).T
    world_corners = world_corners*square_size
    return world_corners


def get_V_mat_element(H,i,j):
    """
    description:
        calculate element of v vector from homography
    input:
        H - homography matrix 3x3

    """
    i = i-1 #converting paper convention to numpy
    j = j-1

    v1 = H[0][i]*H[0][j]
    v2 = H[0][i]*H[1][j] + H[1][i]*H[0][j]
    v3 = H[1][i]*H[1][j]
    v4 = H[2][i]*H[0][j] + H[0][i]*H[2][j]
    v5 = H[2][i]*H[1][j] + H[1][i]*H[2][j]
    v6 = H[2][i]*H[2][j]
    v = np.vstack((v1,v2,v3,v4,v5,v6))
    return v



