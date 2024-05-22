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

def get_V_mat(H):
    """
    description:
        calculate V for a given homograply
    input:
        H - homography matrix 3x3
    output:
        V - 2x6 V matrix
    """
    V1 = get_V_mat_element(H,1,2)
    V1 = V1.T
    V20 = get_V_mat_element(H,1,1)
    V20 = V20.T
    V21 = get_V_mat_element(H,2,2)
    V21 = V21.T
    V2 = V20 - V21
    V = np.vstack((V1,V2))

    return V

def get_L_mat(img_corner, world_corner):
    """
    description:
         calcilate L for a given img_corner and world_corner
    input:
        immage_corner - 2,
        world_corners - 3,
    output:
        L - as per pap convention 2x9
    """
    L1 = np.hstack((world_corner, np.zeros(3), -img_corner[0]*world_corner))
    L2 = np.hstack((np.zeros(3), world_corner, -img_corner[1]*world_corner))
    L = np.vstack((L1, L2))
    return L

def get_homography(img_corners, world_corners, name):
    world_corners = np.hstack((world_corners, np.ones((world_corners.shape[0], 1))))
    L = tuple([get_L_mat(img_corner, world_corner) for img_corner, world_corner in zip(img_corners, world_corners)])
    L = np.vstack(L)
    eig_val, eig_vec = np.linalg.eig(L.T @ L)
    min_eig_vec_ind = np.argmin(eig_val)
    min_eig_vec = eig_vec[:,min_eig_vec_ind]

    h1 = min_eig_vec[0:3]
    h2 = min_eig_vec[3:6]
    h3 = min_eig_vec[6:9]

    H = np.vstack((h1, h2, h3))
    H = H/H[2,2]

    return H

def get_camera_intrinsic_from_b(b):
    """
    description:
        retiirn camera intrinstics given b vector from paper
    input:
        b - vector as per convention from paper
    output:
        camera intrinsic matix 3x3
    """
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    v0_num = B12*B13 - B11*B23
    v0_den = B11*B22 - B12*B12
    v0 = v0_num/v0_den

    lambda1_num = B13*B13 + v0*(B12*B13 - B11*B23)
    lamda = B33 - lambda1_num/B11
    
    alpha = (lamda/B11)**(0.5)

    beta_num = lamda*B11
    beta_den = B11*B22 - B11*B12
    beta = (beta_num/beta_den)**(0.5)

    gamma = (-B12*alpha*alpha*beta)/lamda

    u00 = (gamma*v0)/beta
    u01 = (B13*alpha*alpha)/lamda
    u0 = u00 - u01

    A = convert_A_vector_to_matrix([alpha, gamma, beta, u0, v0])
    return A, lamda