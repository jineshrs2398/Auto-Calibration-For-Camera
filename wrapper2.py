import numpy as np
import cv2
import argparse
import glob
from scipy.spatial.transform import Rotation as scipyRot
from scipy.optimize import least_squares
from utils import *

def test_homography(imgs,imgs_names,homography_list):
    for i,H in enumerate(homography_list):
        img1_wrap = cv2.wrapPerspective(imgs[i],H,[imgs[i].shape[1],
                                                   imgs[i].shape[0]])
        cv2.imshow(f"warp1_{imgs_names[i]}",img1_wrap)

def main(args):
    base_path = args.basePath
    input_extn = ".jpg"
    square_size = 21.5
    pattern_size = (9,6)

    # Load images
    imgs,imgs_names = get_images(base_path,input_extn)
    # Get world coordinates
    world_corners = get_world_corners(pattern_size,square_size)
    imgs_corners = [get_chessboard_corners(img,pattern_size,name,args) for img,name in zip(imgs,imgs_names)]
    homography_list = [get_homography(img_corners,world_corners,name) for img_corners,name in zip(imgs_corners,imgs_names)]

    A_initial, lamda_initial = get_camera_intrinsics(homography_list)

    transformations_initial = tuple([get_transformation_mat(A_initial,lamda_initial,H) for H in homography_list])
    transformations_initial = np.concatenate(transformations_initial)

    k1_initial = np.array([0])
    k2_initial = np.array([0])
    x0 = package_x_vector(A_initial,k1_initial,k2_initial,transformations_initial)

    kwargs1 = {
                "imgs_corners": imgs_corners,
                "world_corners":world_corners,
                "per_img":False,
            }
    result = least_squares(compute_residuals,x0=x0,method='lm',kwargs=kwargs1)

    n_imgs = len(imgs)
    A_final,k1_final,k2_final,transformations_final = dissect_x_vector(result.x,n_imgs)
    
    distorted_imgs = [inverse_warp(img,A_final,k1_final,k2_final) for img in imgs]

    projection_errors_inital = compute_residuals(x0,imgs_corners,world_corners,True)
    projection_errors_final = compute_residuals(result.x,imgs_corners,world_corners,True)

    projection_errors_initial = [error/world_corners.shape[0] for error in projection_errors_initial]
    projection_errors_final = [error/world_corners.shape[0] for error in projection_errors_final]

    projections_final = get_projections(transformations_final,A_final,k1_final,k2_final,world_corners)

    for i, projection_final in enumerate(projections_final):
        draw_circles(distorted_imgs[i],base_path,imgs_names[i],projection_final,imgs_corners[i])

    print("----summary----")
    print(f"A_initial:\n{A_initial}")
    print(f"k1_initial:{k1_initial}")
    print(f"k2_initial:{k2_initial}")
    print(f"projection_errors_initial:\n{projection_errors_final}")
    print(f"A_final:\n{A_final}")
    print(f"k1_final:{k1_final}")
    print(f"k2_final:{k2_final}")
    print(f"projection_errors_final:\n{projection_errors_final}")

    if args.debug:
        test_homography(imgs,imgs_names,homography_list)
    if args.display:
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--basePath',default='../data/Calibration_Imgs')
    parser.add_argument('--display',action='store_true',help="to display images")
    parser.add_argument('--debug',action='store_true',help="to display debug information")

    args = parser.parse_args()
    main(args)