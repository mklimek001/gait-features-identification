# code based on https://github.com/microsoft/multiview-human-pose-estimation-pytorch/blob/master/lib/multiviews/triangulate.py

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem


def parse_camera_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    camera_name = root.attrib['name']

    geometry = root.find('Geometry')
    intrinsic = root.find('Intrinsic')
    extrinsic = root.find('Extrinsic')
    
    camera_width = float(geometry.get('width'))
    camera_height = float(geometry.get('height'))

    f = float(intrinsic.get('focal'))
    c = np.array([
        float(intrinsic.get('cx')),
        float(intrinsic.get('cy'))
    ])

    k = np.array([
        float(intrinsic.get('kappa1')), 0.0, 0.0
        # k2=0 and k3=0 - not provided in camera config file
    ])

    p = np.array([0.0, 0.0])  # p - not provided in camera config file

    T = np.array([
        [float(extrinsic.get("tx"))],
        [float(extrinsic.get("ty"))],
        [float(extrinsic.get("tz"))]
    ]).reshape(3, 1)

    rx, ry, rz = float(extrinsic.get("rx")), float(extrinsic.get("ry")), float(extrinsic.get("rz"))
    R_matrix = R.from_euler('xyz', [rx, ry, rz]).as_matrix()

    return {
        "name": camera_name,
        "width": camera_width,
        "height": camera_height,
        "R": R_matrix,
        "T": T,
        "f": f,
        "c": c,
        "k": k,
        "p": p
    }


def unfold_camera_param(camera):
    """
    Camera parameters:
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    """
    R = camera['R']
    T = camera['T']
    f = camera['f'] 
    c = camera['c']
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def build_multi_camera_system(cameras):
    """
    Build a multi-camera system with pymvg package for triangulation

    Args:
        cameras: list of camera parameters
    Returns:
        cams_system: a multi-cameras system
    """
    pymvg_cameras = []
    for camera in cameras:
        R, T, f, c, k, p = unfold_camera_param(camera)
        camera_matrix = np.array(
            [[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]], dtype=float)
        distortion = np.array([k[0], k[1], p[0], p[1], k[2]])
        distortion.shape = (5,)
        M = camera_matrix.dot(np.concatenate((R, T), axis=1))
        camera = CameraModel.load_camera_from_M(
            M, name=camera['name'], distortion_coefficients=distortion,
            width=camera['width'], height=camera['height']
        )
        pymvg_cameras.append(camera)
    return MultiCameraSystem(pymvg_cameras)


def triangulate_one_point(camera_system, points_2d_set):
    """
    Triangulate 3d point in world coordinates with multi-views 2d points

    Args:
        camera_system: pymvg camera system
        points_2d_set: list of structure (camera_name, point2d)
    Returns:
        points_3d: 3x1 point in world coordinates
    """
    points_3d = camera_system.find3d(points_2d_set)
    return points_3d


def triangulate_poses(cameras_params, poses2d):
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        camera_params: a list of camera parameters, each corresponding to single camera
        poses2d: ndarray of shape nxkx2, len(cameras) == n
    Returns:
        poses3d: ndarray of shape n/nviews x k x 3
    """
    nviews = poses2d.shape[0]
    njoints = poses2d.shape[1]
    ninstances = 1 
    
    poses3d = []
    for i in range(ninstances):
        camera_system = build_multi_camera_system(cameras_params)

        pose3d = np.zeros((njoints, 3))
        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                camera_name = cameras_params[j]['name']
                points_2d = poses2d[i * nviews + j, k, :]
                points_2d_set.append((camera_name, points_2d))
            pose3d[k, :] = triangulate_one_point(camera_system, points_2d_set).T
        poses3d.append(pose3d)
    return np.array(poses3d)
