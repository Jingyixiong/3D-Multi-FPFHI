'''
Render point cloud into images. The correspondance between
point cloud and rendered images are documented. 
The code is modified based on:
https://github.com/caoyunkang/CPMF
'''

import copy
import os
import cv2
import numpy as np
import open3d as o3d
import json
from sklearn.decomposition import PCA

from patchcore.help_func import voxelization

class MultiViewRender():
    COLOR_X = o3d.visualization.PointColorOption.XCoordinate
    COLOR_Y = o3d.visualization.PointColorOption.YCoordinate
    COLOR_Z = o3d.visualization.PointColorOption.ZCoordinate
    COLOR_NORM = o3d.visualization.PointColorOption.Normal
    COLOR_FPFH = 4
    COLOR_UNIFORM = 5
    COLOR_RGB = 6
    def __init__(self, voxel_size, focal, extrinsic,
                 x_angles=[0., -np.pi / 12, np.pi / 12],
                 y_angles=[0., -np.pi / 12, np.pi / 12],
                 z_angles=[0., -np.pi / 12, np.pi / 12],
                 img_res=512, 
                 color=None):
        '''
        Initialize a multi view render for data process
        Args:
            parameters_path: the path to camera parameters
            x_angles: the angles we would like to rotate, be sure the first of x_angles is 0
            y_angles: the angles we would like to rotate, be sure the first of y_angles is 0
            z_angles: the angles we would like to rotate, be sure the first of z_angles is 0
            color: to be added further. Control the rendered color of images.
        '''
        assert x_angles[0] == 0
        assert y_angles[0] == 0
        assert z_angles[0] == 0

        super(MultiViewRender, self).__init__()

        self.voxel_size = voxel_size
        self.W, self.H = img_res, img_res   # fixed to 512
        self.focal = focal          # focal length, to get the largest reception field
        self.extrinsic = extrinsic  # provided extrinsic parameters to fix params
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.W, height=self.H, visible=False)
        self.angles = self.get_viewpoints(x_angles, y_angles, z_angles)

        self.color_option = color
        if color in [self.COLOR_X, self.COLOR_Y, self.COLOR_Z, self.COLOR_NORM]:
            self.vis.get_render_option().point_color_option = color
        elif color == self.COLOR_FPFH: # suggest to calculate color outsize this class
            self.pca = PCA(n_components=3)

    def read_camera_parameters(self, path):
        '''
        Read the camera parameters of mvtec3d category.
        Args:
            path:

        Returns:

        '''
        with open(path, 'r') as f:
            parameters = json.load(f)
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(parameters['image_width'], parameters['image_height'],
                                                             1. / parameters['focus'], 1. / parameters['focus'],
                                                             parameters['cx'], parameters['cy'])
        return camera_intrinsic, parameters['image_height'], parameters['image_width']

    def rotate_render(self, pcd, rotate_angle, ref_points):
        '''
        Rotate a point cloud with the desired angle and then render it to image
        Args:
            pcd:
            rotate_angle:

        Returns:

        '''
        # rotate pcd
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotate_angle)
        pcd_temp = copy.deepcopy(pcd)
        # rotate the point cloud based on give rotational matrix R
        # and the origin of rotation 'pcd_temp.get_center()'
        pcd_temp.rotate(R, pcd_temp.get_center())  
        ref_points_temp = copy.deepcopy(ref_points)
        ref_points_temp.rotate(R, ref_points_temp.get_center())

        vis = self.vis
        # render and calculate 3d to 2d pairs
        vis.add_geometry(pcd_temp)
        # change the point size
        vis.get_render_option().point_size = 1
        
        # adjust the position of camera(it works only on lower version of open3d)
        new_intrinsics = self.adjust_camera_pos(vis)
        image = vis.capture_screen_float_buffer(do_render=True)
        points2d = calculate_points2d(
            new_intrinsics,
            self.extrinsic, 
            np.asarray(ref_points_temp.points).T)
        vis.clear_geometries()

        # convert to rgb
        image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)

        return image, points2d, new_intrinsics

    def adjust_camera_pos(self, vis):
        '''
        Adjust the camera parameters (intrinsic and extrinsic).
        '''
        ctr = vis.get_view_control()       # view controller
        init_param = ctr.convert_to_pinhole_camera_parameters()
        cx, cy = self.W/2, self.H/2
        # new intrinsic params 
        new_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=self.W, height=self.H, 
            fx=self.focal, fy=self.focal, 
            cx=cx, cy=cy
            )  # intrinsic parameter setting 
        init_param.intrinsic = new_intrinsics
        init_param.extrinsic = self.extrinsic
        ctr.convert_from_pinhole_camera_parameters(
            init_param, 
            allow_arbitrary=True
            )
        return new_intrinsics.intrinsic_matrix

    def get_viewpoints(self, x_angles, y_angles, z_angles):
        '''
        Get the full angle list of all viewpoints.
        Args:
            x_angles:
            y_angles:
            z_angles:

        Returns:

        '''
        angles = []
        for x in x_angles:
            for y in y_angles:
                for z in z_angles:
                    angles.append([x, y, z])
        return angles

    def multiview_render(self, xyzs:np.ndarray, masks, rgb, ref_points, binary_masks):
        '''
        Render a point cloud with the selected viewpoints.
        Args:
            pcd:

        Returns:

        '''
        assert xyzs.shape[0] == rgb.shape[0] == ref_points.shape[0]
        assert rgb is not None
        print('The orginal size of point cloud is {}.'.format(xyzs.shape[0]))
        # points downsampling
        binary_masks = np.expand_dims(binary_masks,axis=1) if binary_masks.ndim==1 else binary_masks
        masks = np.expand_dims(masks,axis=1) if masks.ndim==1 else masks
        points = np.concatenate([xyzs, rgb, ref_points, binary_masks, masks], axis=1)
        if self.voxel_size:
            sub_points, _ = voxelization(points, self.voxel_size)
        image_list, point2d_list = [], []
        xyzs, rgb, ref_xyzs, binary_masks, masks = sub_points[:, :3], \
                                     sub_points[:, 3:6],\
                                     sub_points[:, 6:9], \
                                     sub_points[:, 9], \
                                     sub_points[:, 10:]
        binary_masks = np.expand_dims(binary_masks,axis=1) if binary_masks.ndim==1 else binary_masks
        masks = np.expand_dims(masks,axis=1) if masks.ndim==1 else masks
        print('It becomes {} after downsampling.'.format(xyzs.shape[0]))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_xyzs)
        ref_pcd.colors = o3d.utility.Vector3dVector(rgb)
        # Rendering multi-images
        for angle in self.angles:
            image, points2d, intrinsics = self.rotate_render(pcd, angle, ref_pcd)
            image_list.append(image)
            point2d_list.append(points2d)
            
        return image_list, point2d_list, \
               np.concatenate([xyzs, rgb, binary_masks, masks], axis=1),\
               intrinsics, self.extrinsic     # the intrinsics and extrinsics are kept the same object in different angles

def warpImage(image1, points1, points2, backgroud_color=(255, 255, 255)):
    '''
    Warp image1 to image2 using the paired points
    Args:
        image1:
        points1:
        points2:
        backgroud_color:

    Returns:
    '''
    image2 = np.ones_like(image1)
    image2 = image2 * backgroud_color

    H, W = image1.shape[0], image1.shape[1]

    pos1s = points1.astype(int)
    pos2s = points2.astype(int)

    pos1s[0, :] = np.minimum(np.maximum(pos1s[0, :], 0), W - 1)
    pos1s[1, :] = np.minimum(np.maximum(pos1s[1, :], 0), H - 1)
    pos2s[0, :] = np.minimum(np.maximum(pos2s[0, :], 0), W - 1)
    pos2s[1, :] = np.minimum(np.maximum(pos2s[1, :], 0), H - 1)

    image2[np.round(pos2s[1, :]).astype(int), np.round(pos2s[0, :]).astype(int)] = \
        image1[np.round(pos1s[1, :]).astype(int), np.round(pos1s[0, :]).astype(int)]
    return image2


def calculate_points2d(intrinsic, extrinsic, pcd):
    '''
    Project a point cloud into an image plane,
    Args:
        vis: o3d.visualization.Visualizer
        pcd: o3d.geometry.PointCloud

    Returns:

    '''
    intrinsics = intrinsic
    extrinsics = extrinsic
    # extrinsic rotation matrics(3*3)
    rvec = cv2.Rodrigues(extrinsics[:3, :3])[0]
    tvec = extrinsics[:3, 3]
    points2d, _ = cv2.projectPoints(pcd, rvec, tvec, intrinsics, None)
    return points2d[:, 0, :].T


def proj_2d_to_img(save_root, fn, normal_nlist, abnormal_nlist, sample_voxel=0.02):
    '''
    Investigate the correspondance between img and projected point cloud.
    '''
    H, W = 512, 512 
    if fn in normal_nlist:
        save_root = os.path.join(
            save_root, 'memory_bank', 
            fn, 'vs_{}_rs_{}'.format(sample_voxel, H)
            )
    elif fn in abnormal_nlist:
        save_root = os.path.join(
            save_root, fn, 
            'vs_{}_rs_{}'.format(sample_voxel, H)
            )
    else:
        raise Exception('Provided file name is not in the list!')
    
    pc_2d_n, img_2d_n = 'proj_2d.npy', 'img.png' 
    # load data
    cam_params = np.load(os.path.join(save_root, 'mult_view_img', 'cam_param.npz'))
    intrinsic = cam_params['intrinsics']
    extrinsic = cam_params['extrinsics']
    print(intrinsic)
    print('---------------')
    print(extrinsic)

    points = np.load(
        os.path.join(save_root, 'pointcloud', 'points.npz')
        )
    points = points['points']
    print(points.shape)
    print(intrinsic)
    # process the projected H and W
    pc_2d = np.load(
        os.path.join(save_root, 'mult_view_img', 'view_20', pc_2d_n)
        )

    h = pc_2d[1, :]
    w = pc_2d[0, :]
    # h = np.minimum(np.maximum(h, 0), h - 1)
    # w = np.minimum(np.maximum(w, 0), w - 1)
    h, w = np.round(h).astype(int), np.round(w).astype(int)
    img = np.ones([H, W, 3])
    for i in range(points.shape[0]):
        idx_rgb = points[i, 3:6]
        img[h[i], w[i], :] = idx_rgb
        
    cv2.imshow("proj_2d", img) # show numpy array
    
    cv2.waitKey(0)             # wait for ay key to exit window
    cv2.destroyAllWindows()    # close all windows
    return pc_2d


def normalize_colors(colors):
    return (colors - colors.min()) / (colors.max() - colors.min())
