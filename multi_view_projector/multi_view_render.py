import os

import numpy as np
import cv2
import open3d as o3d

from multi_view_projector.render_utils import MultiViewRender
from dataloader.render_dataloader import *
# from multi_view_projector.params import *

class RenderALL():   
    '''
    Render images based on given dataset.
    '''
    def __init__(self, root_p:str, realarch_src:str, realarch_tg,
                 synarch_src:str, synarch_tg:str, 
                 sample_voxel:float, color_render:str, ):
        self.sample_voxel = sample_voxel
        self.color_render = color_render
        assert color_render in ['rgb', 'intensity'], 'Color render should be either rgb or intensity.'

        # return camera parameters (rewrite this function if you apply to other datasets)
        self.cam_params_return()
        
        self.realarch_src_fp = os.path.join(root_p, realarch_src)
        self.realarch_tg_fp = os.path.join(root_p, realarch_tg)

        self.synarch_src_fp = os.path.join(root_p, synarch_src)
        self.synarch_tg_fp = os.path.join(root_p, synarch_tg)

        os.makedirs(self.realarch_tg_fp, exist_ok=True)
        os.makedirs(self.synarch_tg_fp, exist_ok=True)

    def realarch_dataset(self, names):
        '''
        Load real arch dataset.
        '''
        dataset = ArchReal(
            root_p=self.realarch_src_fp, 
            n_arch=names, 
            )
        return dataset

    def synarch_dataset(self, name):
        '''
        Load synthetic arch dataset based on given name.
        Input:
            names: [normal_n, abnormal_n]
            cw_ns: [normal_cw, abnormal_cw], only useful for disp_x
        '''
        dataset = ArchSyn(
            root_p=self.synarch_src_fp, 
            abnormal_dir_n=name,
            color_render=self.color_render,
        )
        return dataset

    def param_select(self, name=None, normal=False):
        '''
            Helps select the best camera positions.
        '''
        dataset = self.synarch_dataset(name)
        normal_data, abnormal_data = dataset.extract_data()
        data = normal_data if normal else abnormal_data
        xyzs, _, rgb, masks = data
        print(masks[0:10])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=512, height=512, visible=True)
        
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1
        ctr = vis.get_view_control()       # view controller
        init_param = ctr.convert_to_pinhole_camera_parameters()
        # intrinsics = init_param.intrinsic.intrinsic_matrix
        new_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=512, height=512, 
            fx=700, fy=700, 
            cx=512/2, cy=512/2
            )  # intrinsic parameter setting
        extrinsics = init_param.extrinsic
        print('Currrent extrinsic parameter:')
        print(extrinsics)
        print('Currrent intrinsic parameter:')
        print(init_param.intrinsic.intrinsic_matrix)
        print('-----------------------------------')

        new_extrinsics = np.array(extrinsics)
        new_extrinsics[0, 0] = -new_extrinsics[0, 0]     # reverse the y coords 
        new_extrinsics[2, 2] = -new_extrinsics[2, 2]     # reverse the z coords   
        new_extrinsics[0, -1] = 0.2
        print('extrinsics after being changed:')
        print(new_extrinsics)
        new_params = o3d.camera.PinholeCameraParameters()      
        new_params.intrinsic = new_intrinsics
        new_params.extrinsic = new_extrinsics
        # new_params.extrinsic = extrinsic_list[names[1]]
        ctr.convert_from_pinhole_camera_parameters(
            new_params, 
            allow_arbitrary=True
            )
        ctr = vis.get_view_control()       # view controller
        changed_param = ctr.convert_to_pinhole_camera_parameters()
        extrinsics = changed_param.extrinsic
        print('extrinsics after being changed:')
        print(extrinsics)
        print('intrinsic after being changed:')
        print(changed_param.intrinsic.intrinsic_matrix)
        vis.run()
        return 0

    def render_images(
            self, 
            save_root, data, focal, extrinsic_params,
            renderer, 
            ):
        '''
        Render images based on given dataset.
        Input:
            rednerer: class obj to render point cloud to images
            dataset: load the arch point cloud
            focal: focal length which control the size of object
            fn: the point cloud name for projection
        '''
        print('Current focal len {}.'.format(focal))
        renderer = MultiViewRender(
            voxel_size=self.sample_voxel, 
            focal=focal,
            extrinsic=extrinsic_params
            )
        H, W = renderer.H, renderer.W       # resolution of images
        xyzs, masks, rgb, binary_masks = data
        xyzs_ref = np.copy(xyzs)
        images, proj_2ds, \
            sub_points, intrinsics, extrinsics\
                = renderer.multiview_render(xyzs, masks, rgb, xyzs_ref, binary_masks)
        
        os.makedirs(save_root, exist_ok=True)
        # it saves the subsampled points, intrinsic parameters
        # and extrinsic parameters of cameras
        # (1) save point cloud
        pc_path = os.path.join(save_root, 'pointcloud')
        os.makedirs(pc_path, exist_ok=True)
        np.savez(
            os.path.join(pc_path, 'points'), 
            points=sub_points,)
        # (2) save camera params
        multi_view_path = os.path.join(save_root, 'mult_view_img')
        os.makedirs(multi_view_path, exist_ok=True)
        np.savez(
            os.path.join(multi_view_path, 'cam_param'), 
            intrinsics=intrinsics,
            extrinsics=extrinsics)
                
        # (3) save rendered images
        for idx, (image, proj_2d) in enumerate(zip(images, proj_2ds)):
            save_single_p = os.path.join(multi_view_path, 'view_{}'.format(idx))
            os.makedirs(save_single_p, exist_ok=True)
            # projected images
            cv2.imwrite(os.path.join(save_single_p, "img.png"), image)
            # the (h, w) idx of each projected points in point cloud 
            np.save(os.path.join(save_single_p, 'proj_2d.npy'), proj_2d)  
        return True
    
    def render_synarch(self, ):
        '''
        Render synthetic arch dataset.
        '''
        print('Render normal data.')
        for normal_n in self.synarch_normal_nlist:
            # camera parameters
            focal = self.synarch_focal_list[normal_n]
            extrinsic_params = self.synarch_extrinsic_list[normal_n]
            # load the dataset
            dataset = self.synarch_dataset('disp_z')   # only normal data is used, abnormal just for reference
            normal_data, _ = dataset.extract_data()
            renderer = MultiViewRender(
                            voxel_size=self.sample_voxel, 
                            focal=focal,
                            extrinsic=extrinsic_params
                            )
            H, W = renderer.H, renderer.W       # resolution of images
            save_root = os.path.join(
                        self.synarch_tg_fp, 'memory_bank', 
                        normal_n, 
                        'vs_{}_rs_{}'.format(self.sample_voxel, H)
                        )
            print('Start to render normal data {}.'.format(normal_n))
            self.render_images(
                save_root=save_root, 
                data=normal_data,
                focal=focal,
                extrinsic_params=extrinsic_params,
                renderer=renderer
                )
            print('Finish rendering, save to {}.'.format(save_root))

        print('Render abnormal data.')

        for abnormal_n in self.synarch_abnormal_nlist:
            focal = self.synarch_focal_list[abnormal_n]
            extrinsic_params = self.synarch_extrinsic_list[abnormal_n]
            renderer = MultiViewRender(
                            voxel_size=self.sample_voxel, 
                            focal=focal,
                            extrinsic=extrinsic_params
                            )
            H, _ = renderer.H, renderer.W        # resolution of images
            #           normal name,        abnormal name
            save_root = os.path.join(
                    self.synarch_tg_fp, abnormal_n, 
                    'vs_{}_rs_{}'.format(self.sample_voxel, H)
                    )
            dataset = self.synarch_dataset(abnormal_n)    # load dataset
            _, abnormal_data = dataset.extract_data()
            print('Start to render abnormal data {}.'.format(abnormal_n))  
            self.render_images(
                save_root=save_root, 
                data=abnormal_data,
                focal=focal,
                extrinsic_params=extrinsic_params,
                renderer=renderer
                )
            print('Finish rendering, save to {}.'.format(save_root))
        return True
    
    def render_realarch(self, ):
        '''
        Render real arch dataset.
        '''
        for normal_n, abnormal_n in zip(
            self.realarch_normal_nlist, self.realarch_abnormal_nlist
            ):
            # camera parameters
            normal_focal = self.realarch_focal_list[normal_n]
            normal_extrinsic_params = self.realarch_extrinsic_list[normal_n]
            abnormal_focal = self.realarch_focal_list[abnormal_n]
            abnormal_extrinsic_params = self.realarch_extrinsic_list[abnormal_n]
            # renderers
            normal_renderer = MultiViewRender(
                            voxel_size=self.sample_voxel, 
                            focal=normal_focal,
                            extrinsic=normal_extrinsic_params
                            )
            abnormal_renderer = MultiViewRender(
                            voxel_size=self.sample_voxel, 
                            focal=abnormal_focal,
                            extrinsic=abnormal_extrinsic_params
                            )
            
            H, _ = normal_renderer.H, normal_renderer.W       # resolution of images
            # save paths
            noraml_save_root = os.path.join(
                            self.realarch_tg_fp, 'memory_bank', 
                            normal_n, 
                            'vs_{}_rs_{}'.format(self.sample_voxel, H)
                            )
            abnormal_save_root = os.path.join(
                            self.realarch_tg_fp,
                            abnormal_n, 
                            'vs_{}_rs_{}'.format(self.sample_voxel, H)
                            )
            # load dataset
            names = [normal_n, abnormal_n]
            dataset = self.realarch_dataset(names)
            normal_data, abnormal_data = dataset.extract_data()

            # render normal data
            print('Start to render normal data {}.'.format(normal_n))
            self.render_images(
                save_root=noraml_save_root, 
                data=normal_data,
                focal=normal_focal,
                extrinsic_params=normal_extrinsic_params,
                renderer=normal_renderer
                )
            print('Finish rendering, save to {}.'.format(noraml_save_root))

            # render abnormal data
            print('Start to render abnormal data {}.'.format(abnormal_n))
            self.render_images(
                save_root=abnormal_save_root, 
                data=abnormal_data,
                focal=abnormal_focal,
                extrinsic_params=abnormal_extrinsic_params,
                renderer=abnormal_renderer
                )
            print('Finish rendering, save to {}.'.format(abnormal_save_root))

        return True
    
    def cam_params_return(self, ):
        '''
        Return the camera parameters.
        '''
        # (1) real arch
        self.realarch_normal_nlist = [
                    'Sarch_130305', 
                    'Narch_130305'
                ]    # the arch which is used for reference
        self.realarch_abnormal_nlist = [                      # S: sourtern arch, _130305: timestep
                    'Sarch_131123',   
                    'Narch_131123'
                    ] 
        self.realarch_focal_list = {
                    'Narch_131123': 450,
                    'Narch_130305': 450,
                    'Sarch_130305': 450,
                    'Sarch_131123': 450
                    }  
                 
        # if mismatching is found in render images(open3d)
        # and projected points(cv2.projectPoints), please
        # check the extrinsic matrix to make sure Rot is correct 
        self.realarch_extrinsic_list = {
            'Narch_131123': np.array([
                [ -1.,  0.,  0.,   0.74],
                [-0.,  -1., -0.,   1.39],
                [-0.,  -0.,   1.,   13.86],
                [ 0.,   0.,  0.,    1.]
                ]),
            'Narch_130305': np.array([
                [ -1.,  0.,  0.,   0.17],
                [-0.,  -1., -0.,   2.35],
                [-0.,  -0.,   1., 13.75],
                [ 0.,   0.,  0.,     1.]
                ]),
            'Sarch_130305': np.array([
                [ -1.,  0.,  0.,  0.43],
                [-0.,  -1., -0.,  0.67],
                [-0.,  -0.,   1., 13.26],
                [ 0.,   0.,  0.,     1.]
                ]), 
            'Sarch_131123':  np.array([
                [ -1.,  0.,  0.,  -0.1],
                [-0.,  -1., -0., -0.74],
                [-0.,  -0.,   1., 12.9],
                [ 0.,   0.,  0.,     1.]
                ]), 
            }
        self.realarch_vis_params = {
                    "field_of_view" : 60.0,
                    "front":[0, 0, 1],
                    "lookat":[0.251, 0.03, -1.151],
                    "up" : [0.484, 0.875, 0.003],
                    "zoom" : 0.480
                    }   

        # (2) Synthetic arch
        self.synarch_normal_nlist = ['min_cw0',]  #  'min_cw0.4'      
        self.synarch_abnormal_nlist = [ 
            # different maximum crack width                  
            'disp_x_cw0.4',
            'disp_x_cw0.8',
            'disp_x_40cm',
            # different displacement
            'disp_x_8cm',
            'disp_x_12cm',
            'disp_x_16cm',
            # different displacement without inner crack
            'disp_x_8cm_noinnerc',
            'disp_x_12cm_noinnerc',
            'disp_x_16cm_noinnerc',

            'disp_z',
            'disp_xz',
            'rot_x'
            ]
        self.synarch_focal_list = {
            'min_cw0': 700,
            'disp_x_cw0.8': 700,
            'disp_x_cw0.4': 700, 
            'disp_x_40cm': 700,
            # different displacement
            'disp_x_8cm': 700,
            'disp_x_12cm': 700,
            'disp_x_16cm': 700,
            # different displacement without inner crack
            'disp_x_8cm_noinnerc': 700,
            'disp_x_12cm_noinnerc': 700,
            'disp_x_16cm_noinnerc': 700,

            'disp_xz': 700,
            'disp_z': 700,
            'rot_x': 700,
            }
        
        disp_x_extrinsic = np.array([
            [ -1.,  0.,  0.,      0],
            [-0.,  -1., -0.,   1.80],
            [-0.,  -0.,   1.,  8.13],
            [ 0.,   0.,  0.,     1.]
            ])
        self.synarch_extrinsic_list = {
            'min_cw0': disp_x_extrinsic, 
            'disp_x_cw0.8': disp_x_extrinsic,
            'disp_x_cw0.4': disp_x_extrinsic,
            'disp_x_40cm': np.array([
                [ -1.,  0.,  0.,   0.20],
                [-0.,  -1., -0.,   1.80],
                [-0.,  -0.,   1.,  8.80],
                [ 0.,   0.,  0.,     1.]
                ]),
                
            'disp_x_8cm': disp_x_extrinsic,
            'disp_x_12cm': disp_x_extrinsic,
            'disp_x_16cm': disp_x_extrinsic,
            'disp_x_8cm_noinnerc': disp_x_extrinsic,
            'disp_x_12cm_noinnerc': disp_x_extrinsic,
            'disp_x_16cm_noinnerc': disp_x_extrinsic,

            'disp_xz': np.array([
                [ -1.,  0.,  0.,   0.1],
                [-0.,  -1., -0.,   1.80],
                [-0.,  -0.,   1.,  9.54],
                [ 0.,   0.,  0.,     1.]
                ]), 
            'disp_z': np.array([
                [ -1.,  0.,  0.,   0.1],
                [-0.,  -1., -0.,   1.80],
                [-0.,  -0.,   1.,  9.54],
                [ 0.,   0.,  0.,     1.]
                ]),
            'rot_x': np.array([
                [ -1.,  0.,  0.,   0.1],
                [-0.,  -1., -0.,   1.80],
                [-0.,  -0.,   1.,  9.54],
                [ 0.,   0.,  0.,     1.]
                ]),     
        }

        self.synarch_vis_params = {
                "field_of_view":60.0,
                "front":[-0.495, -0.738, 0.459],
                "lookat":[-0.267, -0.341, -1.140],
                "up":[ 0.230, 0.398, 0.888],
                "zoom":0.480
                }
