import os

import numpy as np
import pandas as pd

import open3d as o3d

def pc_normalize(pc_unnorm, norm_method):
    """
        This function will return the normalized point cloud with respect to the mean of the points(-1 to 1)
        Args:
            pc: point cloud

        Returns:
            pc: normalized point cloud
    """
    pc = pc_unnorm.copy()
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    if norm_method == 'm3md':
        return pc
    else:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

class ArchSyn():
    """
        This class will read the synthetic data of arch and pre-exist 
        patchlib which saves the memory bank and pre-calculated 
        FPFH features.
    """
    def __init__(self, root_p:str, abnormal_dir_n:str='disp_x',  
                 normal_dir_n='min_cw0',
                 color_render:str='intensity',       # 'rgb' or 'intensity'
                 visuable:bool=False,                # visualize the point cloud with labels
                 ):
        super().__init__()
        # self.f_extractor_n = f_extractor_n
        # self.radius_norm, self.radius_feature = radius_norm, radius_feature
        # self.sample_voxel, self.mesh_size = sample_voxel, mesh_size
        # self.recompute_fpfh = recompute_fpfh       # recompute fpfh even precomputed fpfh exist
        self.color_render = color_render
        self.visuable = visuable
        # print(abnormal_dir_n)
        if abnormal_dir_n.split('_')[0]=='disp' and\
            abnormal_dir_n.split('_')[1] == 'x':                 # crack width has only meaning in x direction

            abnormal_n = abnormal_dir_n.split('_')[-1]  
            self.normal_fdir = os.path.join(root_p, 'memory_bank', normal_dir_n)
            if abnormal_n[0:2] == 'cw':
                self.abnormal_fdir = os.path.join(
                    root_p, 'disp_x', 
                    'diff_cw', abnormal_n, 'pointcloud',
                    ) 
            else:
                abnormal_n = abnormal_dir_n[7:]
                self.abnormal_fdir = os.path.join(
                    root_p, 'disp_x',
                    'diff_disps', abnormal_n, 'pointcloud',
                    )
            # print('Showing the normal and abnormal data directories:')
            # print(self.abnormal_fdir)
            # print(self.normal_fdir)

        else: 
            # print('Sir, this way')  
            self.normal_fdir = os.path.join(root_p, 'memory_bank', normal_dir_n)
            self.abnormal_fdir = os.path.join(
                root_p, abnormal_dir_n, 
                'pointcloud',
                ) 
        self.color_dict = {    # r, g, b
            'non_crack': np.array([65, 105, 225])/255, 
            'intrados_crack': np.array([1, 0, 0]), 
            'extrados_crack': np.array([0, 1, 0]),
            'inner_crack': np.array([160, 32, 240])/255,
            }
        print('Name of normal dir: {}:'.format(self.normal_fdir))
        assert os.path.exists(self.normal_fdir)
        print(self.abnormal_fdir)
        assert os.path.exists(self.abnormal_fdir)
        
    def readXYZfile(self, f_dir):
        '''
            This function is used to extract data from the 
        '''
        scan_fns = os.listdir(f_dir)
        if len(scan_fns) == 0:
            raise Exception('No file is included into the given dir!')
        scan_f_dirs = [os.path.join(f_dir, scan_fn) for scan_fn in scan_fns if 
                    os.path.splitext(scan_fn)[-1] == '.csv']
        xyzs_list, noise_xyzs_list, rgbs_list, labels_list = [], [], [], []
        intensity_list = []   # to substitute RGB color
        for _, scan_f_dir in enumerate(scan_f_dirs):
            frame = pd.read_csv(scan_f_dir, sep=';')
            xyzs = np.array(frame.loc[:, ['X', 'Y', 'Z']].values)
            rgbs = np.array(frame.loc[:, ['red', 'green', 'blue']].values)
            labels = np.array(frame.loc[:, ['categoryID']].astype('int32').values)
            intensity = np.array(frame.loc[:, ['intensity']].values)
            
            xyzs_list.append(xyzs)
            rgbs_list.append(rgbs)
            labels_list.append(labels)
            intensity_list.append(intensity)
            try:
                noise_xyzs = np.array(frame.loc[:, ['X_noise', 'Y_noise', 'Z_noise']].values)
                noise_xyzs_list.append(noise_xyzs)
                noise_xyzs = np.vstack(noise_xyzs_list)
            except:
                noise_xyzs = 0
                pass
            
        xyzs = np.vstack(xyzs_list)
        if self.color_render == 'rgb':
            rgbs = np.vstack(rgbs_list)
        else:
            rgbs = np.vstack(intensity_list).repeat(repeats=3, axis=1) # intensity-->RGB
        labels = np.vstack(labels_list)
        if not self.visuable:         # if not visualize, return binary labels
            binary_labels = np.zeros(labels.shape)
            binary_labels[labels.squeeze(1)!=0] = 1     # non-crack: 0, crack: 1, not differetiate the type of crack

        assert xyzs.shape[0] == rgbs.shape[0] == labels.shape[0]
        return noise_xyzs, noise_xyzs, rgbs, labels     # use noise data only

    def visualize_pc(self, xyzs, labels, rgb=None):
        synarch_vis_params = {
            "field_of_view":60.0,
            "front":[-0.495, -0.738, 0.459],
            "lookat":[-0.267, -0.341, -1.140],
            "up":[ 0.230, 0.398, 0.888],
            "zoom":0.480
        }     # camera angles
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if isinstance(rgb, np.ndarray):   # visualize with RGB color
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            labels = np.squeeze(labels)
            colors = np.zeros((labels.shape[0], 3))
            colors[labels==0, :] = self.color_dict['non_crack']
            colors[labels==1, :] = self.color_dict['intrados_crack']
            colors[labels==2, :] = self.color_dict['extrados_crack']
            colors[labels==3, :] = self.color_dict['inner_crack']
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries(
            [pcd], 
            front=synarch_vis_params['front'],      
            lookat=synarch_vis_params['lookat'],
            up=synarch_vis_params['up'],
            zoom=synarch_vis_params['zoom'],
        )
        return True

    def extract_data(self, ):
        abnormal_data = self.readXYZfile(self.abnormal_fdir)
        try:
            normal_data = self.readXYZfile(self.normal_fdir)
        except:
            print('No normal data is found! The path is: ', self.normal_fdir)
            normal_data = None
        try:
            abnormal_data = self.readXYZfile(self.abnormal_fdir)
        except:
            print('No abnormal data is found! The path is: ', self.abnormal_fdir)
            abnormal_data = None

        if self.visuable:
            xyzs, _, _, labels = abnormal_data
            self.visualize_pc(xyzs, labels)     
        return normal_data, abnormal_data

class ArchReal():
    """
        This class will read the real data of arch (London Bridge) 
        and pre-exist patchlib which saves the memory bank 
        and pre-calculated FPFH features.
    """            
    def __init__(self, root_p:str, n_arch, visuable:bool=False):
        super().__init__()
        n_normal, n_abnormal = n_arch
        self.normal_fdir = os.path.join(root_p, 'memory_bank', n_normal,
                                        'pointcloud', '3ddisp')
        self.abnormal_fdir = os.path.join(root_p, n_abnormal, 'pointcloud', 
                                          '3ddisp')
        self.color_dict = {    # r, g, b
            'non_crack': np.array([65, 105, 225])/255, 
            'crack': np.array([1, 0, 0]), 
            'water_patch': np.array([0, 1, 0]),
            }
        self.visuable = visuable
        print(self.normal_fdir)
        assert os.path.exists(self.normal_fdir)
        assert os.path.exists(self.abnormal_fdir)

    def readXYZfile(self, normal, f_dir, sep=' '):
        '''
        This is for extracting data from real point cloud
        '''
        arch_n = os.listdir(f_dir)
        arch_n = [i for i in arch_n if i.split('_')[-1]=='arch.asc']
        f_path = os.path.join(f_dir, arch_n[0])
        points = pd.read_csv(f_path, sep=sep)
        xyzs = points.iloc[:, 0:3].astype('float64').values
        rgb = points.iloc[:, 3:6].astype('int32').values      # rgb colors
        rgb = rgb/256          # normalized to 0-1
        if normal:
            masks = np.zeros(points.shape[0])
        else:
            masks = points.iloc[:, 6].astype('int32').values
        xyzs = pc_normalize(xyzs, norm_method='m3md')
        
        return xyzs, xyzs, rgb, masks
    
    def visualize_pc(self, xyzs, labels, rgb=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if isinstance(rgb, np.ndarray):   # visualize with RGB color
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            labels = np.squeeze(labels)
            colors = np.zeros((labels.shape[0], 3))
            colors[labels==0, :] = self.color_dict['non_crack']
            colors[labels==1, :] = self.color_dict['crack']
            colors[labels==2, :] = self.color_dict['water_patch']
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        o3d.visualization.draw_geometries([pcd])
        return True

    def extract_data(self, ):
        normal_data = self.readXYZfile(True, self.normal_fdir)
        abnormal_data = self.readXYZfile(False, self.abnormal_fdir)
        if self.visuable:
            xyzs, _, _, masks = abnormal_data
            self.visualize_pc(xyzs, masks) 
        return normal_data, abnormal_data

