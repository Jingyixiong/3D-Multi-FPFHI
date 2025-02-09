import os

import numpy as np
import pandas as pd
import open3d as o3d

def voxelization(points: np.ndarray, voxel_size:float)-> np.ndarray:
    xyzs = points[:, 0:3]
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(xyzs)
    _, _, idx_list = pc_o3d.voxel_down_sample_and_trace(voxel_size=voxel_size,
                                            min_bound=pc_o3d.get_min_bound(),
                                            max_bound=pc_o3d.get_max_bound())   # downsampling points
    idx_list = [idx[0] for idx in idx_list]       # the point index which is closet to the voxel centre
    points = points[idx_list]
    return points, idx_list

class TunnelReal():
    """
        This class will read the synthetic data of arch and pre-exist 
        patchlib which saves the memory bank and pre-calculated 
        FPFH features.
    """
    def __init__(self, root_p:str, dataset_n, fn_tunnels,
                 voxel_size:float=None,  # voxel size for voxelization
                 recompute_fpfh:bool=False, vis:bool=False):
        super().__init__()
        self.root_p = os.path.join(root_p, dataset_n)
        self.recompute_fpfh = recompute_fpfh       # recompute fpfh even precomputed fpfh exist
        self.voxel_size = voxel_size

        self.fn_normal, self.fn_abnormal = fn_tunnels
        self.normal_fdir = os.path.join(self.root_p, 'memory_bank', self.fn_normal)
        self.abnormal_fdir = os.path.join(self.root_p, self.fn_abnormal)

        self.vis = vis
        self.color_dict = {    # r, g, b
            'non_crack': np.array([65, 105, 225])/255, 
            'inner_crack': np.array([1, 0, 0]), 
            'extra_crack': np.array([0, 1, 0]),
            }
        assert os.path.exists(self.normal_fdir)
        assert os.path.exists(self.abnormal_fdir)

    def visualize_pc(self, xyzs, labels, rgb=None):
        realtunnel_vis_params = {
			"field_of_view":60.0,
			"front":[-0.641, -0.095, -0.762],
			"lookat":[-0.941, -0.044, -0.257],
			"up":[-0.452, -0.755, 0.474],
			"zoom":0.5
            }
    
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyzs)
        if isinstance(rgb, np.ndarray):   # visualize with RGB color
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            labels = np.squeeze(labels)
            colors = np.zeros((labels.shape[0], 3))
            print(np.unique(labels))
            colors[labels==0, :] = self.color_dict['non_crack']
            colors[labels==1, :] = self.color_dict['inner_crack']
            colors[labels==2, :] = self.color_dict['extra_crack']
            pcd.colors = o3d.utility.Vector3dVector(colors)
        # save the point cloud and color 
        data = np.concatenate([xyzs, colors], axis=1)
        datakeys = ['x', 'y', 'z', 'r', 'g', 'b']
        df = pd.DataFrame(data, columns=datakeys)
        df.to_csv('103_colored.csv', index=False)
        
        o3d.visualization.draw_geometries(
            [pcd], 
            front=realtunnel_vis_params['front'],      
            lookat=realtunnel_vis_params['lookat'],
            up=realtunnel_vis_params['up'],
            zoom=realtunnel_vis_params['zoom'],
        )
        return True

    def intensity_to_rgb(self, intensity: np.ndarray):
        '''
            Normalize intensity within [0, 1]
        '''
        max_inte, min_inte = np.max(intensity), np.min(intensity)
        norm_intensity = (intensity-min_inte)/(max_inte-min_inte)
        colors = np.expand_dims(norm_intensity, 1).repeat(3, axis=1)
        return colors
    
    def readXYZfile(self, f_dir):
        '''
            This function is used to extract data from the 
        '''
        points = pd.read_csv(f_dir, sep=' ').values
        if self.voxel_size:
            points, _ = voxelization(points, self.voxel_size)
        registered_xyzs = points[:, 0:3]
        intensity = points[:, 3]
        label_inner = points[:, 5]
        label_outer = points[:, 6]
        # new label, 1 means inner crack, 2 means outer crack
        label = np.zeros_like(label_inner)
        label[label_inner==1] = 1
        label[label_outer==2] = 2

        colors = self.intensity_to_rgb(intensity)
        return registered_xyzs, registered_xyzs, colors, label

    def extract_data(self):
        normal_data = self.readXYZfile(self.normal_fdir)
        abnormal_data = self.readXYZfile(self.abnormal_fdir)
        # redefine dir to save the extracted features
        normal_fdir = os.path.join(self.root_p, 'memory_bank', 'load_step{}'.format(
            self.fn_normal.split('-')[1]
        ))
        abnormal_fdir = os.path.join(self.root_p, 'load_step{}'.format(
            self.fn_abnormal.split('-')[1]
        ))
        if self.vis:
            registered_xyzs, registered_xyzs, colors, label = abnormal_data
            self.visualize_pc(registered_xyzs, label)
        return [normal_data, [None, None]], [abnormal_data, [None, None]],\
               [normal_fdir, abnormal_fdir]
