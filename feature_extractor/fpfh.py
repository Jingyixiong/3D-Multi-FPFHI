import numpy as np
import open3d as o3d

class FPFH:
    '''
    It calculates the FPFH features of points. 
    '''
    def __init__(self, fpfh_norm:bool=True):
        self.fpfh_norm = fpfh_norm

    def fpfh(self, pc_o3d: o3d.geometry.PointCloud, neigh_feature):
        fpfh_f = o3d.pipelines.registration.compute_fpfh_feature(pc_o3d,
                                                                 neigh_feature)
        fpfh_f = fpfh_f.data.T
        if self.fpfh_norm:
            fpfh_fnorm = np.zeros_like(fpfh_f)
            f1_range = np.sum(fpfh_f[:, :11], axis=-1, keepdims=True)
            f1_range_flag = (np.squeeze(f1_range)!=0)
            f2_range = np.sum(fpfh_f[:, 11:22], axis=-1, keepdims=True)
            f2_range_flag = (np.squeeze(f2_range)!=0)
            f3_range = np.sum(fpfh_f[:, 22:], axis=-1, keepdims=True)
            f3_range_flag = (np.squeeze(f3_range)!=0)

            fpfh_fnorm[f1_range_flag, :11] = fpfh_f[f1_range_flag, :11]/f1_range[f1_range_flag,:]
            fpfh_fnorm[f2_range_flag, 11:22] = fpfh_f[f2_range_flag, 11:22]/f2_range[f2_range_flag,:]
            fpfh_fnorm[f3_range_flag, 22:] = fpfh_f[f3_range_flag, 22:]/f3_range[f3_range_flag,:]
            return fpfh_fnorm
        else:
            return fpfh_f
        