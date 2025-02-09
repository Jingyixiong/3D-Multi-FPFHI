import numpy as np

from patchcore.patchcore import PatchCore

from feature_extractor.fpfh import FPFH
from feature_extractor.fpfh_scrach import FPFH_S

from dataloader.arch_dataloader import ArchALL
from dataloader.tunnel_dataloader import TunnelReal

class ALRunner():
    '''
    Anomaly detection runner for all the datasets
    '''
    def __init__(self, root_p:str, realarch_dn, synarch_dn, tunnel_dn,
                 sample_voxel:float, mesh_size:float,
                 f_extractor_n:str, feature_types,
                 radius_fs_ratios:list=[50], 
                 anomaly_vis=False, display='heatmap'):
        self.root_p = root_p
        self.realarch_dn, self.synarch_dn, self.tunnel_dn =\
              realarch_dn, synarch_dn, tunnel_dn
        self.sample_voxel = sample_voxel
        self.mesh_size = mesh_size
        self.f_extractor_n = f_extractor_n
        self.radius_norm = mesh_size*6
        self.radius_fs_ratios = radius_fs_ratios
        self.paras_return()
        self.fpfh_params =  {                        # only used with self-defined FPFH
                'n_bins':30,                         # bins for intensity
                'neigh_mode':'ball_query', 
                'nsample':None,                      # neighbors
                'intensity':True,  
                'disable_progress_bars':False
                }
        self.geo_extractors:dict = {
        'FPFH_Self': FPFH_S, 
        'FPFH': FPFH,
                }
        self.feature_types = feature_types
        self.radius_fs = [self.mesh_size*self.radius_fs_ratio
                for self.radius_fs_ratio in self.radius_fs_ratios]
        self.anomalyDetection = PatchCore(
            root_p=self.root_p,
            fpfh_params=self.fpfh_params, 
            geo_extractors=self.geo_extractors, 
            sample_voxel=self.sample_voxel, mesh_size=self.mesh_size,
            radius_norm=self.radius_norm,
            radius_fs_ratios=self.radius_fs_ratios,
            mem_n=4000, 
            f_extractor_n=self.f_extractor_n,
            backbone_name='resnet50d', 
            out_indices=(1, 2, 3), image_size=512,
            vis=anomaly_vis, display=display,  normal_visual=False,   # visualize results
            )  
    
    def load_realarch(self, arch_n:str):
        '''
        Load the synthetic or real arch dataset
        '''
        realarch_normal_nlist = [
                'Sarch_130305', 
                'Narch_130305'
            ]    # the arch which is used for reference
        realarch_abnormal_nlist = [       # S: sourtern arch, _130305: timestep
                'Sarch_131123',   
                'Narch_131123'
            ]
        if arch_n == 'North':      # return real north arch
            names = [realarch_normal_nlist[1], 
                     realarch_abnormal_nlist[1]]
        elif arch_n == 'South':
            names = [realarch_normal_nlist[0], 
                     realarch_abnormal_nlist[0]]
        else:
            raise ValueError('The arch name is not correct.')
        
        dataset = ArchALL(
            self.root_p, self.realarch_dn,
            names,
            self.radius_norm, self.radius_fs, self.sample_voxel, self.mesh_size,   # info for extracting corresponding memory bank
            img_res=512
            )
        return dataset
    
    def load_synarch(self, arch_n:str):
        '''
        Load the synthetic arch dataset
        '''
        #  min_cw0(default, can be tested with min_cw0.4), displacement
        names = ['min_cw0', arch_n]
        dataset = ArchALL(
            self.root_p, self.synarch_dn,
            names,
            self.radius_norm, self.radius_fs, self.sample_voxel, self.mesh_size,   # info for extracting corresponding memory bank
            img_res=512
            )
        return dataset
    
    def load_tunnel(self, tunnel_n:str):
        '''
        Load the tunnel dataset
        '''
        abnormal_n = '0-{}-2.txt'.format(tunnel_n)
        names = ['0-0-2.txt', abnormal_n]
        dataset = TunnelReal(
            self.root_p, self.tunnel_dn, names, voxel_size=self.sample_voxel
            )
        return dataset
    
    def anomaly_detect(self, inspect_tg, names):
        '''
        Anomaly detection on batch data of dataset.
        It is used for measuring metrics on different 
        datasets. 
        '''
        s_dict_list = []
        metrics_dict_list = []
        for name in names:
            if inspect_tg == 'real_arch':    
                vis_params = self.realarch_vis_params
                dataset = self.load_realarch(name)
            elif inspect_tg == 'syn_arch':
                vis_params = self.synarch_vis_params
                dataset = self.load_synarch(name)
            elif inspect_tg == 'tunnel':
                vis_params = self.realtunnel_vis_params
                if 'rgb_img' in self.feature_types or \
                    'rgb_img_FPFH' in self.feature_types:
                    raise ValueError('The tunnel dataset does not support RGB image.')
                dataset = self.load_tunnel(name)
            else:
                raise ValueError('The dataset name is not correct.')
            
            self.anomalyDetection.load_data(dataset)
            s_dict, metrics_dict = self.anomalyDetection.compute(
                        abscore_coefs=np.arange(0.01, 1.01, 0.01), 
                        inspect_ftypes=self.feature_types,
                        vis_params=vis_params
                        )
            s_dict_list.append(s_dict)
            metrics_dict_list.append(metrics_dict)

        return s_dict_list, metrics_dict_list
    
    def paras_return(self, ):
        self.realarch_vis_params = {
			"field_of_view" : 60.0,
			"front":[0, 0, 1],
			"lookat":[0.251, 0.03, -1.151],
			"up" : [0.484, 0.875, 0.003],
			"zoom" : 0.480
            }
        self.synarch_vis_params = {
            "field_of_view":60.0,
            "front":[-0.495, -0.738, 0.459],
            "lookat":[-0.267, -0.341, -1.140],
            "up":[ 0.230, 0.398, 0.888],
            "zoom":0.480
            }
        self.realtunnel_vis_params = {
			"field_of_view":60.0,
			"front":[-0.641, -0.095, -0.762],
			"lookat":[-0.941, -0.044, -0.257],
			"up":[-0.452, -0.755, 0.474],
			"zoom":0.5
            }
        
    