import os 

import numpy as np
import open3d as o3d
import timm
import torch

from torch.nn import functional as F
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from tqdm import tqdm
from patchcore.help_func import colfunc_2

class PatchCore:
    def __init__(self, root_p:str,
                 fpfh_params:dict, geo_extractors:dict,
                 sample_voxel:float=0.02, mesh_size:float=0.03, radius_fs_ratios=[50], radius_norm:float=0.12,
                 f_extractor_n:str='FPFH_Self',
                 n_views:int=27, backbone_name:str='resnet50d', 
                 out_indices=(1, 2, 3), image_size:int=512,
                 mem_n:int=4000, n_reweight:int=16,
                 vis:bool=False, display:str='heatmap', normal_visual:bool=False):
        # Data extract path
        self.root_p = root_p

        # Geometric feature extract parameters 
        self.f_extractor_n = f_extractor_n
        self.sample_voxel = sample_voxel             # sampling ratio, unit: m
        self.mesh_size = mesh_size                   # unit: m
        self.radius_norm = radius_norm
        self.radius_fs = [mesh_size*radius_fs_ratio 
                          for radius_fs_ratio in radius_fs_ratios] 
        self.vis: bool = vis                         # visualize results
        self.normal_visual = normal_visual           # whether to visualize the normal vector
        self.crack_c = np.array([1, 0, 0])
        self.crack_nc = np.array([93, 93, 93])/255
        
        self.fpfh_params = fpfh_params
        self.geo_extractors = geo_extractors

        # RGB feature extract parameters
        self.device:str = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_views = n_views
        self.backbone_name = backbone_name
        self.out_indices = out_indices
        self.image_size = image_size                 # defualt is 512, should be adjusted based on input img size
        self.model = timm.create_model(
                    model_name=self.backbone_name,
                    in_chans=3,       # rgb channel
                    out_indices=self.out_indices,
                    features_only=True,
                    pretrained=True, checkpoint_path='',
                    ).to(self.device)
        self.model.eval()

        # Parameters for the patchcore method
        self.mem_n = mem_n
        self.n_reweight = n_reweight    

        # Visualization 
        self.display = display                       # it could be 'binary' or 'heatmap'
        assert self.display in ['binary', 'heatmap']  

        self.pointsf_n = 'ms_{}_rn_{}_rf_{}'.format(
                    self.mesh_size, 
                    self.radius_norm, 
                    self.radius_fs,
                    )   
    
    def load_data(self, dataset):
        self.dataset = dataset
        # Extract data
        self.normal_data, self.abnormal_data, self.mem_dirs = dataset.extract_data()
        normal_pc_data, normal_img_data = self.normal_data
        abnormal_pc_data, abnormal_img_data = self.abnormal_data
        # extract point cloud data
        self.pc_nodisp, self.pc_disp = normal_pc_data[0], abnormal_pc_data[0]
        self.mask_nodisp, self.mask_disp = normal_pc_data[-1], abnormal_pc_data[-1]
        self.color_nodisp, self.color_disp = normal_pc_data[2], abnormal_pc_data[2]
        # extract image data
        self.imgs_nodisp, self.pos_nodisp = normal_img_data
        self.imgs_disp, self.pos_disp = abnormal_img_data
        print('Finish loading data!')
          
    def camera_loc(self, pc):
        '''
            Location of the camera for unifying the orientation of normal vector.
        '''
        max_pc = np.max(pc, axis=0)
        min_pc = np.min(pc, axis=0)
        # max_pcy = np.max(pc, axis=1)
        # min_pcy = np.min(pc, axis=1)
        cam_loc = (max_pc-min_pc)/2 + min_pc
        # x + 100, for different components, different strategy needs to be used for adjusting the normal vector dir
        cam_loc[-1] +=0.2    
        return cam_loc
    
    def return_membank_ns(self, feature_types):
        '''
        Return the names of all memory banks.
        '''
        membank_ns_dict = {}
        for feature_type in feature_types:
            membank_n = feature_type+'_membank'
            print('Memory bank name: {}'.format(membank_n))
            if membank_n == 'FPFH_membank' or membank_n == 'relaRGB_membank'\
                or membank_n == 'FPFH_relaRGB_membank':   # based purely on FPFH algorithm
                name = 'ms_{}_rn_{}_rf_{}_bins_{}_mem_{}'.format(
                    self.mesh_size, 
                    self.radius_norm, 
                    self.radius_fs,
                    self.fpfh_params['n_bins'],
                    self.mem_n
                    )           # FPFH feature only controlled by those parameters
            elif membank_n == 'rgb_img_membank':
                name = 'bn_{}_rs_{}_outidx_{}_mem_n_{}'.format(
                        self.backbone_name, 
                        self.image_size,
                        self.out_indices,
                        self.mem_n
                        )        # name of potentially existing memory bank
            elif membank_n == 'rgb_img_FPFH_membank':                # FPFH+img
                name = 'ms_{}_rn_{}_rf_{}_bins_{}_bn_{}_rs_{}_outidx_{}_mem_n_{}'.format(
                    self.mesh_size, 
                    self.radius_norm, 
                    self.radius_fs,
                    self.fpfh_params['n_bins'],
                    self.backbone_name, 
                    self.image_size,
                    self.out_indices,
                    self.mem_n
                    )        # name of potentially existing memory bank
            else:
                raise KeyError()
            membank_ns_dict[membank_n] = name
        return membank_ns_dict
    
    def rgb_feature(self, f_dir, image_list, point2d_list, recompute_rgb=False):
        '''
            Extract the corresponding rgb features from the 
            images and related to each point in point cloud.
            Input:
                image_list: list of images
                point2d_list: index on images for each point 
        '''
        
        n_views = self.n_views if self.n_views<len(image_list) \
                               else len(image_list)
        rgb_features_list = np.zeros([point2d_list[0].shape[1], 1792])   # 1792 is the feature dim from resnet50d
        with torch.no_grad():
            for image, point2d in zip(image_list[:n_views], point2d_list[:n_views]):
                image = image.unsqueeze(dim=0).to(self.device)
                # print('Image size: {}'.format(image.shape))
                feature_maps = self.model(image)            # extract feature maps from images
                rgb_features = []
                # transfer h into int type
                h = np.rint(point2d[1, :]).astype(int)     # corresponding h idx in image for each point in source pc
                w = np.rint(point2d[0, :]).astype(int)     # corresponding w idx in image for each point in source pc
                for feature_map in feature_maps:
                    f_resize = F.interpolate(
                        feature_map, self.image_size, mode='bilinear'
                        ) # (B, ch, H, W)
                    f_resize_np = f_resize.to('cpu').numpy()
                    del f_resize        # free memory
                    rgb_feature = f_resize_np[:, :, h, w]
                    rgb_features.append(rgb_feature)
                    # rgb_feature = f_resize[:, :, h, w]

                    # rgb_feature_np = rgb_feature.to('cpu').numpy()
                    # rgb_features.append(rgb_feature_np)
                    # del rgb_feature        # free memory

                rgb_features = np.concatenate(rgb_features, axis=1)
                rgb_features_list += rgb_features.squeeze(0).transpose(1, 0)
            # averange over all images
            # rgb_features = np.concatenate(rgb_features_list, 0)       
            rgb_features = rgb_features_list/n_views
        del image, feature_maps
        torch.cuda.empty_cache()
        # print('The data type of rgb_features is: {}'.format(rgb_features.dtype))
        return rgb_features.astype(np.float32)    # [N, f_dim]
          
    def points_3d_feature(self, xyzs, colors):
        '''
            Calculate the normal vector of given point cloud.
            This function assumes the color already exists in
            point cloud. Color can be intensity value, which commonly
            exists in point clouds.
        '''
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(xyzs)
        # (1) subsampled color
        if isinstance(colors, np.ndarray):  
            pc_o3d.colors = o3d.utility.Vector3dVector(colors)
        else:
            raise Exception('Lack of RGB information in point cloud!')
            
        # estimate normal
        neigh_norm = o3d.geometry.KDTreeSearchParamRadius(radius=self.radius_norm)
        cam_loc = self.camera_loc(xyzs)
        pc_o3d.estimate_normals(search_param=neigh_norm)
        pc_o3d.orient_normals_towards_camera_location(cam_loc)            # orient the normal to the same direction

        if self.normal_visual:
            o3d.visualization.draw_geometries([pc_o3d],
                                               zoom=15,
                                               front=[0.5439, -0.2333, -0.8060],
                                               lookat=[2.4615, 2.1331, 1.338],
                                               up=[-0.1781, -0.9708, 0.1608],
                                               point_show_normal=True)
        
        f_lists = []
        assert self.f_extractor_n in self.geo_extractors
        f_extractor = self.geo_extractors[self.f_extractor_n]

        if self.f_extractor_n == 'FPFH':                  # FPFH feature in multiple scales
            print('It returns only FPFH for point cloud!')
            fpfh_extractor = f_extractor()           # initialize feature calculator
            for i, radius_f in enumerate(self.radius_fs):
                print('Start to calculate neighbor index for FPFH with radius {}!'.format(radius_f))
                neigh_feature = o3d.geometry.KDTreeSearchParamRadius(radius=radius_f)
                points_feature = fpfh_extractor.fpfh(pc_o3d, neigh_feature)
                print('Finish calculate FPFH in layer {}, there are {} in total!'.format(i, len(self.radius_fs)))                                                                
                f_lists.append(points_feature)
        
        elif self.f_extractor_n == 'FPFH_Self':
            sub_xyzs = np.asarray(pc_o3d.points)
            sub_colors = np.asarray(pc_o3d.colors)
            sub_points = np.concatenate([sub_xyzs, sub_colors], axis=1)
            sub_normals = np.asarray(pc_o3d.normals)
            for i, radius_f in enumerate(self.radius_fs):
                print(
                    'Start to calculate neighbor index for FPFH(self defined) with radius {}!'.format(radius_f)
                    )
                fpfh_extractor = f_extractor(
                    radius=radius_f, 
                    n_bins=self.fpfh_params['n_bins'], 
                    intensity=self.fpfh_params['intensity'],     # default should be True
                    neigh_mode=self.fpfh_params['neigh_mode'], 
                    nsample=self.fpfh_params['nsample'],
                    disable_progress_bars=self.fpfh_params['disable_progress_bars']
                    )
                points_feature = fpfh_extractor.fpfh(
                    points=sub_points, normals=sub_normals)
                f_lists.append(points_feature)

        # process multiscale features
        assert len(f_lists) != 0
        points_feature = np.sum(f_lists, axis=0)/len(f_lists)
        return np.asarray(pc_o3d.points), points_feature

    def feature_factory(self, feature_type, points_feature, rgb_feature):
        '''
        Return features based on the give names. Further adjustments on 
        FPFH and RGB features can be done in this function.
        '''
        n_bins = self.fpfh_params['n_bins']           # number of bins
        n_geobins = int(3*n_bins)
        geo_feature = points_feature[:, :n_geobins]   # FPFH geometric feature
        naiveRGB_feature = points_feature[:, n_geobins:(n_geobins+n_bins)]
        relaRGB_feature = points_feature[:, (n_geobins+n_bins):(n_geobins+n_bins*2)]

        # normalize to unit vector
        geo_feature, naiveRGB_feature, relaRGB_feature = normalize(geo_feature),\
                    normalize(naiveRGB_feature),normalize(relaRGB_feature)  
        rgb_feature = normalize(rgb_feature) if type(rgb_feature)==np.ndarray else None
        if feature_type == 'FPFH':
            feature = normalize(geo_feature)
        # elif feature_type == 'FPFH_naiveRGB':
        #     feature = np.concatenate([geo_feature, naiveRGB_feature], axis=1)
        elif feature_type == 'relaRGB':
            feature = relaRGB_feature
        elif feature_type == 'FPFH_relaRGB':
            feature = np.concatenate([geo_feature, relaRGB_feature], axis=1)
        elif feature_type == 'rgb_img':
            feature = rgb_feature
        elif feature_type == 'rgb_img_FPFH':
            feature = np.concatenate([geo_feature, rgb_feature], axis=1)
        else:
            raise Exception("Given key '{}' does not includes in default feature type".format(feature_type))
        return feature
    
    def memory_bank(self, features):
        '''
            It condenses features extracted from the normal, which are undeformed point 
            cloud data.
            Input: 
                features: [N, f]
        '''
        N, _ = features.shape
        features = torch.tensor(features)
        select_idx = np.random.randint(N)                   # randomly initialize a centre
        last_item = features[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(features - last_item, dim=1, keepdims=True)

        for _ in tqdm(range(self.mem_n - 1)):
            distances = torch.linalg.norm(features - last_item, dim=1, keepdims=True)     # broadcasting step
            min_distances = torch.minimum(distances, min_distances)                    # iterative step
            select_idx = torch.argmax(min_distances)                                   # selection step
            # bookkeeping
            last_item = features[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        coreset_idx = torch.stack(coreset_idx)
        patch_lib = features[coreset_idx]                                              # condensated patch from the nodisp fpfh features
        return patch_lib

    def patchcore(self, geo_f, patch_lib):
        patch_lib = patch_lib if torch.is_tensor(patch_lib) else torch.tensor(patch_lib).detach()
        geo_f = torch.tensor(geo_f).detach()
        dist = torch.cdist(geo_f, patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)            # find the best match between the test patch and patch in memory bank
        s_idx = torch.argmax(min_val)                        # largest idx might indicate the abnormal
        s_star = torch.max(min_val)

        # reweighting
        m_test = geo_f[s_idx].unsqueeze(0)                                # anomalous patch
        m_star = patch_lib[min_idx[s_idx]].unsqueeze(0)                   # closest neighbour
        w_dist = torch.cdist(m_star, patch_lib)                           # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test - patch_lib[nn_idx[0, 1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(geo_f.shape[1]))    # scaling factor 
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        return min_val.numpy(), s.numpy()

    def generate_and_save_points_feature(self, 
            f_dir, pc, color
            ):
        '''
            This function generates and save the FPFH features.
            If FPFH features exists, this func will load it.
        '''
        save_p = os.path.join(f_dir, 'pointcloud', 'FPFH')
        os.makedirs(save_p, exist_ok=True)
        feature_p = os.path.join(save_p, self.pointsf_n+'.npz')
        if os.path.isfile(feature_p):
            print('Point feature exists for the point cloud in dir:')
            print(feature_p)
            points_feature = np.load(feature_p)['points_feature']
        else:
            print('No point feature exists for the point cloud in dir:')
            _, points_feature = self.points_3d_feature(
                xyzs=pc, colors=color
                    )
            print('Save point feature in:')
            save_p = os.path.join(f_dir, 'pointcloud', 'FPFH')
            print(save_p)
            os.makedirs(save_p, exist_ok=True)
            np.savez(os.path.join(save_p, self.pointsf_n+'.npz'), 
                     points_feature=points_feature)
        return points_feature
    
    def generate_and_save_membank(self, points_feature, rgb_feature):
        # check if all types of memery banks exist
        mem_banks = {}
        for membank_fn, name in self.membank_ns_dict.items():
            os.makedirs(os.path.join(self.mem_dirs[0], membank_fn), exist_ok=True)
            membank_p = os.path.join(self.mem_dirs[0], membank_fn, name+'.npz')

            if os.path.isfile(membank_p):
                print('{} exists!'.format(membank_fn))
                patch_lib = np.load(membank_p)['patch_lib']
            else:
                print('{} does not exist!'.format(membank_fn))
                feature = self.feature_factory(
                    membank_fn.strip('_membank'),
                    points_feature=points_feature, 
                    rgb_feature=rgb_feature
                    )   # including feature normalizing
                patch_lib = self.memory_bank(feature)
                print('Save {}!'.format(membank_fn))
                np.savez(membank_p, patch_lib=patch_lib)
            mem_banks[membank_fn] = patch_lib
        return mem_banks

    def eager_membank(self,):
        '''
            This function is used to eagerly load 
            the memory bank for all types of features.
        '''
        membank_ns_dict = self.return_membank_ns(['FPFH', 'relaRGB', 'FPFH_relaRGB', 'rgb_img', 'rgb_img_FPFH'])
        print(membank_ns_dict)
        mem_banks = {}
        for membank_fn, name in membank_ns_dict.items():
            membank_p = os.path.join(self.mem_dirs[0], membank_fn, name+'.npz')
            if os.path.isfile(membank_p):
                print('{} exists!'.format(membank_fn))
                patch_lib = np.load(membank_p)['patch_lib']
            else:
                print('{} does not exist!'.format(membank_fn))
                patch_lib = None
            mem_banks[membank_fn] = patch_lib
        return mem_banks
    
    def eager_feature(self, normal_flag, inspect_ftype):
        '''
            This function is used to eagerly load 
            the feature for statistical analysis.
        '''
        # load point feature
        f_dir = self.mem_dirs[0] if normal_flag else self.mem_dirs[1]
        save_p = os.path.join(f_dir, 'pointcloud', 'FPFH')
        feature_p = os.path.join(save_p, self.pointsf_n+'.npz')
        if os.path.isfile(feature_p):
            print('Point feature exists for the point cloud in dir:')
            print(feature_p)
            points_feature = np.load(feature_p)['points_feature']
        else:
            raise Exception('No point feature exists for the point cloud in dir: \n{}'.format(feature_p))
        # load rgb feature
        if inspect_ftype == 'rgb_img' or inspect_ftype == 'rgb_img_FPFH':
            if normal_flag:
                rgb_feature = self.rgb_feature(self.imgs_nodisp, self.pos_nodisp)
            else:
                rgb_feature = self.rgb_feature(self.imgs_disp, self.pos_disp)
        else:
            rgb_feature = None
        # return the feature based on the given type
        feature = self.feature_factory(
            feature_type=inspect_ftype, 
            points_feature=points_feature,
            rgb_feature=rgb_feature
            )
        # label
        label = self.mask_nodisp if normal_flag else self.mask_disp
        return feature, label

    def compute(self, abscore_coefs, inspect_ftypes, vis_params):
        '''
        Input:
            dataset: provided dataset
            absocre_coef: abduct coefficient 
            inspect_ftypes: feature types to be inspected
        '''
        # 1. Extract memory bank, point feature and image features
        # (1) normal samples
        img_flag = any(['img' in inspect_ftype for inspect_ftype in inspect_ftypes])
        if img_flag:                        # 2D images exist for the inspection
            if type(self.imgs_nodisp) is list:
                normal_rgbf = self.rgb_feature(
                    f_dir=self.mem_dirs[0], 
                    image_list=self.imgs_nodisp, 
                    point2d_list=self.pos_nodisp
                    )                       # rgb feature not saved
                                            # normalize into a unit vector
            else:
                normal_rgbf = None
                print('Images not exist!')
        else:
            normal_rgbf = None
        print('Finish extracting RGB features!')  

        normal_pointsf = self.generate_and_save_points_feature(
            f_dir=self.mem_dirs[0],         # dir saves memory bank
            pc=self.pc_nodisp, 
            color=self.color_nodisp
            )                               # FPFH feature saved
        
        self.membank_ns_dict = self.return_membank_ns(inspect_ftypes)
        memory_banks = self.generate_and_save_membank(
            points_feature=normal_pointsf, 
            rgb_feature=normal_rgbf,
        )               # memory bank for all five types of features                   

        # (2) abnormal samples 
        if img_flag:
            if type(self.imgs_disp) is list:
                abnormal_rgbf = self.rgb_feature(
                    f_dir=self.mem_dirs[1], 
                    image_list=self.imgs_disp, 
                    point2d_list=self.pos_disp
                    )   # rgb feature not saved
                print('abnormal_rgbf: {}'.format(abnormal_rgbf.shape))
            else:
                abnormal_rgbf = None
                print('Images not exist!')
        else:
            abnormal_rgbf = None
        print('Finish extracting RGB features!')
        abnormal_pointsf = self.generate_and_save_points_feature(
            f_dir=self.mem_dirs[1],         # dir saves memory bank
            pc=self.pc_disp, 
            color=self.color_disp, 
            )
    
        # 2. Compute anomaly score
        s_dict = {}                             # store the anomaly score
        binary_masks = np.zeros(self.mask_disp.shape)
        # non-crack: 0, crack+water_patch(real arch): 1, not differetiate the type of crack
        binary_masks[self.mask_disp!=0] = 1     
        recalls_dict, precisions_dict, f1s_dict = {}, {}, {}  # save metrics in different scales
        for inspect_ftype in inspect_ftypes:
            membank_n = inspect_ftype+'_membank'
            patch_lib = memory_banks[membank_n]
            
            abnormal_f = self.feature_factory(
                feature_type=inspect_ftype,
                points_feature=abnormal_pointsf,
                rgb_feature=abnormal_rgbf
                )
            print('Start to compute anomaly score for {}!'.format(inspect_ftype))
            min_v, s = self.patchcore(abnormal_f, patch_lib)
            if s == 0:
                raise Exception('Features are not identifiable by the algorithm, try different parameters!')
            norm_diff = self.vis_results(min_v, s, self.pc_disp, vis_params)
            s_dict[inspect_ftype] = norm_diff
            recalls, precisions, f1s = self.metrics(
                binary_masks, 
                min_v, s,
                abscore_coefs)
            recalls_dict[inspect_ftype] = recalls
            precisions_dict[inspect_ftype] = precisions
            f1s_dict[inspect_ftype] = f1s
            
        return s_dict, [recalls_dict, precisions_dict, f1s_dict]
      
    def metrics(self, tgt, min_v, s, abscore_coefs):
        '''
            Return the wanted metrics from 
        '''
        min_v = min_v.numpy() if torch.is_tensor(min_v) else min_v
        predicts = []
        recalls, precisions, f1s = [], [], []
        for abscore_coef in abscore_coefs:
            predicts.append(min_v > s*abscore_coef)
            predict = (min_v > s*abscore_coef)
            TP = np.sum((tgt==1)&(predict==1))+1e-9    # avoid 0
            FN = np.sum((tgt==0)&(predict==1))
            FP = np.sum((tgt==1)&(predict==0))
            recall, precision = TP/(TP+FN), TP/(TP+FP)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append((2*precision*recall)/(precision + recall))   # element-wise computation

        return recalls, precisions, f1s

    def vis_results(self, min_v, s, pc_disp, vis_params):
        predict_cmap = np.zeros([pc_disp.shape[0], 3])
        if self.display == 'binary':
            predict = min_v > s*0.2
            predict_cmap[predict, :] = self.crack_c
            predict_cmap[~predict, :] = self.crack_nc
        else:
            s_diff = min_v+(s-np.min(min_v))           
            predict_cmap, norm_diff = colfunc_2(s_diff)
        ######## color map based on scaled difference between perfect and non-perfect arch ########
        if self.vis:
            vis_pc = o3d.geometry.PointCloud()
            vis_pc.points = o3d.utility.Vector3dVector(pc_disp)
            vis_pc.colors = o3d.utility.Vector3dVector(predict_cmap)
            o3d.visualization.draw_geometries(
                [vis_pc], 
                front=vis_params['front'],
                lookat=vis_params['lookat'],
                up=vis_params['up'],
                zoom=vis_params['zoom'],
                )   
        return norm_diff