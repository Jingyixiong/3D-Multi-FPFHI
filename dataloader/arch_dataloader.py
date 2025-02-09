import os
import glob

import numpy as np

from PIL import Image
from torchvision import transforms
# from help_func import pc_normalize

# mean and std of imagenet dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

class ArchALL():
    """
        This class will read the real data (Multi view rendered images and point cloud)
        of arch (London Bridge&Marsh Line) 
        and pre-exist patchlib which saves the memory bank 
        and pre-calculated FPFH features.
    """            
    def __init__(self, root_p:str, dataset_n:str,
                 n_arch,                  # n_arch:['Sarch_121214', 'Sarch_130305',] 
                 radius_norm:float, radius_feature:float,
                 sample_voxel:float, mesh_size:float, 
                 img_res:int,             # image resolution (H, W)
                 pc_extractor_n:str='FPFH',
                 recompute_fpfh:bool=False,
                 ):
        super().__init__()
        self.root_p = root_p
        self.dataset_n = dataset_n
        self.pc_extractor_n = pc_extractor_n
        self.radius_norm, self.radius_feature = radius_norm, radius_feature
        self.sample_voxel, self.mesh_size = sample_voxel, mesh_size
        self.img_res = img_res
        self.recompute_fpfh = recompute_fpfh  # recompute fpfh even precomputed fpfh exist
        self.n_arch = n_arch

        # return the base root of normal and abnormal objs
        self.extract_path()
        # print('Direction of normal dir: {}:'.format(self.normal_fdir))
        assert os.path.exists(self.normal_fdir)
        assert os.path.exists(self.abnormal_fdir)

        # process images
        H, W = self.img_res, self.img_res         # the size of images
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def extract_path(self,):
        '''
            Given different dataset, it will return the path of normal 
            and abnormal data file path.
        '''
        fn = 'vs_{}_rs_{}'.format(self.sample_voxel, self.img_res)  
        n_normal, n_abnormal = self.n_arch
        self.normal_fdir = os.path.join(
                    self.root_p, 
                    self.dataset_n,
                    'memory_bank', n_normal, fn
                    )
        self.abnormal_fdir = os.path.join(
                    self.root_p, 
                    self.dataset_n, n_abnormal, fn 
                    )

    def read_pc(self, f_dir):
        '''
        This is for extracting data from real point cloud.
        '''
        f_path = os.path.join(f_dir, 'pointcloud', 'points.npz')
        points = np.load(f_path)['points']
        xyzs, rgb, binary_masks, masks = points[:, :3], \
                                         points[:, 3:6],\
                                         points[:, 6], \
                                         points[:, 7:]
        xyzs = pc_normalize(xyzs, norm_method='m3md')
        
        return xyzs, masks, rgb, binary_masks

    def read_img(self, f_dir):
        f_path = os.path.join(f_dir, 'mult_view_img')
        view_image_paths = glob.glob(f_path + "/*/img.png")
        view_position_paths = glob.glob(f_path + '/*/proj_2d.npy')
        view_image_paths.sort()
        view_position_paths.sort()

        view_images = [self.rgb_transform(Image.open(image_path).convert('RGB')) 
                       for image_path in view_image_paths]
        view_positions = [np.load(position_path) for position_path in view_position_paths]
        return view_images, view_positions

    def extract_data(self):
        # (1) extract point cloud data
        normal_pc_data = self.read_pc(self.normal_fdir)
        abnormal_pc_data = self.read_pc(self.abnormal_fdir)

        # (2) extract image data
        normal_img_data = self.read_img(self.normal_fdir)
        abnormal_img_data = self.read_img(self.abnormal_fdir)

        return [normal_pc_data, normal_img_data], \
               [abnormal_pc_data, abnormal_img_data], \
               [self.normal_fdir, self.abnormal_fdir]