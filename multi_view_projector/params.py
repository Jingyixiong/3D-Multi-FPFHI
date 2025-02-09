'''
    Save the parameters for conducting multi-view projection.
'''
import numpy as np
# 1. Real arch
data_root_realarch = '/home/jing/Desktop/JING_PhD/DIANA/Jing_project/new_arch_shell_final_results/'
save_root_realarch = '/media/jing/Storage/anomaly_dataset/al_general_infra/real_arch_test'
realarch_normal_nlist = [
        'Sarch_130305', 
        'Narch_130305'
    ]    # the arch which is used for reference
realarch_abnormal_nlist = [                      # S: sourtern arch, _130305: timestep
    'Sarch_131123',   
    'Narch_131123'
    ]
realarch_focal_list = {
    'Narch_131123': 450,
    'Narch_130305': 450,
    'Sarch_130305': 450,
    'Sarch_131123': 450
    }
# if mismatching is found in render images(open3d)
    # and projected points(cv2.projectPoints), please
    # check the extrinsic matrix to make sure Rot is correct 
realarch_extrinsic_list = {
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
realarch_vis_params = {
			"field_of_view" : 60.0,
			"front":[0, 0, 1],
			"lookat":[0.251, 0.03, -1.151],
			"up" : [0.484, 0.875, 0.003],
			"zoom" : 0.480
            }

# 2. Synthetic arch
data_root_synarch = '/home/jing/Desktop/JING_PhD/DIANA/Jing_project/updated_arch_shell_results'
save_root_synarch = '/media/jing/Storage/anomaly_dataset/al_general_infra/syn_arch'
synarch_normal_nlist = ['min_cw0',]  #  'min_cw0.4'      
synarch_abnormal_nlist = [ 
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
    'disp_z_0.4m',
    'disp_xz',
    'rot_x'
    ]
synarch_focal_list = {
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
    'disp_z_0.4m': 700,
    'rot_x': 700,
    }
# if mismatching is found in render images(open3d)
# and projected points(cv2.projectPoints), please
# check the extrinsic matrix to make sure Rot is correct 
disp_x_extrinsic = np.array([
        [ -1.,  0.,  0.,      0],
        [-0.,  -1., -0.,   1.80],
        [-0.,  -0.,   1.,  8.13],
        [ 0.,   0.,  0.,     1.]
        ])

synarch_extrinsic_list = {
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
    'disp_z_0.4m': np.array([
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

synarch_vis_params = {
        "field_of_view":60.0,
        "front":[-0.495, -0.738, 0.459],
        "lookat":[-0.267, -0.341, -1.140],
        "up":[ 0.230, 0.398, 0.888],
        "zoom":0.480
        }

# 3. real tunnel
realtunnel_normal_nlist = ['0']    # the init state of tunnel
realtunnel_abnormal_nlist = [                    
    '76', '89', '96', '103'
    ]                              # different loading steps
realtunnel_vis_params = {
			"field_of_view":60.0,
			"front":[-0.641, -0.095, -0.762],
			"lookat":[-0.941, -0.044, -0.257],
			"up":[-0.452, -0.755, 0.474],
			"zoom":0.5
            }
