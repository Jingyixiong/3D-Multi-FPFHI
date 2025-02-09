import os 

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

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
    
def readXYZfile(f_dir, sep=' '):
    '''
    This is for extracting data from real point cloud
    '''
    arch_n = os.listdir(f_dir)
    arch_n = [i for i in arch_n if i.split('_')[-1]=='arch.asc']
    f_path = os.path.join(f_dir, arch_n[0])
    points = pd.read_csv(f_path, sep=sep)
    points = points.astype('float64').values[:, 0:3]
    points = pc_normalize(points, norm_method='m3md')
    
    return points, points, None, None

def readXYZfile_2(f_dir):
    '''
        This function is used to extract data from the 
    '''
    scan_fns = os.listdir(f_dir)
    if len(scan_fns) == 0:
        raise Exception('No file is included into the given dir!')
    scan_f_dirs = [os.path.join(f_dir, scan_fn) for scan_fn in scan_fns if 
                   os.path.splitext(scan_fn)[-1] == '.csv']
    xyzs_list, noise_xyzs_list, rgbs_list, labels_list = [], [], [], []
    for i, scan_f_dir in enumerate(scan_f_dirs):
        frame = pd.read_csv(scan_f_dir, sep=';')
        xyzs = np.array(frame.loc[:, ['X', 'Y', 'Z']].values)
        rgbs = np.array(frame.loc[:, ['red', 'green', 'blue']].values)
        labels = np.array(frame.loc[:, ['categoryID']].astype('int32').values)
        
        xyzs_list.append(xyzs)
        rgbs_list.append(rgbs)
        labels_list.append(labels)
        try:
            noise_xyzs = np.array(frame.loc[:, ['X_noise', 'Y_noise', 'Z_noise']].values)
            noise_xyzs_list.append(noise_xyzs)
            noise_xyzs = np.vstack(noise_xyzs_list)
        except:
            noise_xyzs = 0
            pass
        
    xyzs = np.vstack(xyzs_list)
    rgbs = np.vstack(rgbs_list)
    labels = np.vstack(labels_list)
    assert xyzs.shape[0] == rgbs.shape[0] == labels.shape[0]
    return xyzs, noise_xyzs, rgbs, labels


def extract_neighidx(pc, radius):
    '''
    Return list of array for the nearest neighbors.
    Input:
        pc: it is the point cloud with o3d.geometry.PointCloud data format
    
    '''
    N, _ = np.asarray(pc.points).shape
    pc_tree = o3d.geometry.KDTreeFlann(pc)
    idx_list = [pc_tree.search_radius_vector_3d(pc.points[i], radius)[1][1:]
               for i in range(N)]
    return idx_list

def rgb(value):
    '''
    Generate the value for the color maps
    '''
    ratio = 2*value
    b = np.maximum(0, 1 - ratio)
    r = np.maximum(0, ratio - 1)
    g = 1 - b - r
    return np.concatenate([r[:, None], g[:, None], b[:, None]], axis=-1)

def colfunc(val, startcolor, stopcolor):
    """ Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the one returned are
        composed of a sequence of N component values (e.g. RGB).
    """
    minval, maxval = np.min(val), np.max(val)
    f = (val-minval)/(maxval-minval)
    return np.stack([f*(b-a)+a for (a, b) in zip(startcolor, stopcolor)], axis=1)

def colfunc_2(val, threshold=0.5, uni_color=np.array([255, 0, 0])):
    """ 
    Return a array which contains normalized [r, g, b].
    A threshold is used to seperate the color bar.
    Over the threshold, the point will be set to red.
    Below, a color bar will be used to represent it. 
    """
    # Normalize to 0-1
    minval, maxval = np.min(val), np.max(val)
    f = (val-minval)/(maxval-minval)
    colors = np.zeros([f.shape[0], 3])
    # colors_below = np.zeros([f.shape[0], 3])  # colors below the threshold
    colors[f > threshold] = uni_color/255
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    #     "custom", [
    #         "royalblue", "lime", "yellow", "red", "darkred"
    #         ]
    #     )
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom", [
            'royalblue', 'cornflowerblue', 'indianred', 'red',
            ],
        gamma=0.5
        )
    # cmap = matplotlib.colormaps[cmap_n]
    colors_below = cmap(f[f <= threshold])
    colors[f <= threshold] = colors_below[:, 0:3]
    return colors, f

def best_threshold(recalls_list, precisions_list, 
                   f1s_list, abscore_coefs):
    '''
        It returns the best threshold for x-displacement case.
    '''
    # set all font to be Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    # Figure plot
    figure, ax_list = plt.subplots(1, 4)
    ax1, ax2, ax3, ax4 = ax_list
    disp_n = ['$\it{x}$', '$\it{z}$', '$\it{x}$+$\it{z}$', 
              'Combined $\it{x}$ rotation and $\it{z}$ \ntranslation']

    for i, ax in enumerate(ax_list):
        if i > 0:
            ax.get_yaxis().set_visible(False)
        f1s = f1s_list[i]
        recalls = recalls_list[i]
        precisions = precisions_list[i]

        ax.plot(abscore_coefs, f1s, 'k')
        ax.set_title('{} displacement'.format(disp_n[i]), 
                     fontsize=18)
        # ax.title.set_text('{} displacement'.format(disp_n[i]), fontsize=18)

        best_idx = np.argmax(f1s)
        best_f1_x = abscore_coefs[best_idx]       # Best F1 score being selected 
        ax.axvline(best_f1_x, color='r')
        best_f1_x = round(best_f1_x, 3)
        if i == 1:
            bias = 0.15
        else:
            bias = 0.05
        ax.text(best_f1_x+bias, 0.9, '$s=$'+str(best_f1_x), rotation=0, fontsize=18)

        print('For {} the best F1 score, recall and precision are {}, {} and {}'.format(disp_n[i],
                                                                                        recalls[best_idx],
                                                                                precisions[i],
                                                                                        f1s[best_idx]))
    figure.supxlabel('$s$', fontsize=18)
    figure.supylabel('F1 score', fontsize=18)
    plt.rcParams['savefig.dpi']=600
    plt.setp((ax1, ax2, ax3, ax4), ylim=[0, 1.0])
    plt.show()
    return True

def voxelization(points: np.ndarray, voxel_size:float)-> np.ndarray:
    xyzs = points[:, 0:3]
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(xyzs)
    sub_pc_o3d, _, idx_list = pc_o3d.voxel_down_sample_and_trace(voxel_size=voxel_size,
                                            min_bound=pc_o3d.get_min_bound(),
                                            max_bound=pc_o3d.get_max_bound())   # downsampling points
    idx_list = [idx[0] for idx in idx_list]       # the point index which is closet to the voxel centre
    points = points[idx_list]

    return points, idx_list

def subsample_feature_label(feature: np.ndarray, label: np.ndarray, 
                            nsample:int, random_seed:int=16)-> np.ndarray:
    '''
    This function is used to subsample the feature and label
    '''
    np.random.seed(random_seed)
    normal_flag = (label==0)   
    normal_points = feature[normal_flag, :] 
    abnoral_flag = (label==1)
    abnormal_points = feature[abnoral_flag, :]
    
    num_points = min(nsample, abnormal_points.shape[0])
    
    normal_sample = np.random.choice(normal_points.shape[0], num_points, replace=False)
    print(normal_sample.shape)
    abnormal_sample = np.random.choice(abnormal_points.shape[0], num_points, replace=False)
    print(abnormal_sample.shape)
    sub_features = np.concatenate((
        normal_points[normal_sample], 
        abnormal_points[abnormal_sample]), axis=0)
    sub_label = np.concatenate((
        np.zeros(num_points), 
        np.ones(num_points)), axis=0)
    
    return sub_features, sub_label

def save_csv(detector, s, save_dir, save_name):
    '''
    Save the point cloud results into the csv files,
    '''
    pc_disp = detector.pc_disp
    mask_disp = np.expand_dims(
        detector.mask_disp, axis=1
        )
    color_disp = detector.color_disp
    anomaly_c, _ = colfunc_2(s)      # color transferred from the anomaly score
    s = np.expand_dims(s, axis=1)
    data = np.concatenate((pc_disp, mask_disp, anomaly_c, s), axis=1)
    data_keys = ['x', 'y', 'z', 'mask', 'r', 'g', 'b', 'abscore']
    data = pd.DataFrame(data, columns=data_keys)
    save_path = os.path.join(save_dir, save_name)
    data.to_csv(save_path, index=False)
    return True

def save_csv_company_proj(detector, s, save_dir):
    '''
    Save the point cloud results into the csv files,
    '''
    color_dict = {    # r, g, b
            'non_crack': np.array([65, 105, 225])/255, 
            'crack': np.array([1, 0, 0]), 
            }
    pc_disp = detector.pc_disp
    mask_disp = detector.mask_disp
    coords_keys = ['x', 'y', 'z']
    color_keys = ['r', 'g', 'b']
    # define 3 different color maps
    # (1) Binary
    color_binary = np.zeros((mask_disp.shape[0], 3))
    color_binary[mask_disp==0, :] = color_dict['non_crack']
    color_binary[mask_disp==1, :] = color_dict['crack']
    color_disp = detector.color_disp
    color_anomaly, _ = colfunc_2(s)      # color transferred from the anomaly score

    coords = pd.DataFrame(pc_disp, columns=coords_keys)
    coords.to_csv(os.path.join(save_dir, 'coords.txt'), index=False)
    bi_color = pd.DataFrame(color_binary, columns=color_keys)
    real_color = pd.DataFrame(color_disp, columns=color_keys)
    anomaly_color = pd.DataFrame(color_anomaly, columns=color_keys)
    bi_color.to_csv(os.path.join(save_dir, 'bi_color.txt'), index=False)
    real_color.to_csv(os.path.join(save_dir, 'real_color.txt'), index=False)
    anomaly_color.to_csv(os.path.join(save_dir, 'anomaly_color.txt'), index=False)

    # save_path = os.path.join(save_dir, save_name)
    # data.to_csv(save_path, index=False)
    return True


def histogram_s(masks, s_dict, 
                inspection_types=['FPFH','FPFH_naiveRGB','FPFH_relaRGB','rgb_img','rgb_img_FPFH']
                ):
    '''
    This function is used to visualize the histogram of the anomaly scores
    '''
    sns.set_style('white')
    kwargs = dict(
        bins=20, 
        kde=True,
        line_kws={'linewidth':2},
        edgecolor='white',
        shrink=0.8,
        # hue="species", element="poly"
        )       #  general figure setting

    fig, axes = plt.subplots(1, 5, )
    new_inspection_ns = [
        '(a) FPFH', '(b) FPFH+simple', '(c) FPFH + '+ r'$ L_{1}$'+'-'+r'$norm$', 
        '(d) 2D feature', '(e) CMPF'
        ]
    for i in range(len(inspection_types)):
        ax = axes[i]
        title = new_inspection_ns[i]
        ax.set_title(title, fontsize=12)
        s = s_dict[inspection_types[i]]
        non_crack_s = s[masks == 0]
        crack_s = s[masks == 1]

        sns.histplot(
            non_crack_s, 
            ax=ax,     # color=non_crack_c,     
            element="bars", 
            label='Non-crack', facecolor=np.array([163, 201, 226, 200])/255,**kwargs
            )
        sns.histplot(
            crack_s, 
            ax=ax,     # color=non_crack_c,     
            element="bars", 
            label='Crack', facecolor=np.array([242, 171, 171, 200])/255,
            **kwargs
            )
        # hide y label
        ax.get_yaxis().set_visible(False)
    
    # set unique legend
    line, label = ax.get_legend_handles_labels()    # the last ax
    fig.legend(line, label, loc='upper right') 
    
    # set unique x label
    fig.supxlabel("Anomaly score ($\it{s}$)", fontsize=14)
    # plt.ylabel("common Y")
    plt.show()
    return True
