"""
Implementation from scratch of the FPFH descriptor based on a careful reading of:
R. B. Rusu, N. Blodow and M. Beetz,
Fast Point Feature Histograms (FPFH) for 3D registration,
2009 IEEE International Conference on Robotics and Automation
"""

import numpy as np
from sklearn.neighbors import KDTree
from tqdm import tqdm

class FPFH_S:
    '''
    It calculates the FPFH features of points. 
    Input:
        radius: radius for ball query
        n_bins: bins for each feature
        intensity: whether to use intensity in point cloud
        use_l1: whether to use l1 norm in computing neighbor intensity or not
    '''
    def __init__(self, radius: float, n_bins: int, intensity: bool, 
                 neigh_mode: str, nsample: int, 
                 disable_progress_bars: bool=True,
                 verbose: bool=True):
        
        self.radius = radius
        self.n_bins = n_bins
        self.intensity = intensity
        self.disable_progress_bars = disable_progress_bars
        self.verbose = verbose
        self.neigh_mode = neigh_mode
        if self.neigh_mode=='ball_query':
            self.max_nn = nsample    # maximum neighbor in ball query
        elif self.neigh_mode=='knn': 
            self.nsample = nsample

    def neighbor_query(self, xyzs):
        def neigh_sampling(neigh_idx:np.ndarray,
                           distances:np.ndarray, 
                           nsample:int):
            assert neigh_idx.ndim == 1
            if neigh_idx.shape[0] > nsample:        # if the number of points are beyond the required
                choice = np.random.choice(neigh_idx.shape[0], nsample, replace=False)
                neigh_idx = neigh_idx[choice]
                distances = distances[choice]
            return (neigh_idx, distances)
        
        kdtree = KDTree(xyzs)
        if self.neigh_mode == 'ball_query':
            neighborhoods, distances = kdtree.query_radius(
            xyzs, self.radius, return_distance=True
            )   # neighbor index and corresponding distances
            if self.verbose:
                print("Mean neigh size: {}".format(np.sum([neighborhood.shape[0] for 
                                                           neighborhood in 
                                                           neighborhoods])/xyzs.shape[0])
                )
            if self.max_nn:          # neighbors randomly sampled to certain number
                filtered_neighs = [
                        neigh_sampling(neighborhood, distance, self.max_nn) 
                                for neighborhood, distance
                                in zip(neighborhoods, distances)
                        ]
                neighborhoods = [filtered_neigh[0] for filtered_neigh in filtered_neighs]
                distances = [filtered_neigh[1] for filtered_neigh in filtered_neighs]
        elif self.neigh_mode == 'knn':
            distances, neighborhoods = kdtree.query(
            xyzs, self.nsample, return_distance=True
            )   # neighbor index and corresponding distances
        
        print("Mean neigh size after subsampling: {}".format(
                    np.sum([neighborhood.shape[0] for neighborhood in 
                    neighborhoods])/xyzs.shape[0])
                )
        return neighborhoods, distances

    def fpfh(self, points, normals,):
        # data extraction
        xyzs = points[:, 0:3]
        if self.intensity:
            intensity = points[:, 3]
        # neighbor query
        neighborhoods, distances = self.neighbor_query(xyzs)
        num_neighs = [neighborhood.shape[0] for neighborhood
                         in neighborhoods]
        
        # compute SPFH
        bins_num = 5 if self.intensity else 3  # 5 since naive_rgb and rela_rgb are considered
        spfh = np.zeros(
            (xyzs.shape[0], self.n_bins * bins_num)
        )
        for i, centre in tqdm(
            enumerate(xyzs),
            desc="SPFH",
            total=xyzs.shape[0],
            disable=self.disable_progress_bars,
        ):
            if neighborhoods[i].shape[0] > 0:       # ball query with different number of points
                num_neigh = num_neighs[i]
                # geometric section
                neigh_xyzs = xyzs[neighborhoods[i]]
                neighbors_normals = normals[neighborhoods[i]]
                centered_neighbors = neigh_xyzs - centre
                dist = np.linalg.norm(centered_neighbors, axis=1)
                u = normals[i]
                v = np.cross(centered_neighbors[dist > 0], u)
                w = np.cross(u, v)
                alpha = np.einsum("ij,ij->i", v, neighbors_normals[dist > 0])
                phi = centered_neighbors[dist > 0].dot(u) / dist[dist > 0]
                theta = np.arctan2(
                    np.einsum("ij,ij->i", neighbors_normals[dist > 0], w),
                    neighbors_normals[dist > 0].dot(u),
                )
                alpha_hist = np.histogram(alpha, bins=self.n_bins, range=(-1, 1), )[0]
                phi_hist = np.histogram(phi, bins=self.n_bins, range=(-1, 1), )[0]
                theta_hist = np.histogram(theta, bins=self.n_bins, range=(-np.pi/2, np.pi/2))[0]

                if self.intensity:
                    neigh_inten = intensity[neighborhoods[i]]
                    center_inten = intensity[i]
                    naive_inten = neigh_inten
                    rela_inten = abs(neigh_inten-center_inten)
                    naive_inten_hist = np.histogram(
                        naive_inten, bins=self.n_bins, range=(0, 1)
                        )[0]
                    rela_inten_hist = np.histogram(
                        rela_inten, bins=self.n_bins, range=(0, 1)
                        )[0]
                    spfh[i, :] = np.concatenate([
                        alpha_hist, 
                        phi_hist, 
                        theta_hist,
                        naive_inten_hist,
                        rela_inten_hist])/num_neigh 
                
                else:
                    spfh[i, :] = np.concatenate([
                        alpha_hist, 
                        phi_hist, 
                        theta_hist])/num_neigh 
                
        spfh = spfh.reshape(xyzs.shape[0], -1)    # [N, n_bins*3] or [N, n_bins*features_dim]
        fpfh = np.zeros(
            (xyzs.shape[0], self.n_bins*bins_num)
        )
        # averange over spfh feature for fpfh feature
        for i, neighborhood in tqdm(
            enumerate(neighborhoods),
            desc="FPFH",
            total=xyzs.shape[0],
            delay=0.5,
            disable=self.disable_progress_bars,
        ):
            with np.errstate(invalid="ignore", divide="ignore"):
                fpfh[i] = (
                    spfh[i]
                    # should be ok to encounter a RuntimeWarning here since we apply a mask after the divide
                    + (spfh[neighborhood] / distances[i][:, None])[
                        distances[i] > 0
                    ].sum(axis=0)
                    / num_neighs[i]
                )
        return fpfh

if __name__ == '__main__':
    # shit = np.random.rand(500, 3)
    # kdtree = KDTree(shit)
    # neighborhoods, distances = kdtree.query(
    #     shit, 64, return_distance=True
    # )
    # print(neighborhoods[0].shape, distances[0].shape)
    # print(neighborhoods[1].shape, distances[1].shape)
    points = np.random.rand(1000, 4)
    normals = np.random.rand(1000, 3)
    fpfh_test = FPFH_S
    fpfh_init = fpfh_test(radius=0.8, n_bins=11, intensity=True,
                     neigh_mode='ball_query', nsample=500,
                     use_l1=False, disable_progress_bars=False)
    # fpfh_test = FPFH_S(radius=0.8, n_bins=11, intensity=False,
    #                  neigh_mode='ball_query', nsample=500,
    #                  use_l1=False, disable_progress_bars=False)
    feature = fpfh_init.fpfh(points=points, normals=normals)
    print(feature.shape)