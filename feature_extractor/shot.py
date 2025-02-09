from dataclasses import dataclass
from multiprocessing import Pool
from types import TracebackType

from tqdm import tqdm
import numpy as np

def get_local_rf(
     values):
    """
    Extracts a local reference frame based on the eigendecomposition of the weighted covariance matrix.
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.
    Input:
        points: the given neighbor points of the feature point
        f_point: the feature point
        radius: the radius of the query ball
    """
    point, neighbors, radius = values
    if neighbors.shape[0] == 0:
        return np.eye(3)

    centered_points = neighbors - point

    # EVD of the weighted covariance matrix
    radius_minus_distances = radius - np.linalg.norm(centered_points, axis=1)
    weighted_cov_matrix = (
        centered_points.T
        @ (centered_points * radius_minus_distances[:, None])
        / radius_minus_distances.sum()
    )
    eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov_matrix)

    # disambiguating the axes
    # TODO: deal with the equality case (where the two sums below are equal)
    x_orient = (neighbors - point) @ eigenvectors[:, 2]
    if (x_orient < 0).sum() > (x_orient >= 0).sum():  # the x orient directs towards to the minus direction
        eigenvectors[:, 2] *= -1
    z_orient = (neighbors - point) @ eigenvectors[:, 0]
    if (z_orient < 0).sum() > (z_orient >= 0).sum():
        eigenvectors[:, 0] *= -1
    eigenvectors[:, 1] = np.cross(eigenvectors[:, 0], eigenvectors[:, 2])

    return np.flip(eigenvectors, axis=1)

def get_azimuth_idx(
    x, y
):
    """
    Finds the bin index of the azimuth of a point in a division in 8 bins.
    Bins are indexed clockwise, and the first bin is between pi and 3 * pi / 4.
    To use 4 bins instead of 8, divide each factor by 2 and remove the last one that compares |y| and |x|.
    To use 2 bins instead of 8, repeat the process to only keep the condition (y > 0) | ((y == 0) & (x < 0)).
    """
    a = (y > 0) | ((y == 0) & (x < 0))
    
    return (
        4 * a  # top or bottom half
        + 2 * np.logical_xor((x > 0) | ((x == 0) & (y > 0)), a)  # left or right, true when both are different
        # half of each corner
        + np.where(
            (x * y > 0) | (x == 0),
            np.abs(x) < np.abs(y),
            np.abs(x) > np.abs(y),
        )
    )

def interpolate_on_adjacent_husks(
    distance: np.ndarray[np.float64], radius: float
):
    """
    Interpolates on the adjacent husks.
    Assumes there are only two husks, centered around radius / 4 and 3 * radius / 4.

    Args:
        distance: distance or array of distances to the center of the sphere.
        radius: radius of the neighborhood sphere.

    Returns:
        outer_bin: value or array of values to add to the outer bin.
        Equal to 0 if the point is in the outer bin.
        inner_bin: value or array of values to add to the inner bin.
        Equal to 0 if the point is in the inner bin.
        current_bin: value or array of values to add to the current bin.
    """
    
    radial_bin_size = radius / 2  # normalized distance between two radial neighbor bins
    # external sphere that is closer to radius / 2 than to the border
    inner_bin = (
        ((distance > radius / 2) & (distance < radius * 3 / 4))
        * (radius * 3 / 4 - distance)
        / radial_bin_size
    )
    # internal sphere that is closer to radius / 2 than to a null radius
    outer_bin = (
        ((distance < radius / 2) & (distance > radius / 4))
        * (distance - radius / 4)
        / radial_bin_size
    )
    current_bin = (
        # radius / 4 is the center of the inner bin
        (distance < radius / 2)
        * (1 - np.abs(distance - radius / 4) / radial_bin_size)
    ) + (
        # 3 * radius / 4 is the center of the outer bin
        (distance > radius / 2)
        * (1 - np.abs(distance - radius * 3 / 4) / radial_bin_size)
    )

    return outer_bin, inner_bin, current_bin

def interpolate_vertical_volumes(
    phi: np.ndarray[np.float64], z: np.ndarray[np.float64]
):
    """
    Interpolates on the adjacent vertical volumes.
    Assumes there are only two volumes, centered around pi / 4 and 3 * pi / 4.
    The upper volume is the one found for z > 0.

    Args:
        phi: elevation or array of elevations.
        z: vertical coordinate.

    Returns:
        outer_volume: value or array of values to add to the outer volume.
        inner_volume: value or array of values to add to the inner volume.
        current_volume: value or array of values to add to the current volume.
    """
    
    phi_bin_size = np.pi / 2
    # phi between pi / 2 and 3 * pi / 4
    upper_volume = (
        (
            ((phi > np.pi / 2) | ((np.abs(phi - np.pi / 2) < 1e-10) & (z <= 0)))
            & (phi <= np.pi * 3 / 4)
        )
        * (np.pi * 3 / 4 - phi)
        / phi_bin_size
    )
    # phi between pi / 4 and pi / 2
    lower_volume = (
        (
            ((phi < np.pi / 2) & ((np.abs(phi - np.pi / 2) >= 1e-10) | (z > 0)))
            & (phi >= np.pi / 4)
        )
        * (phi - np.pi / 4)
        / phi_bin_size
    )
    current_volume = (
        # pi / 4 is the center of the upper bin
        (phi < np.pi / 2)
        * (1 - np.abs(phi - np.pi / 4) / phi_bin_size)
    ) + (
        # 3 * pi / 4 is the center of the lower bin
        (phi >= np.pi / 2)
        * (1 - np.abs(phi - np.pi * 3 / 4) / phi_bin_size)
    )

    return upper_volume, lower_volume, current_volume

def compute_single_shot_descriptor(
    values):
    """
    Computes a single SHOT descriptor.----> it means that it only considered a single point as the center point
    Arguments are given in a tuple to allow for multiprocessing using multiprocessing.Pool.

    Args:
        values: (point, neighbors, normals, radius, local_rf, normalize, min_neighborhood_size).

    Returns:
        The SHOT descriptor.
    """
    # the number of bins are hardcoded in this version, passing parameters in a multiprocessed settings gets cumbersome
    n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins = 11, 8, 2, 2

    descriptor = np.zeros(
        (n_cosine_bins, n_azimuth_bins, n_elevation_bins, n_radial_bins)
    )

    (
        point,
        neighbors,
        normals,
        radius,
        eigenvectors,
        normalize,
        min_neighborhood_size,
    ) = values

    rho = np.linalg.norm(neighbors - point, axis=1)    # distance bewteen neigh points to center points
    
    if (rho > 0).sum() > min_neighborhood_size:
        neighbors = neighbors[rho > 0]  
        local_coordinates = (neighbors - point) @ eigenvectors
        cosine = np.clip(normals[rho > 0] @ eigenvectors[:, 2].T, -1, 1)
        rho = rho[rho > 0]

        order = np.argsort(rho)
        rho = rho[order]
        local_coordinates = local_coordinates[order]
        cosine = cosine[order]

        # computing the spherical coordinates in the local coordinate system
        theta = np.arctan2(local_coordinates[:, 1], local_coordinates[:, 0])
        phi = np.arccos(np.clip(local_coordinates[:, 2] / rho, -1, 1))  # arcsin or arccos?

        # computing the indices in the histograms
        cos_bin_pos = (cosine + 1.0) * n_cosine_bins / 2.0 - 0.5   # cos_bin_pos --> (-0.5, 11.5)
        cos_bin_idx = np.rint(cos_bin_pos).astype(int)             # cos_bin_idx --> [0, 11]
        theta_bin_idx = get_azimuth_idx(
            local_coordinates[:, 0], local_coordinates[:, 1]
        )                                                          # theta_bin_idx --> [0, 7]

        # the two arrays below have to be cast as ints, otherwise they will be treated as masks
        phi_bin_idx = (local_coordinates[:, 2] > 0).astype(int)    # phi_bin_idx --> [1, 0]
        rho_bin_idx = (rho > radius / 2).astype(int)               # rho_bin_idx --> [1, 0]

        # interpolation on the local bins
        delta_cos = (
            cos_bin_pos - cos_bin_idx
        )  # normalized distance with the neighbor bin
        delta_cos_sign = np.sign(delta_cos)                        # left-neighbor or right-neighbor
        abs_delta_cos = delta_cos_sign * delta_cos                 # probably faster than np.abs
        # noinspection PyRedundantParentheses
        descriptor[
            (cos_bin_idx + delta_cos_sign).astype(int) % n_cosine_bins,   # it equals to 1 if cos_bin_idx=12, 10 if cos_bin_idx=-1, the neighbor is then properly defined 
            theta_bin_idx,
            phi_bin_idx,
            rho_bin_idx,
        ] += abs_delta_cos * (
            (cos_bin_idx > -0.5) & (cos_bin_idx < n_cosine_bins - 0.5)
        )
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (
            1 - abs_delta_cos
        )

        # interpolation on the adjacent husks
        outer_bin, inner_bin, current_bin = interpolate_on_adjacent_husks(rho, radius)
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 1] += outer_bin * (
            rho_bin_idx == 0
        )
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, 0] += inner_bin * (
            rho_bin_idx == 1
        )
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += current_bin

        # interpolation between adjacent vertical volumes
        upper_volume, lower_volume, current_volume = interpolate_vertical_volumes(
            phi, local_coordinates[:, 2]
        )
        descriptor[cos_bin_idx, theta_bin_idx, 1, rho_bin_idx] += upper_volume * (
            phi_bin_idx == 0
        )
        descriptor[cos_bin_idx, theta_bin_idx, 0, rho_bin_idx] += lower_volume * (
            phi_bin_idx == 1
        )
        descriptor[
            cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx
        ] += current_volume

        # interpolation between adjacent horizontal volumes
        # local_coordinates[:, 0] * local_coordinates[:, 1] != 0
        theta_bin_size = 2 * np.pi / n_azimuth_bins
        delta_theta = np.clip(
            (theta - (-np.pi + theta_bin_idx * theta_bin_size)) / theta_bin_size - 0.5,
            -0.5,
            0.5,
        )
        delta_theta_sign = np.sign(delta_theta)  # left-neighbor or right-neighbor
        abs_delta_theta = delta_theta_sign * delta_theta
        descriptor[
            cos_bin_idx,
            (theta_bin_idx + delta_theta_sign).astype(int) % n_azimuth_bins,
            phi_bin_idx,
            rho_bin_idx,
        ] += abs_delta_theta
        descriptor[cos_bin_idx, theta_bin_idx, phi_bin_idx, rho_bin_idx] += (
            1 - abs_delta_theta
        )

        # normalizing the descriptor to Euclidian norm 1
        if (descriptor_norm := np.linalg.norm(descriptor)) > 0:
            if normalize:
                return descriptor.ravel() / descriptor_norm
            else:
                return descriptor.ravel()
            
    return np.zeros(n_cosine_bins * n_azimuth_bins * n_elevation_bins * n_radial_bins)

@dataclass
class ShotMultiprocessor:
    """
    Base class to compute SHOT descriptors in parallel on multiple processes.
    """

    normalize: bool = True
    min_neighborhood_size: int = 100
    n_procs: int = 8
    disable_progress_bar: bool = False

    def __enter__(self):
        self.pool = Pool(processes=self.n_procs)
        return self

    def __exit__(
        self,
        exc_type: type,
        exc_val: Exception,
        exc_tb: TracebackType,
    ) -> None:
        if exc_type is not None:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()

    def compute_local_rf(
        self,
        keypoints: np.ndarray,
        neighborhoods: np.ndarray[np.object_],
        support: np.ndarray[np.float64],
        radius: float,
    ) -> np.ndarray[np.float64]:
        """
        Parallelization of the function get_local_rf.

        Args:
            support: The supporting point cloud.
            keypoints: The keypoints to compute local reference frames on.
            radius: The radius used to compute the local reference frames.
            neighborhoods: The neighborhoods associated with each keypoint. neighborhoods[i] should be an array of ints.

        Returns:
            The local reference frames computed on every keypoint.
        """
        return np.array(
            list(
                tqdm(
                    self.pool.imap(
                        get_local_rf,
                        [
                            (
                                keypoints[i, :],
                                support[neighborhoods[i]],
                                radius,
                            )
                            for i in range(keypoints.shape[0])
                        ],
                    ),
                    desc=f"Local RFs with radius {radius}",
                    total=keypoints.shape[0],
                    disable=self.disable_progress_bar,
                )
            )
        )

    def compute_descriptor(
        self,
        keypoints: np.ndarray[np.float64],
        normals: np.ndarray[np.float64],
        neighborhoods: np.ndarray[np.object_],
        local_rfs: np.ndarray[np.float64],
        support: np.ndarray[np.float64],
        radius: float,
    ) -> np.ndarray[np.float64]:
        """
        Parallelization of the function compute_single_shot_descriptor.

        Args:
            keypoints: The keypoints to compute descriptors on.
            normals: The normals of points in the support.
            neighborhoods: The neighborhoods associated with each keypoint. neighborhoods[i] should be an array of ints.
            local_rfs: The local reference frames associated with each keypoint.
            support: The supporting point cloud.
            radius: The radius used to compute SHOT.

        Returns:
            The descriptor computed on every keypoint.
        """
        return np.array(
            list(
                tqdm(
                    self.pool.imap(
                        compute_single_shot_descriptor,
                        [
                            (
                                keypoints[i, :],
                                support[neighborhoods[i]],
                                normals[neighborhoods[i]],
                                radius,
                                local_rfs[i],
                                self.normalize,
                                self.min_neighborhood_size,
                            )
                            for i in range(keypoints.shape[0])
                        ],
                        chunksize=int(np.ceil(keypoints.shape[0] / (2 * self.n_procs))),
                    ),
                    desc=f"SHOT desc with radius {radius}",
                    total=keypoints.shape[0],
                    disable=self.disable_progress_bar,
                )
            )
        )

    def compute_descriptor_single_scale(
        self,
        point_cloud: np.ndarray[np.float64],
        normals: np.ndarray[np.float64],
        keypoints: np.ndarray[np.float64],
        neighborhoods, 
        radius: float,
    ) -> np.ndarray[np.float64]:
        """
        Computes the SHOT descriptor on a single scale.
        Normals are expected to be normalized to 1.

        Args:
            point_cloud: The entire point cloud.
            normals: The normals computed on the point cloud.
            keypoints: The keypoints to compute descriptors on.
            radius: Radius used to compute the SHOT descriptors.
            subsampling_voxel_size: Subsampling strength. Leave empty to keep the whole support.

        Returns:
            The descriptor as a (self.keypoints.shape[0], 352) array.
        """
        local_rfs = self.compute_local_rf(
            keypoints=keypoints,
            neighborhoods=neighborhoods,
            support=point_cloud,
            radius=radius,
        )

        return self.compute_descriptor(
            keypoints=keypoints,
            normals=normals,
            neighborhoods=neighborhoods,
            local_rfs=local_rfs,
            radius=radius,
            support=point_cloud,
        )