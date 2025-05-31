import jax.numpy as jnp
import numpy as np


def sort_indices(pcl: jnp.ndarray) -> jnp.ndarray:
    """Get the sort indices of an (n,3) point cloud"""
    return jnp.lexsort((pcl[:, 2], pcl[:, 1], pcl[:, 0]))


def intersect2d(ar1: jnp.ndarray, ar2: jnp.ndarray) -> jnp.ndarray:
    """Takes an (m,n) and a (o,n) array, and returns a (p,n) array.
    It takes the intersection along axis 1.
    """
    return jnp.array(intersect2d_np(np.array(ar1), np.array(ar2)))


def intersect2d_np(ar1: np.ndarray, ar2: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    cols = ar1.shape[1]
    row_structure = np.dtype(
        {  # create special dtype so that numpy can detect a row as single element
            "names": ["f{}".format(i) for i in range(cols)],
            "formats": cols * [ar1.dtype],
        }
    )

    # take intersection
    intersection = np.intersect1d(ar1.view(row_structure), ar2.view(row_structure))
    return intersection.view(ar1.dtype).reshape(-1, cols)


def difference2d(ar1: jnp.ndarray, ar2: jnp.ndarray) -> jnp.ndarray:
    """Takes an (m,n) and a (o,n) array, and returns a (p,n) array.
    It takes the difference along axis 1 (ar1 \\ ar2).
    """
    return jnp.array(difference2d_np(np.array(ar1), np.array(ar2)))


def difference2d_np(ar1: np.ndarray, ar2: np.ndarray) -> np.ndarray:
    cols = ar1.shape[1]
    row_structure = np.dtype(
        {  # create special dtype so that numpy can detect a row as single element
            "names": ["f{}".format(i) for i in range(cols)],
            "formats": cols * [ar1.dtype],
        }
    )

    # take intersection
    diff = np.setdiff1d(ar1.view(row_structure), ar2.view(row_structure))
    return diff.view(ar1.dtype).reshape(-1, cols)
