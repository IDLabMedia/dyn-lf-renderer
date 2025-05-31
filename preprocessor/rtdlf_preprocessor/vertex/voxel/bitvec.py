import jax.numpy as jnp

from jax import jit


def compute_voxel_space_params(voxel_space: jnp.ndarray, grid_size: float):
    """Compute the fixed voxel space parameters for reuse."""
    # Extract the min and max points from voxel_space
    min_corner = voxel_space[0]
    max_corner = voxel_space[1]

    # Compute the voxel grid dimensions
    grid_shape = jnp.ceil((max_corner - min_corner) / grid_size).astype(jnp.int64)

    # Compute the total number of voxels in the grid
    total_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]

    # Compute maximum allowed value for the index
    max_cols = jnp.uint64(jnp.iinfo(jnp.int32).max)

    # Compute 2D bitvector dimensions
    bitvec_rows = (total_voxels + max_cols - 1) // max_cols
    bitvec_cols = min(int(total_voxels), max_cols)

    return (
        min_corner,
        grid_shape.astype(jnp.int32),
        max_cols,
        int(bitvec_rows.astype(jnp.int32)),
        bitvec_cols,
    )


def vcl_to_bitvec(
    voxel_space_params, grid_size: float, vcl: jnp.ndarray
) -> jnp.ndarray:
    """Transform an absolute voxel cloud to a bitvector representation.

    :param voxel_space_params: Precomputed voxel space parameters including:
        min_corner, grid_shape, max_int32, bitvec_rows, bitvec_cols.
    :param grid_size: The spacing between 2 neighboring voxel grid points.
    :param vcl: The voxel cloud, represented with absolute coordinates.

    :returns: (n,m) bitvector, represented by uint8.
    """
    # Unpack precomputed parameters
    print(voxel_space_params)
    (
        min_corner,
        grid_shape,
        max_cols,
        bitvec_rows,
        bitvec_cols,
    ) = voxel_space_params

    # Compute voxel indices for the voxel cloud
    voxel_indices = jnp.floor((vcl - min_corner) / grid_size).astype(jnp.int32)

    # Compute flat voxel index
    flat_indices = voxel_indices[:, 0] + grid_shape[0] * (
        voxel_indices[:, 1] + grid_shape[1] * voxel_indices[:, 2]
    )

    # Compute 2D bitvector indices
    row_indices = (flat_indices // max_cols).astype(jnp.int32)
    col_indices = (flat_indices % max_cols).astype(jnp.int32)

    # Initialize 2D bitvector
    bitvector = jnp.zeros((bitvec_rows, bitvec_cols), dtype=jnp.uint8)
    bitvector = bitvector.at[row_indices, col_indices].set(1)

    return bitvector


def bitvec_to_vcl(
    voxel_space_params, grid_size: float, bitvector: jnp.ndarray
) -> jnp.ndarray:
    """Inverse operation: Convert bitvector back to voxel cloud in absolute coordinates.

    :param voxel_space_params: Precomputed voxel space parameters including:
        min_corner, max_corner, grid_shape, total_voxels, max_int32, bitvec_rows, bitvec_cols.
    :param grid_size: The spacing between 2 neighboring voxel grid points.
    :param bitvector: The bitvector representation of the voxel grid.

    :returns: Voxel cloud in absolute coordinates (n, 3) where each point corresponds to a voxel.
    :rtype: jnp.ndarray
    """
    # Unpack precomputed parameters
    (
        min_corner,
        grid_shape,
        max_int32,
        bitvec_rows,
        bitvec_cols,
    ) = voxel_space_params

    # Unpack the bitvector to a 1D array
    unpacked_bitvector = jnp.unpackbits(bitvector.flatten())

    # Find the indices of the "1" bits in the bitvector (the active voxels)
    flat_voxel_indices = jnp.nonzero(unpacked_bitvector)[0]

    # Reverse the voxel hash to get voxel grid indices
    voxel_indices = jnp.zeros((flat_voxel_indices.shape[0], 3), dtype=jnp.int32)

    voxel_indices = voxel_indices.at[:, 0].set(
        flat_voxel_indices // (grid_shape[1] * grid_shape[2])
    )
    voxel_indices = voxel_indices.at[:, 1].set(
        (flat_voxel_indices % (grid_shape[1] * grid_shape[2])) // grid_shape[2]
    )
    voxel_indices = voxel_indices.at[:, 2].set(flat_voxel_indices % grid_shape[2])

    # Convert voxel indices back to absolute coordinates
    vcl = voxel_indices * grid_size + min_corner

    return vcl
