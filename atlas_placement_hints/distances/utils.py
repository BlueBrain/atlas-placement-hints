"""
Utility functions to compute distances to boundaries.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from atlas_commons.typing import FloatArray, NDArray

if TYPE_CHECKING:  # pragma: no cover
    import trimesh  # type: ignore

RAY_TRIANGLE_COST = 51.3  # bytes, see https://bbpteam.epfl.ch/project/issues/browse/NSETM-1670


def _get_split_count(memory_bound: int, chunk_length: int, ray_count: int, face_count: int) -> int:
    """
    Compute the number of batches which are required to compute all ray-mesh interections.

    Each batch should not contain more than `chunk_length` rays and the memory required for
    the corresponding computations in batch should fit into `memory_bound` GBs.

    Args:
        memory_bound: available RAM in Gigabytes.
        chunk_length: the target length of each batch of rays to be interseced with
            `face_count` triangle faces.
        ray_count: the total number of rays to be intersected with `face_count` triangle faces.
        face_count: the number of triangle faces in the mesh to intersected with the rays.

    Returns:
        the minimal number of batches satisfying the above constraints.
    """
    maximal_length = (1e9 * memory_bound) / (face_count * RAY_TRIANGLE_COST)
    chunk_length = min(int(maximal_length), chunk_length)

    return max(ray_count // chunk_length, 1)


def memory_efficient_intersection(
    intersector: "trimesh.ray.ray_triangle.RayMeshIntersector",
    ray_origins: NDArray[np.integer],
    ray_directions: FloatArray,
    memory_bound: int = 150,
    chunk_length: int = 100000,
) -> Tuple[NDArray[np.integer], NDArray[np.integer], NDArray[np.integer]]:
    """
    Split the computations of ray intersections using several chunks of a
    specified length.
    It is slower than getting intersections directly but costs less memory.

    Args:
        intersector: Ray-mesh intersector.
        ray_origins: array of shape (N, 3). Origins to cast rays from.
        ray_directions: array of shape (N, 3). Directions in which to cast rays.
        chunk_length: (Optional) the number of rays to calculate intersections for
            at a time. Defaults to 100'000. Can be automatically lowered if it
            the amount of required memory exceeds `memory_bound`.
        memory_bound: (Optional) amount of available RAM in Gigabytes. The actual
            length of a chunk of rays is the minimum of `chunk_length` and the maximal
            length that can fit into `memory_bound` GBs.

    Returns:
        A tuple (locations, ray_ids, tri_ids).
        locations: array (N, 3): locations of intersections.
        ray_ids: array (N, 1): ids of intersecting rays.
        tri_ids: array (N, 1): ids of mesh triangles intersecting a ray.
    """

    locations: NDArray[np.integer] = np.array([[]], dtype=int)
    locations = np.reshape(locations, (0, 3))
    ray_ids: NDArray[np.integer] = np.array([], dtype=int)
    tri_ids: NDArray[np.integer] = np.array([], dtype=int)

    split_count = _get_split_count(
        memory_bound,
        chunk_length,
        ray_origins.shape[0],
        intersector.mesh.faces.shape[0],
    )

    ray_ids_offset = 0
    for ray_pos, ray_dir in zip(
        np.array_split(ray_origins, split_count),
        np.array_split(ray_directions, split_count),
    ):
        locs, rays, tris = intersector.intersects_location(ray_pos, ray_dir, multiple_hits=False)
        rays = rays + ray_ids_offset
        locations = np.vstack([locations, locs])
        ray_ids = np.hstack([ray_ids, rays])
        tri_ids = np.hstack([tri_ids, tris])
        ray_ids_offset += len(ray_pos)

    return locations, ray_ids, tri_ids
