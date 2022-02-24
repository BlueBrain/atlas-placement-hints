"""
Utility functions for the computation of placement hints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import voxcell
from atlas_commons.typing import BoolArray, FloatArray, NDArray, NumericArray
from scipy.ndimage import correlate

# I don't know why, but pylint believes scipy.spatial.ConvexHull is a myth
# pylint: disable=E0611
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)


def is_obtuse_angle(vector_field_1: NumericArray, vector_field_2: NumericArray) -> BoolArray:
    """
    Returns a mask indicating which vector pairs form an obtuse angle.

    Arguments:
        vector_field_1: 3D vector field, i.e., numeric array of shape
            (M, N, ..., 3).
        vector_field_2: 3D vector field, i.e., numeric array of shape
            (M, N, ..., 3).
    Returns:
       Binary mask of shape (M, N, ...) indicating which pairs of vectors
        form an obtuse angle.
    """
    return np.sum(vector_field_1 * vector_field_2, axis=-1) < 0


def centroid_outfacing_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Returns a mesh made of the faces that face away from `mesh`'s centroid.
    """
    toward_centroid = mesh.centroid - mesh.triangles_center
    point_away_from_centroid = is_obtuse_angle(mesh.face_normals, toward_centroid)
    away_faces = mesh.faces[point_away_from_centroid]

    return trimesh.Trimesh(vertices=mesh.vertices, faces=away_faces)


def save_placement_hints(
    distances: FloatArray,
    output_dir: str,
    voxel_data: voxcell.VoxelData,
    layer_names: List[str],
):
    """
    Convert distances to meshes wrt to direction vectors into placement hints
    and save these hints into a nrrd files, one for each layer.

    A placement hint of a voxel is a (lowest layer bottom)-to-layer distance wrt to
    the voxel direction vector. The last axis specifies if it is the distance
    to the layer bottom (0) or to the layer top (1).

    Args:
        distances: 4D array of shape (number-of-layers + 1, length, width, height)
            holding the signed distances from voxel centers to layer tops wrt to voxel direction
            vectors.
        output_dir: directory in which to save the placement hints nrrd files.
        voxel_data: VoxelData object of the region for which placement hints are computed,
            or any VoxelData object with the same offset and same voxel dimensions.
        layer_names: list of layer names used to compose the placement hints file names.
    """
    # [PH]y holds, for each voxel, the distance from the bottom of the atlas region to the voxel
    # along its direction vector (non-negative value).
    y = -distances[-1]  # pylint: disable=invalid-name
    L.info("Saving placement hints [PH]y to file ...")
    placement_hints_y_path = str(Path(output_dir, "[PH]y.nrrd"))
    voxel_data.with_data(y).save_nrrd(placement_hints_y_path)
    L.info("Saving placement hints for each layer to file ...")
    for index, name in enumerate(layer_names):
        bottom = distances[index + 1]
        top = distances[index]
        # A placement hint array is a 1D array of size `number of layers` of float arrays of
        # shape (W, H, D, 2).
        # The description of a placement hint [PH]layer_i is quite convoluted.
        # (TODO: check if distances to boundaries could be used directly).
        # Given a voxel v, let L be the line passing through v along the direction vector of v. The
        # line L intersects the bottom of the deepest layer (e.g., layer 6 for the AIBS mouse
        # isocortex) in a voxel w. Then [PH]_layer_i.raw[..., 0] gives the (non-negative) distance
        # of w to the bottom of layer i while [PH]_layer_i.raw[..., 1] gives the (non-negative)
        # distance of w to the top of layer i.
        # Example: both arrays [PH]_layer_i.raw[...,0] and [PH]_layer_i.raw[..., 1] are constant
        # for O1 column atlases.
        # Note that [PH]_layer_i.raw[...,0] - [PH]y.raw holds the non-negative distances wrt to
        # direction vectors of all voxels to the bottom of layer i (these are non-positive values
        # for the voxels lying inside layer_i). [PH]_layer_i.raw[..., 1] - [PH]y.raw holds the
        # distances wrt to direction vectors of all voxels to the top of layer i (these are
        # non-negative values for voxels lying inside layer_i).
        placement_hints = np.stack((bottom, top), axis=-1) + y[..., np.newaxis]
        layer_placement_hints_path = str(Path(output_dir, f"[PH]{name}.nrrd"))
        voxel_data.with_data(placement_hints).save_nrrd(layer_placement_hints_path)


def detailed_mesh_mask(mesh: trimesh.Trimesh, shape: Tuple[int, ...]) -> BoolArray:
    """
    Generate a mask for the voxels occupied by `mesh`.

    The vertex coordinates of the mesh are assumed to be expressed in voxel index space and
    the dimensions of this space are given by `shape`.

    Arguments:
        mesh: the mesh in question
        shape: the desired shape of the mask

    Returns:
        boolean mask of the voxels occupied by mesh. The shape of `mask` is `shape`.
    """

    def points_in_between(p_1, p_2) -> FloatArray:
        """
        Create points at regular intervals between `p_1` and `p_2`.

        The 3D points `p_1` and `p_2` are expressed in voxel index space and hence have
        non-negative float coordinates. If the distance between them is larger than 1.0 (edge
        length of a voxel in voxel index space), regularly spaced points are inserted between `p_1`
        and `p_2`.

        Args:
            p_1: NDArray[float] of shape (3,), the 3D coordinates of a vertex in voxel index space
            p_2: NDArray[float] of shape (3,), the 3D coordinates of a vertex in voxel index space

        Returns:
            A float array of shape (N, 3) where N is the number of new points inserted on
            the edge [p_1, p_2]. The integer N depends on the distance between p_1 and p_2.
        """
        distance = np.linalg.norm(p_1 - p_2)

        return np.linspace(p_1, p_2, int(np.ceil(distance)) + 1)

    points = []
    for triangle in mesh.triangles:
        edge1 = points_in_between(triangle[0], triangle[1])
        edge2 = points_in_between(triangle[1], triangle[2])
        edge3 = points_in_between(triangle[0], triangle[2])
        points.append(edge1)
        points.append(edge2)
        points.append(edge3)
        for p_1 in edge1:
            for p_2 in edge2:
                points.append(points_in_between(p_1, p_2))

    mesh_voxels = indexable(np.concatenate(points, axis=0))  # round coordinates
    mask = np.zeros(shape, dtype=bool)
    mask[mesh_voxels] = True

    return mask


def get_convex_hull_boundary(mask: BoolArray) -> trimesh.Trimesh:
    """Get the convex hull boundary of the volume defined by `mask`

    Args:
        mask: boolean mask defining a 3D volume

    Returns:
        A surface mesh representing the boundary of the convex hull of the
        `mask` volume.
    """
    convex_hull = ConvexHull(np.array(np.nonzero(mask)).T)

    return trimesh.Trimesh(vertices=convex_hull.points, faces=convex_hull.simplices)


def indexable(
    positions: FloatArray,
) -> Tuple[NDArray[np.int16], ...]:
    """
    Convert a float array of shape (N, 3) into a 3-tuple of voxel indices.

    The array `positions` holds 3D points whose coordinates are expressed in voxel index
    space. This function rounds coordinates to integers and returns the corresponding voxel
    indices under the form of a tuple (X, Y, Z).

    Args:
        positions: float array of shape (N, 3).

    Returns:
        A tuple (X, Y, Z) where each component is a vector of shape (N,)
        and of dtype equal to np.uint16.


    """
    return tuple(
        np.asarray(positions[..., ax], dtype=np.int16) for ax in range(positions.shape[-1])
    )


def clip_mesh(
    mesh: "trimesh.Mesh",
    mask: BoolArray,
    remainder: bool = False,
    dilation: int = 5,
) -> trimesh.Trimesh:
    """
    Get the parts of a mesh which are in the space of `mask`'s volume or the remainder.

    Args:
        mesh: the mesh to clip. Vertex coordinates are assumed to be expressed wrt to voxel index
            space.
        mask: the mask used when clipping.
        remainder: (optional) if True, the complement of the (dilated) mask is used for clipping.
            Defaults to False.
        dilation: edge length, expressed in number of voxels, of the box used to dilate `mask`
            prior to clipping.

    Returns:
        The mesh obtained by intersecting `mesh` with the `mask` volume.
    """
    # Dilate the mask slightly before use
    mask = correlate(mask, np.ones([dilation] * 3)) > 0

    if remainder:
        mask = ~mask

    # Keep every vertex sitting in `mask`
    vertex_mask = mask[indexable(mesh.vertices)]
    vertex_indices = np.nonzero(vertex_mask)[0]

    # Keep every face whose vertices and whose center are in `mask`
    face_mask = np.all(np.isin(mesh.faces, vertex_indices), axis=-1)
    face_mask = np.logical_and(face_mask, mask[indexable(mesh.triangles_center)])

    vertices = mesh.vertices[vertex_mask]
    faces = mesh.faces[face_mask]
    # Reset vertex numbering
    for i, val in enumerate(vertex_indices):
        faces[faces == val] = i

    return trimesh.Trimesh(vertices=vertices, faces=faces)
