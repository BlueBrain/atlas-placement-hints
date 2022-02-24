"""
Module containing free functions for the computation of
distances to boundary meshes with respect to voxels direction vectors.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
from atlas_commons.typing import BoolArray, FloatArray, NDArray
from atlas_commons.utils import normalized, split_into_halves
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from atlas_placement_hints.distances.utils import memory_efficient_intersection
from atlas_placement_hints.exceptions import AtlasPlacementHintsError
from atlas_placement_hints.utils import is_obtuse_angle

logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
L = logging.getLogger(__name__)


def distances_to_mesh_wrt_dir(
    mesh: "trimesh.TriMesh",
    origins: FloatArray,
    directions: FloatArray,
    backward: bool = False,
) -> Tuple[FloatArray, BoolArray]:
    """
    Compute the distances from `origins` to the input mesh along `directions`.

    The computation of distances is based on ray-mesh intersections.
    The function also reports which vectors do not point in roughly the same direction
    as the normal vector of the intersected face.

    Args:
        mesh: mesh to get distances to.
        origins(array(N, 3)): origins of direction vectors
        directions(array(N, 3)): directions of vectors to compute along.
        backward: if True, distances will be checked for negated
                  direction vectors. resulting distances will be negative.
                  The check for vector direction in relation to mesh normals
                  still uses unnegated directions.
                  This option is intended to be used to check the locations
                  of deeper distances (e.g. L5, L6 for L4 voxels)

    Returns:
        float array(N, ) array holding the distance of each voxel
                          to the nearest point on `mesh` along voxel's direction vector.
        bool array(N, ) True if the ray (origin, direction) intersects with the input mesh
                         such that its angle with the mesh normal is > pi/2.
                         False otherwise.
    """
    sign = -1 if backward else 1

    # If available, embree provides a significant speedup
    ray = trimesh.ray.ray_pyembree if trimesh.ray.has_embree else trimesh.ray.ray_triangle

    intersector = ray.RayMeshIntersector(mesh)

    number_of_voxels = directions.shape[0]
    assert origins.shape[0] == number_of_voxels
    locations, ray_ids, triangle_ids = memory_efficient_intersection(
        intersector, origins, directions * sign
    )
    dist = np.full(number_of_voxels, np.nan)
    wrong_side = np.zeros(number_of_voxels, dtype=bool)

    if locations.shape[0] > 0:  # Non empty intersections
        dist[ray_ids] = sign * np.linalg.norm(locations - origins[ray_ids], axis=1)
        wrong_side[ray_ids] = is_obtuse_angle(directions[ray_ids], mesh.face_normals[triangle_ids])
        # pylint: disable=logging-unsupported-format, consider-using-f-string
        L.info(
            "Proportion of intersecting rays: {:.3%}".format(ray_ids.shape[0] / number_of_voxels)
        )
    return dist, wrong_side


def _split_indices_along_layer(
    layers_volume: NDArray[np.integer],
    layer: int,
    valid_direction_vectors_mask: BoolArray,
) -> Tuple[List[NDArray[np.integer]], List[NDArray[np.integer]]]:
    """
    Separate in two groups the voxels in `layers_volume` according to
    their position with respect to `layer`.

    The function outputs two groups under the form of numpy indices:
    the indices of the voxels lying below `layer` (`layer`included) and those of the voxels
     lying above (`layer`excluded).

    Args:
        layers_volume: volume enclosed by the union of all layers.
            Each voxel is labelled by an integer representing a layer.
            The higher is the label, the deeper is the layer.
            The value 0 represents a voxel that lies outside this volume.
        layer: the layer label identifying which layer to use for splitting.
        valid_direction_vectors_mask: 3D boolean mask for the voxels with valid direction vectors.
            Voxels whose direction vectors are invalid, i.e., of the form (NaN, NaN, NaN) are
            skipped.

    Returns:
        (below_indices, above_indices): a pair of lists. Each list has length 3. An item
            in a list is a one-dimensional numpy array holding the
            indices of the coordinate corresponding to the item index.

    """
    below_indices = np.nonzero(np.logical_and(layers_volume >= layer, valid_direction_vectors_mask))
    above_mask = np.logical_and(layers_volume < layer, layers_volume > 0)
    above_indices = np.nonzero(np.logical_and(above_mask, valid_direction_vectors_mask))

    return below_indices, above_indices  # type: ignore


# pylint: disable=too-many-arguments
def _compute_distances_to_mesh(
    directions: FloatArray,
    dists: FloatArray,
    any_obtuse_intersection: BoolArray,
    voxel_indices: List[NDArray[np.integer]],
    mesh: "trimesh.Trimesh",
    index: int,
    backward: bool = False,
    rollback_distance: int = 4,
) -> None:
    """
    Compute distances from voxels to `mesh` along direction vectors.

    Computations are based on ray-mesh intersections.
    This funcion fill the `dists` array with the outcome.

    Args:
        directions(array(N, 3)): direction vectors to compute along.
        dists: 3D distances array corresponding to the layer with label `mesh_index` + 1.
            A distances array is float 3D numpy array which holds the distance
            of every voxel in the underlying volume (wrt to its direction vector) to a fixed
            layer mesh.
        any_obtuse_intersection: mask of voxels where the intersection with
            a mesh resulted in an obtuse angle between the face and the direction vector.
        voxel_indices: list of the form [X, Y, Z], where the items are 1D numpy arrays
            of the same length. These are the indices of the voxels for which
            the computation is requested.
        mesh: mesh representing the upper boundary of the layer with index
            `index`. The mesh is usually bigger than the upper boundary alone
            and rays are assumed to hit this upper boundary only.
        index: index of the mesh or its corresponding layer.
        backward: (Optional) If True, the direction vectors are used as is to cast rays.
            Otherwise, direction vectors are negated.
        rollback_distance: (Optional) how far to step back along the directions before
            computing distances. Should be >= the max Hausdorff distance of the meshes from the
            voxelized layers it represents. This offset for the ray origins allows to obtain
            more valid intersections for voxels close to the mesh. The default value 4 was found
            by trials and errors.
    """
    if len(voxel_indices[0]) == 0:
        return

    # Adjusted ray origin: voxel position  -  an added buffer along direction
    sign = -1 if backward else 1
    origins = np.transpose(voxel_indices) + 0.5 - directions * (sign * rollback_distance)
    L.info(
        "Computing distances for the %s mesh with index %d ...",
        "lower" if backward is False else "upper",
        index,
    )
    dist, wrong = distances_to_mesh_wrt_dir(mesh, origins, directions, backward=backward)
    dist -= sign * rollback_distance
    with np.errstate(invalid="ignore"):
        dist[(dist * sign) < 0] = 0

    # Set distances
    dists[voxel_indices] = dist
    any_obtuse_intersection[voxel_indices] += wrong

    return


def distances_from_voxels_to_meshes_wrt_dir(
    layers_volume: NDArray[np.integer],
    layer_meshes: List[trimesh.Trimesh],
    directions: FloatArray,
) -> Tuple[FloatArray, BoolArray]:
    """
    For each voxel of the layers volume, compute the distance to each layer mesh along the
    the voxel direction vector.

    Args:
        layers_volume: volume enclosed by the union of all layers.
            Each voxel is labelled by an integer representing a layer.
            The higher is the label, the deeper is the layer.
            The value 0 represents a voxel that lies outside this volume.
        layer_meshes: list of meshes representing the upper boundaries of the layers.
        directions: array of shape (N, 3).
            The direction vectors of the voxels. Should be finite (not nan)
            wherever `layers_volume` > 0.

    Returns:
        Tuple (dists, any_obtuse_intersection).
        dists: 4D numpy array interpreted as a 1D array of 3D distances arrays, one for each layer.
            A distances array is a float 3D numpy array which holds the distance
            of every voxel in `layers_volume` (wrt to its direction vector) to a fixed layer mesh.
        any_obtuse_intersection: mask of voxels where the intersection with
            a mesh resulted in an obtuse angle between the face and the direction vector.
    """
    directions = normalized(directions)

    # dists is a list of 3D numpy arrays, one for each layer
    dists = np.full((len(layer_meshes),) + layers_volume.shape, np.nan)
    any_obtuse_intersection = np.zeros(layers_volume.shape, dtype=bool)

    invalid_direction_vectors_mask = np.logical_and(
        np.isnan(np.linalg.norm(directions, axis=-1)), (layers_volume > 0)
    )
    if np.any(invalid_direction_vectors_mask):
        proportion = float(np.mean(invalid_direction_vectors_mask[layers_volume > 0]))
        warnings.warn(
            f"NaN direction vectors assigned to {proportion:.5%} of the voxels."
            f" Consider interpolating invalid vectors beforehand.",
            UserWarning,
        )
    valid_mask = ~invalid_direction_vectors_mask
    L.info("Computing distances for each of the %d meshes", len(layer_meshes))
    for mesh_index, mesh in enumerate(layer_meshes):
        below_indices, above_indices = _split_indices_along_layer(
            layers_volume, mesh_index + 1, valid_mask
        )
        for part, backward in [(below_indices, False), (above_indices, True)]:
            _compute_distances_to_mesh(
                directions[part],
                dists[mesh_index],
                any_obtuse_intersection,
                part,
                mesh,
                mesh_index,
                backward=backward,
            )

    return dists, any_obtuse_intersection


def fix_disordered_distances(distances: FloatArray) -> None:
    """
    Meshes close to one another may intersect one another, leading to distances which do not match
    the layer order.
    Boundaries must be in the correct order for thicknesses to be computed.
    In these problematic cases, both distances must be set to the same value
    (an average of the two).

    The function mutates distances in-place.

    Args:
        distances:
            4D numpy array interpreted as a 1D array of 3D distances arrays, one for each layer.
            A distances array is a float 3D numpy array which holds the distance of every voxel
            (wrt to its direction vector) to a fixed layer mesh.
    """
    for layer, distance in enumerate(distances[1:], 1):
        with np.errstate(invalid="ignore"):
            previous_is_deeper = distances[layer - 1] < distance
            means = np.mean(
                [
                    distances[layer - 1][previous_is_deeper],
                    distance[previous_is_deeper],
                ],
                axis=0,
            )
            distances[layer - 1, previous_is_deeper] = means
            distances[layer, previous_is_deeper] = means


def get_thickness_excess_mask(
    distances: FloatArray, max_thicknesses: FloatArray, tolerance: float
) -> BoolArray:
    """
    Retrieve the boolean mask of the voxels which bear at least one invalid layer thickness hint

        Args:
            distances(numpy.ndarray): the distances of each voxel to each boundary,
                array of shape (number of distances, length, width, height).
            max_thicknesses: 1D float array, the maximum expected thickness for each
                layer.
            tolerance: tolerance of the error with respect to thickness.

        Returns:
            3D boolean mask of the voxels with at least one invalid thickness hint.
    """
    too_thick = np.zeros(distances[0].shape, dtype=bool)
    for i, max_thickness in enumerate(max_thicknesses):
        with np.errstate(invalid="ignore"):
            # distances[i] holds the non-negative distances wrt to direction vectors
            # from voxels to the top of layer i.
            # distances[i + 1] holds the non-positive distances wrt to direction vectors
            # from voxels to the top of layer i + 1 = bottom of layer i.
            excess = (distances[i] - distances[i + 1]) > (max_thickness + tolerance)
        too_thick = np.logical_or(too_thick, excess)

    return too_thick


def _handle_nan_distances(
    distances: FloatArray, region_mask: BoolArray, report: Dict[str, float]
) -> BoolArray:
    """
    Reports the proportions of voxels which have been assigned a NaN distance.

    Updates `report` in place and returns the mask of voxels with some NaN distance.

    Args:
        distances(numpy.ndarray): the distances of each voxel to each boundary,
            array of shape (number of distances, length, width, height).
        region_mask: mask of the region to be checked.
        report: dict containing the proportions of voxels of each problem

    Returns:
        boolean mask of the voxels with at least one NaN distance information.
    """
    do_not_intersect_bottom = np.isnan(distances[-1])
    do_not_intersect_top = np.isnan(distances[0])
    report[
        "Proportion of voxels whose rays do not intersect with the bottom surface"
        " of the deepest layer"
    ] = float(np.mean(do_not_intersect_bottom[region_mask]))
    report[
        "Proportion of voxels whose rays do not intersect with the top surface"
        " of the shallowest layer"
    ] = float(np.mean(do_not_intersect_top[region_mask]))
    nan_distances_mask = np.full(region_mask.shape, False)
    for distance in distances[1:-1]:
        nan_distances_mask = np.logical_or(np.isnan(distance), nan_distances_mask)
    report[
        "Proportion of voxels with a NaN distance with respect to at least one layer boundary"
        " distinct from the top and the bottom distances of the region"
    ] = float(np.mean(nan_distances_mask[region_mask]))

    for mask in [do_not_intersect_bottom, do_not_intersect_top]:
        nan_distances_mask = np.logical_or(nan_distances_mask, mask)

    return np.logical_and(nan_distances_mask, region_mask)


def _handle_distance_inconsistencies(
    distances: FloatArray, layered_volume: NDArray[np.integer], report: Dict[str, float]
) -> BoolArray:
    """
    Reports the proportions of voxels which have been assigned inconsistent distances.

    Updates `report` in place and returns the mask of voxels with some inconsistent distance.

    Args:
        distances(numpy.ndarray): the distances of each voxel to each boundary,
            array of shape (number of distances, length, width, height).
        layered_volume: array whose voxels are labelled by integers encoding layers
            (e.g., 1, 2, 3, 4, 5 and 6 are the integers used to label the 6 layers of the mouse
            isocortex)
        report: dict containing the proportions of voxels of each problem

    Returns:
        boolean mask of the voxels with at least one inconsistent distance information.
    """

    def _invalid_layer_order(distances, region_mask, report):
        invalid = np.any(np.diff(distances, axis=0) > 0.0, axis=0)

        report[
            "Proportion of voxels whose distances to layer boundaries are not ordered consistently"
        ] = float(np.mean(invalid[region_mask]))

        return invalid

    def _non_positive_top_layer_thickness(distances, region_mask, report):
        invalid = distances[0] <= distances[1]

        report[
            "Proportion of voxels for which the top layer has a non-positive thickness along their"
            " direction vectors"
        ] = float(np.mean(invalid[region_mask]))

        return invalid

    def _out_of_layer(distances, layers, report):
        invalid = np.zeros(layers.shape, dtype=bool)
        labels = np.unique(layers)
        labels = labels[labels != 0]
        for label in labels:
            layer_mask = layers == label
            invalid[layer_mask] = np.logical_or(
                distances[label - 1][layer_mask] < 0.0, distances[label][layer_mask] > 0.0
            )

        report[
            "Proportion of voxels whose distances to layer boundaries are inconsistent with their"
            " actual layer location"
        ] = float(np.mean(invalid[region_mask]))

        return invalid

    region_mask = layered_volume != 0

    result = _invalid_layer_order(distances, region_mask, report)
    result = np.logical_or(
        result, _non_positive_top_layer_thickness(distances, region_mask, report)
    )
    result = np.logical_or(result, _out_of_layer(distances, layered_volume, report))

    return result


def report_distance_problems(
    distances: FloatArray,
    layered_volume: NDArray[np.integer],
    obtuse_intersection: Optional[BoolArray] = None,
    max_thicknesses: Optional[FloatArray] = None,
    tolerance: float = 0.0,
) -> Tuple[Dict[str, float], BoolArray]:
    """
    Reports the proportions of voxels subject to some distance-related problem.

    These problems are:
        * the ray issued by a voxel intersects at an obtuse angle
          with the normal to the boundary mesh.
        * no ray intersection with bottom boundary.
        * no ray intersecton with top boundary.
        * some layer thickness exceeds the expected amount.

    Args:
        distances: the distances of each voxel to each boundary,
            array of shape (number of distances, length, width, height).
        layered_volume: array whose voxels are labelled by integers encoding layers
            (e.g., 1, 2, 3, 4, 5 and 6 are the integers used to label the 6 layers of the mouse
            isocortex).
        obtuse_intersection (Optional): mask of the voxels issuing a ray that intersects with some
            boundary making an obtuse angle with the boundary normal vector. Defaults to None.
        max_thicknesses: (Optional) 1D float array, the maximum expected thickness for each layer.
            Defaults to None.
        tolerance: (Optional) tolerance of the error with respect to thickness.
            Defaults to 0.0.

     Returns: tuple of the form
        (
            dict containing the proportions of voxels of each problem,
            mask of all voxels displaying at least one problem listed above
        )
    """
    report: Dict[str, float] = {}
    region_mask = layered_volume != 0
    if obtuse_intersection is not None:
        report[
            "Proportion of voxels whose rays make an obtuse angle "
            "with the mesh normal at the intersection point"
        ] = float(np.mean(obtuse_intersection[region_mask]))

    too_thick = None
    if max_thicknesses is not None:
        too_thick = get_thickness_excess_mask(distances, max_thicknesses, tolerance)
        report[
            "Proportion of voxels with a distance gap greater than the maximum thickness "
            "(NaN distances are ignored)"
        ] = float(np.mean(too_thick[region_mask]))

    problematic_volume = _handle_nan_distances(distances, region_mask, report)
    if obtuse_intersection is not None:
        problematic_volume = np.logical_or(problematic_volume, obtuse_intersection)

    problematic_volume = np.logical_or(
        problematic_volume,
        _handle_distance_inconsistencies(distances, layered_volume, report),
    )

    if too_thick is not None:
        problematic_volume = np.logical_or(problematic_volume, too_thick)

    report["Proportion of voxels with at least one distance-related problem"] = float(
        np.mean(problematic_volume[region_mask])
    )

    return report, np.logical_and(problematic_volume, region_mask)


Interpolator = Union[NearestNDInterpolator, LinearNDInterpolator]


def interpolate_scalar_field(
    field: FloatArray,
    unknown_values_mask: BoolArray,
    known_values_mask: BoolArray,
    interpolator: Interpolator = NearestNDInterpolator,
) -> None:
    """
    Interpolate `unknown_values_mask` based on `known_values_mask` using
    the `interpolator` algorithm.

    Mutates `field` in place.

    Args:
        field: float array of shape (W, H, D) where W, H and D are the integer dimensions of
            the field domain.
        unknown_values_mask: 3D boolean mask of the voxels which are not assigned a value yet.
        known_values_mask: 3D boolean mask of the voxels which are assigned a known value.
        interpolator: the scipy interpolation algorithm.
    """
    nonzero_known_values = np.nonzero(known_values_mask)
    known_positions = np.transpose(np.asarray(nonzero_known_values))
    if len(known_positions) == 0:
        raise AtlasPlacementHintsError(
            "known_values_mask is empty, no values to use for interpolation"
        )
    known_values = np.array(field)[nonzero_known_values]
    interpolated_values = interpolator(known_positions, known_values)(
        np.transpose(np.asarray(np.nonzero(unknown_values_mask)))
    )
    field[unknown_values_mask] = interpolated_values


def interpolate_problematic_distances(
    distances: FloatArray,
    problematic_mask: BoolArray,
    layered_volume: NDArray[np.integer],
    has_hemispheres: bool = True,
) -> None:
    """
    Interpolate the distances associated to voxels in `problematic_mask` with values of
    non-problematic voxels.

    Mutate `distances` in place.

    The valid values selected to interpolate to a voxel with problematic distances are the valid
    values of voxels in the same layer, and in the same hemispshere if moreover `has_hemispheres`
    is True.

    Args:
        distances: array of shape (N, W, H, D), holding the volumetric distance data
            where N stands for the number of layers augmented by 1 and (W, H, D) =
            `region_mask.shape`.
        region_mask: boolean mask of the region of interest of shape (W, H, D) where
            W, H and D are the region integer dimensions.
        problematic_mask: boolean mask of shape `region_mask.shape` holding to voxels to
            handle.
        layered_volume: integer array of shape `region_mask.shape` whose voxels are labeled
            by numbers representing layers. (The labels of six layers of the mouse isocortex
            are for instance the integers from 1 to 6).
        has_hemispheres: If True, the valid values to interpolate a problematic voxel are taken
            in the same hemisphere only.

    Returns:
        None (mutates `distances` in place).
    """
    layered_volumes: Tuple[NDArray[np.integer], ...] = (layered_volume,)
    if has_hemispheres:
        layered_volumes = split_into_halves(layered_volume)

    for layered_hemisphere in layered_volumes:
        for label in np.unique(layered_hemisphere[layered_hemisphere != 0]):
            layer_mask = layered_hemisphere == label
            for distance in distances:
                valid = np.logical_and(layer_mask, ~problematic_mask)
                invalid = np.logical_and(layer_mask, problematic_mask)
                interpolate_scalar_field(distance, invalid, valid)
