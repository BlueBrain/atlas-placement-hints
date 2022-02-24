"""Generic sript for the computation of voxel-to-layer distances wrt
to direction vectors, a.k.a placement hints, in a layered atlas.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from atlas_commons.typing import BoolArray, FloatArray

from atlas_placement_hints.distances.distances_to_meshes import (
    interpolate_problematic_distances,
    report_distance_problems,
)

if TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=ungrouped-imports
    from atlas_placement_hints.placement_hints.layered_atlas import AbstractLayeredAtlas


DistanceInfo = Dict[str, Union["AbstractLayeredAtlas", FloatArray, BoolArray]]
DistancesReport = Dict[str, float]


# pylint: disable=too-many-arguments, too-many-locals
def compute_placement_hints(
    atlas: "AbstractLayeredAtlas",
    direction_vectors: FloatArray,
    max_thicknesses: Optional[List[float]] = None,
    flip_direction_vectors: bool = False,
    has_hemispheres: bool = True,
) -> Tuple[DistanceInfo, Dict]:
    """
    Compute the placement hints for a laminar region of the mouse brain.

    Args:
        atlas: layered atlas for which the placement hints are computed.
        direction_vectors: unit vector field of shape (W, H, D, 3) if `annotation`'s array
            is of shape (W, H, D).
        max_thicknesses: (optional) thicknesses of `region_acronym` layers.
            Defaults to None, i.e., there will be no validity check with input from literature.
        flip_direction_vectors: If True, the input direction vectors are negated before use.
            This is required if direction vectors flaw from the top layer (shallowest) to the
            bottom layer (deepest). Otherwise, they are left unchanged. Defaults to false.
        has_hemispheres: (optional) If True, split the volume into halves along the z-axis and
            handle each of theses 'hemispheres' separately. Otherwise the whole volume is
            handled. Defaults to True.

    Returns:
        Tuple with the following items.
        distances_info: dict with the following entries.
            obtuse_angles: 3D binary mask indicating which voxels have rays
                intersecting a layer boundary with an obtuse angle. The direction vectors
                of such voxels are considered as problematic.
            distances_to_layer_meshes(numpy.ndarray): 4D float array of shape
                (number of layers + 1, W, H, D) holding the distances from
                voxel centers to the upper boundaries of layers wrt to voxel direction vectors.
        problems: dict with two keys, "before interpolation" and "after interpolation".
            The amount of distance problems are expected to decrease after interpolation of
            invalid distance information (NaN or thickness excess) by valid information of
            neighbouring voxels.
            The corresponding values are dict with keys "report" and "volume".
            report:
                the value associated to "report" is dict reporting the proportion of voxels
                subject to each distance-related problem,
                see distances.distance_to_meshes.report_distance_problems doc.
            volume: the value associated to "volume" is a 3D boolean mask of the voxels with at
                least one distance-related problem.
                See distances.distance_to_meshes.report_distance_problems doc.
    """
    distances_info = atlas.compute_distances_to_layer_boundaries(
        direction_vectors,
        flip_direction_vectors=flip_direction_vectors,
        has_hemispheres=has_hemispheres,
    )

    distances_to_meshes = distances_info["distances_to_layer_boundaries"]
    tolerance = 2.0 * atlas.region.voxel_dimensions[0]
    distances_report, problematic_mask = report_distance_problems(
        distances_to_meshes,
        atlas.volume,
        distances_info.get("obtuse_angles", None),
        max_thicknesses=max_thicknesses,
        tolerance=tolerance,
    )
    interpolate_problematic_distances(
        distances_to_meshes,
        problematic_mask,
        atlas.volume,
        has_hemispheres=has_hemispheres,
    )
    (interpolated_distances_report, filtered_problematic_mask,) = report_distance_problems(
        distances_to_meshes,
        atlas.volume,
        max_thicknesses=max_thicknesses,
        tolerance=tolerance,
    )
    problems = {
        "before interpolation": {
            "report": distances_report,
            "volume": problematic_mask,
        },
        "after interpolation": {
            "report": interpolated_distances_report,
            "volume": filtered_problematic_mask,
        },
    }

    return distances_info, problems
