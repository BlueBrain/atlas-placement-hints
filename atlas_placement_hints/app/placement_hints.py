"""Generate and save the placement hints of different regions of the AIBS mouse brain

See https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/index.html
for the specifications of the placement hints.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import click
import voxcell  # type: ignore
from atlas_commons.app_utils import (  # type: ignore
    EXISTING_FILE_PATH,
    assert_meta_properties,
    common_atlas_options,
    log_args,
    set_verbose,
    verbose_option,
)
from atlas_commons.utils import assert_metadata_content

from atlas_placement_hints.compute_placement_hints import compute_placement_hints
from atlas_placement_hints.exceptions import AtlasPlacementHintsError
from atlas_placement_hints.layered_atlas import (
    AbstractLayeredAtlas,
    MeshBasedLayeredAtlas,
    ThalamusAtlas,
    VoxelBasedLayeredAtlas,
    save_problematic_voxel_mask,
)
from atlas_placement_hints.utils import save_placement_hints

L = logging.getLogger(__name__)
METADATA_PATH = Path(Path(__file__).parent, "metadata")
METADATA_REL_PATH = METADATA_PATH.relative_to(Path(__file__).parent.parent.parent)
ALGORITHMS = ["mesh-based", "voxel-based"]


def _create_layered_atlas(
    annotation_path: str, hierarchy_path: str, metadata_path: str, algorithm: str = "mesh-based"
) -> AbstractLayeredAtlas:
    """
    Create the LayeredAtlas of the region `region_acronym`.

    Args:
        annotation_path: path to the whole mouse brain annotation nrrd file.
        hierarchy_path: path to hierarchy.json.
        metadata_path: path to the metadata json file of the brain region of interest.

    Returns:
        A layered atlas instance
    """
    annotation = voxcell.VoxelData.load_nrrd(annotation_path)
    region_map = voxcell.RegionMap.load_json(hierarchy_path)
    with open(metadata_path, "r", encoding="utf-8") as file_:
        metadata = json.load(file_)

    assert_metadata_content(metadata)

    if metadata["region"]["name"] == "Thalamus":
        return ThalamusAtlas(annotation, region_map, metadata)

    if algorithm == "voxel-based":
        return VoxelBasedLayeredAtlas(annotation, region_map, metadata)

    return MeshBasedLayeredAtlas(annotation, region_map, metadata)


def _placement_hints(  # pylint: disable=too-many-locals
    atlas: AbstractLayeredAtlas,
    direction_vectors_path: str,
    output_dir: str,
    max_thicknesses: Optional[List[float]] = None,
    flip_direction_vectors: bool = False,
    has_hemispheres: bool = False,
) -> None:
    """
    Compute the placement hints for a laminar region of the mouse brain.

    Args:
        atlas: atlas for which the placement hints are computed.
        direction_vectors_path: path to the `region_arconym` direction vectors file, e.g.,
            direction_vectors.nrrd.
        output_dir: path to the output directory.
        max_thicknesses: (optional) thicknesses of `region_acronym` layers.
            Defaults to None, i.e., there will be no validity check against desired thickness
            bounds. Otherwise layer thicknesses inferred from distance computations are checked
            against `max_thicknesses` (the latter values usually originate from experimental data).
        flip_direction_vectors: (optional) if True, the input direction vectors are negated before
            use. This is required if direction vectors flaw from the top layer (shallowest) to the
            bottom layer (deepest). Otherwise, they are left unchanged. Defaults to False.
        has_hemispheres: (optional) If True, split the volume into halves along the z-axis and
            handle each of theses 'hemispheres' separately. Otherwise the whole volume is handled.
            Defaults to True.
    """
    direction_vectors = voxcell.VoxelData.load_nrrd(direction_vectors_path)
    assert_meta_properties([direction_vectors, atlas.region])
    if direction_vectors.raw.shape[3] != 3:
        raise AtlasPlacementHintsError(
            f"Direction vectors have dimension {direction_vectors.raw.shape[3]}. Expected: 3."
        )
    distances_info, problems = compute_placement_hints(
        atlas,
        direction_vectors.raw,
        max_thicknesses,
        flip_direction_vectors=flip_direction_vectors,
        has_hemispheres=has_hemispheres,
    )
    if not Path(output_dir).exists():
        os.makedirs(output_dir)

    distance_report = {
        "before interpolation": problems["before interpolation"]["report"],
        "after interpolation": problems["after interpolation"]["report"],
    }
    with open(Path(output_dir, "distance_report.json"), mode="w+", encoding="utf-8") as file_:
        json.dump(distance_report, file_, indent=1, separators=(",", ": "))

    save_placement_hints(
        distances_info["distances_to_layer_boundaries"],
        output_dir,
        atlas.region,
        atlas.metadata["layers"]["names"],
    )
    # The problematic voxel mask is a 3D uint8 mask of the voxels for which distances
    # computation has been troublesome.
    # See atlas_placement_hints.distances.distances_to_meshes.report_distance_problems.
    save_problematic_voxel_mask(atlas, problems, output_dir)


@click.group()
def app():
    """Run the different placement hints CLI

    Besides the placement hints of the region of interest, each command generates:

    \b
    - the file `distance_report.json` reporting several distance-related problems caused by the
    region or the algorithm.
    - an nrrd file containing the 'mask' of the voxels whose placement hints are subject to one of
    the problems listed in `distance_report.json`.

    In the latter nrrd file, voxels are labeled in the following way. A voxel value of

    \b
    - 0 means there is no problem or that the voxel lies outside the region of interest,
    - 1 means that a problem was detected before interpolation and has been addressed by
    interpolation, with the placement hints of nearby voxels,
    - 2 means that a problem persisted after interpolation,
    - 3 means that a problem was created by interpolation.

    For the definition of the placement hints, see:
    https://bbpteam.epfl.ch/documentation/projects/placement-algorithm/latest/index.html#input-data
    """


@app.command()
@verbose_option
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"{str(METADATA_PATH / 'ca1_metadata.json')}"
    ),
    default=str(METADATA_PATH / "ca1_metadata.json"),
)
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the CA1 direction vectors file, e.g., `direction_vectors.nrrd`."),
)
@click.option(
    "--output-dir",
    required=True,
    help="path of the directory to write. It will be created if it doesn't exist",
)
@log_args(L)
def ca1(
    verbose, annotation_path, hierarchy_path, metadata_path, direction_vectors_path, output_dir
):
    """Generate and save the placement hints for the CA1 region of the mouse hippocampus.

    Placement hints are saved under the names specified in ca1_metadata.json.
    Default to:

    \b
    - `[PH]y.nrrd`
    - `[PH]CA1so.nrrd`, `[PH]CA1sp.nrrd`, `[PH]CA1sr.nrrd` and `[PH]CA1slm.nrrd`

    A report and an nrrd volume on problematic distance computations are generated
    in `output_dir` under the names:

    \b
    - `distance_report.json`
    - `<CA1>_problematic_voxel_mask.nrrd` (mask of the voxels for which the computed
    placement hints cannot be trusted).  <CA1> is the region name specified in
    `ca1_isocortex.json`. Defaults to "CA1".
    """
    set_verbose(L, verbose)

    atlas = _create_layered_atlas(annotation_path, hierarchy_path, metadata_path)
    _placement_hints(
        atlas,
        direction_vectors_path,
        output_dir,
        flip_direction_vectors=True,
        has_hemispheres=False,
    )


@app.command()
@verbose_option
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'isocortex_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "isocortex_metadata.json"),
)
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the isocortex direction vectors file, e.g., `direction_vectors.nrrd`."),
)
@click.option(
    "--output-dir",
    required=True,
    help="path of the directory to write. It will be created if it doesn't exist",
)
@click.option("--algorithm", type=click.Choice(ALGORITHMS))
@log_args(L)
def isocortex(
    verbose,
    annotation_path,
    hierarchy_path,
    metadata_path,
    direction_vectors_path,
    output_dir,
    algorithm,
):
    """Generate and save the placement hints of the mouse isocortex.

    This command assumes that the layer 2/3 of the isocortex has been split into
    layer 2 and layer 3.

    Placement hints are saved under the names specified in `app/data/isocortex_metadata.json`.
    Default to:

    \b
    - `[PH]y.nrrd`
    - `[PH]layer_1.nrrd`, ..., `[PH]layer_6.nrrd`

    A report together with an nrrd volume on problematic distance computations are generated
    in `output_dir` under the names:

    \b
    - `distance_report.json`
    - `<Isocortex>_problematic_voxel_mask.nrrd` (mask of the voxels for which the computed
    placement hints cannot be trusted). <Isocortex> is the region name specified in
    `isocortex_metadata.json`. Defaults to "Isocortex".
    """
    set_verbose(L, verbose)

    atlas = _create_layered_atlas(annotation_path, hierarchy_path, metadata_path, algorithm)
    _placement_hints(
        atlas,
        direction_vectors_path,
        output_dir,
        # Layer thicknesses from J. Defilipe 2017 (unpublished), see Section 5.1.1.4
        # of the release report "Neocortex Tissue Reconstruction",
        # https://github.com/BlueBrain/ncx_release_report.git
        max_thicknesses=[210.639, 190.2134, 450.6398, 242.554, 670.2, 893.62],
        has_hemispheres=True,
    )


@app.command()
@verbose_option
@common_atlas_options
@click.option(
    "--metadata-path",
    type=EXISTING_FILE_PATH,
    required=False,
    help=(
        "(Optional) Path to the metadata json file. Defaults to "
        f"`{str(METADATA_REL_PATH / 'thalamus_metadata.json')}`"
    ),
    default=str(METADATA_PATH / "thalamus_metadata.json"),
)
@click.option(
    "--direction-vectors-path",
    type=EXISTING_FILE_PATH,
    required=True,
    help=("Path to the thalamus direction vectors file, e.g., `direction_vectors.nrrd`."),
)
@click.option(
    "--output-dir",
    required=True,
    help="path of the directory to write. It will be created if it doesn't exist.",
)
@log_args(L)
def thalamus(
    verbose, annotation_path, hierarchy_path, metadata_path, direction_vectors_path, output_dir
):
    """Generate and save the placement hints of the mouse thalamus.

    Placement hints are saved under the names sepecified in `app/metadata/thalamus_metadata.json`.
    Default to:

    \b
    - `[PH]y.nrrd`
    - `[PH]Rt.nrrd`, `[PH]VPL.nrrd`

    A report together with an nrrd volume on problematic distance computations are generated
    in `output_dir` under the names:

    \b
    - `distance_report.json`
    - `<Thalamus>_problematic_voxel_mask.nrrd` (mask of the voxels for which the computed
    placement hints cannot be trusted).  <Thalamus> is the region name specified in
    thalamus_metadata.json. Defaults to "Thalamus".

    The annotation file can contain the thalamus or a superset.
    For the algorithm to work properly, some space should separate the boundary
    of the thalamus from the boundary of its enclosing array.
    """
    set_verbose(L, verbose)

    atlas = _create_layered_atlas(annotation_path, hierarchy_path, metadata_path)
    _placement_hints(
        atlas,
        direction_vectors_path,
        output_dir,
        has_hemispheres=True,
    )
