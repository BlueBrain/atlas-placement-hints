"""Module for the computation of
voxel-to-layer distances wrt to direction vectors in a laminar brain region.

This module is used for the computation of placement hints in the mouse
isocortex and in the mouse Hippocampus CA1 region.
"""

from __future__ import annotations

import logging
import os
import sys
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np
import trimesh  # type: ignore
from atlas_commons.typing import BoolArray, FloatArray, NDArray
from atlas_commons.utils import create_layered_volume, query_region_mask, split_into_halves
from cached_property import cached_property  # type: ignore
from cgal_pybind import estimate_thicknesses
from tqdm import tqdm  # type: ignore
from voxcell import RegionMap, VoxelData  # type: ignore

from atlas_placement_hints.distances.create_watertight_mesh import create_watertight_trimesh
from atlas_placement_hints.distances.distances_to_meshes import (
    distances_from_voxels_to_meshes_wrt_dir,
    fix_disordered_distances,
)
from atlas_placement_hints.utils import (
    centroid_outfacing_mesh,
    clip_mesh,
    detailed_mesh_mask,
    get_convex_hull_boundary,
)

# from typing import TYPE_CHECKING, Dict, List, Union
# if TYPE_CHECKING:  # pragma: no cover
#     import trimesh  # type: ignore
#     from voxcell import RegionMap, VoxelData  # type: ignore


logging.basicConfig(level=logging.INFO)
L = logging.getLogger(__name__)

# Constants
LEFT = 0  # left hemisphere
RIGHT = 1  # right hemisphere


class DistanceProblem(IntEnum):
    """
    Enumerate distance-related problems detected after the computation of placement hints.
    """

    NO_PROBLEM = 0
    # Problems are, e.g., NaN distance value or excessive layer thickness, see
    # distances/distances_to_meshes/report_distance_problems for a complete list of problems
    BEFORE_INTERPOLATION = 1  # the problem is fixed after interpolation
    PERSISTENT_AFTER_INTERPOLATION = 2
    NEW_AFTER_INTERPOLATION = 3


class AbstractLayeredAtlas(ABC):
    """
    Abstract class holding the data of a layered atlas, i. e., an atlas with well-defined layers
    for which boundary meshes can be created.
    """

    def __init__(self, annotation: "VoxelData", region_map: "RegionMap", metadata: dict):
        """
        annotation: annotated volume enclosing the whole brain atlas.
        region_map: Object to navigate the brain regions hierarchy.
        metadata: dict of the form
            {
                "region": {
                    "name": "Isocortex",
                    "query": "Isocortex",
                    "attribute": "acronym",
                    "with_descendants": true
                },
                "layers": {
                    "names":
                        ["layer 1", "layer 2", "layer 3", "layer 4", "layer 5", "layer 6"],
                    "queries":
                        ["@.*1[ab]?$", "@.*2[ab]?$", "@.*3[ab]?$", "@.*4[ab]?$", "@.*5[ab]?$",
                            "@.*6[ab]?$"],
                    "attribute": "acronym",
                    "with_descendants": true
                }
            }
            Find more examples in app/data/metadata.
        """
        self.annotation = annotation
        self.region_map = region_map
        self.metadata = metadata

    @cached_property
    def region(self) -> "VoxelData":
        """
        Accessor of the layered atlas as a VoxelData object.

        Returns:
            VoxelData instance of the layered atlas.
        """
        region_mask = query_region_mask(
            self.metadata["region"], self.annotation.raw, self.region_map
        )
        return self.annotation.with_data(region_mask)

    @cached_property
    def volume(self) -> NDArray[np.integer]:
        """
        Get the volume enclosed by the specified layers.

        Returns:
            layers_volume: numpy 3D array whose voxels are labelled by the indices of
                `self.metadata["layers"]["names"]` augmented by 1.
        """

        number_of_layers = len(self.metadata["layers"]["names"])
        L.info(
            "Creating a volume for each of the %d layers of %s ...",
            number_of_layers,
            self.metadata["region"]["name"],
        )

        return create_layered_volume(self.annotation.raw, self.region_map, self.metadata)

    @abstractmethod
    def compute_distances_to_layer_boundaries(
        self,
        direction_vectors: FloatArray,
        has_hemispheres: bool = True,
        flip_direction_vectors: bool = False,
    ) -> Dict[str, Union[FloatArray, BoolArray]]:
        """
        Compute distances from voxels to layers boundariea wrt to direction vectors.
        """


class MeshBasedLayeredAtlas(AbstractLayeredAtlas):
    """
    Class holding the data of a layered atlas, i. e., an atlas with well-defined layers
    for which boundary meshes can be created. Layer boundaries are approximated by surface
    meshes when computing distances to boundaries wrt to direction vectors.

    Appropriate for the Isocortex, CA1 and Thalamus regions.
    """

    def __init__(
        self,
        annotation: "VoxelData",
        region_map: "RegionMap",
        metadata: dict,
    ):
        """
        annotation: annotated volume enclosing the whole brain atlas.
        region_map: Object to navigate the brain regions hierarchy.
        metadata: dict, see json examples in app/data/metadata
        """

        AbstractLayeredAtlas.__init__(self, annotation, region_map, metadata)

    def create_layer_meshes(self, layered_volume: NDArray[np.integer]) -> List["trimesh.Trimesh"]:
        """
        Create meshes representing the upper boundary of each layer
        in the laminar region volume, referred to as `layered_volume`.

        Args:
            layered_volume: numpy 3D array whose voxels are labelled by the indices of
            `self.metadata["layers"]["names"]` augmented by 1.
        Returns:
            meshes: list of the layers meshes, together with the mesh of the complement of the
                whole region. Each mesh is used to define the upper boundary of the
                corresponding layer. Meshes from the first to the last layer have decreasing sizes:
                the first mesh encloses all layers, the second mesh encloses all layers but the
                first one, the second mesh encloses all layers but the first two, and so on so
                forth. The last mesh represents the bottom of the last layer. It has the vertices
                of the first mesh, but its normal are inverted.
        """
        layers_values = np.unique(layered_volume)
        layers_values = layers_values[layers_values > 0]
        layer_str_count = len(self.metadata["layers"]["names"])
        assert len(layers_values) == len(
            self.metadata["layers"]["names"]
        ), f"{len(layers_values)} layer indices, {layer_str_count} layer strings"
        L.info(
            "Creating a watertight mesh for each of the %d layers of %s ...",
            len(layers_values),
            self.metadata["region"]["name"],
        )
        meshes = []
        for index in tqdm(layers_values):
            mesh = create_watertight_trimesh(layered_volume >= index)
            meshes.append(mesh)

        L.info(
            "Trimming inward faces of the %d meshes of %s ...",
            len(meshes),
            self.metadata["region"]["name"],
        )
        full_mesh_bottom = meshes[0].copy()
        # Inverting normals as we select the complement of the layered atlas
        full_mesh_bottom.invert()
        meshes.append(full_mesh_bottom)
        for i, mesh in tqdm(enumerate(meshes), total=len(meshes)):
            newmesh = centroid_outfacing_mesh(mesh)
            # This sometimes results in isolated faces which
            # cause ray intersection to fail.
            # So we trim them off by taking only the largest submesh.
            submeshes = newmesh.split(only_watertight=False)
            if len(submeshes) > 0:
                big_mesh = np.argmax([len(submesh.vertices) for submesh in submeshes])
                meshes[i] = submeshes[big_mesh]
            else:
                meshes[i] = mesh

        return meshes

    # pylint: disable=W0613
    def _compute_dists_and_obtuse_angles(
        self, volume, direction_vectors, hemisphere=LEFT, thalamus_meshes_dir: str = ""
    ):
        layer_meshes = self.create_layer_meshes(volume)
        # pylint: disable=fixme
        # TODO: compute max_smooth_error and use it as the value of rollback_distance
        # in the call of distances_from_voxels_to_meshes_wrt_dir()
        return distances_from_voxels_to_meshes_wrt_dir(volume, layer_meshes, direction_vectors)

    def _dists_and_obtuse_angles(
        self, direction_vectors, has_hemispheres=False, thalamus_meshes_dir: str = ""
    ):
        if not has_hemispheres:
            return self._compute_dists_and_obtuse_angles(
                self.volume, direction_vectors, thalamus_meshes_dir=thalamus_meshes_dir
            )
        # Processing each hemisphere individually
        hemisphere_distances = []
        hemisphere_volumes = split_into_halves(self.volume)
        hemisphere_obtuse_angles = []
        L.info(
            "Computing distances from voxels to layers meshes ...",
        )
        for hemisphere in [LEFT, RIGHT]:
            L.info(
                "Computing distances for the hemisphere %d of the %s region ...",
                hemisphere,
                self.metadata["region"]["name"],
            )
            dists_to_layer_meshes, obtuse = self._compute_dists_and_obtuse_angles(
                hemisphere_volumes[hemisphere],
                direction_vectors,
                thalamus_meshes_dir=thalamus_meshes_dir,
                hemisphere=hemisphere,
            )
            hemisphere_distances.append(dists_to_layer_meshes)
            hemisphere_obtuse_angles.append(obtuse)
        obtuse_angles = np.logical_or(
            hemisphere_obtuse_angles[LEFT], hemisphere_obtuse_angles[RIGHT]
        )
        # Merging the distances arrays of the two hemispheres
        distances_to_layer_meshes = hemisphere_distances[LEFT]
        right_hemisphere_mask = hemisphere_volumes[RIGHT] > 0
        distances_to_layer_meshes[:, right_hemisphere_mask] = hemisphere_distances[RIGHT][
            :, right_hemisphere_mask
        ]
        return distances_to_layer_meshes, obtuse_angles

    def compute_distances_to_layer_boundaries(
        self,
        direction_vectors: FloatArray,
        has_hemispheres: bool = True,
        flip_direction_vectors: bool = False,
        thalamus_meshes_dir: str = "",
    ) -> Dict[str, Union[FloatArray, BoolArray]]:
        """
        Compute distances from voxels to layers boundaries wrt to direction vectors.
        Boundaries are represented by 3D surface meshes.

        Compute also the volume of voxels with problematic direction, i.e.,
        voxels for which no reliable distance information can be obtained.

        Args:
            direction_vectors: unit vector field of shape (W, H, D, 3)
                if `annotation.raw`is of shape (W, H, D).
            has_hemispheres: True if the brain region of interest
                should be split in two hemispheres, False otherwise.
            flip_direction_vectors: True if the direction vectors should
                be reverted, False otherwise. This flag needs to be set to True
                depending on the algorithm used to generated orientation.nrrd.
            thalamus_meshes_dir: (optional) Path of the directory to load thalamus meshes
                from. Currently only used for thalamus. Required if you are producing thalamus
                placement-hints. Defaults to None.

        Returns:
            distances_info: dict with the following entries.
                obtuse_angles: 3D boolean array indicating which voxels have rays
                    intersecting a layer boundary with an obtuse angle. The direction vectors
                    of such voxels are considered as problematic.
                distances_to_layer_meshes(numpy.ndarray): 4D float array of shape
                    (number of layers + 1, W, H, D) holding the distances from
                    voxel centers to the upper boundaries of layers wrt to voxel direction vectors.
        """
        if flip_direction_vectors:
            direction_vectors = -direction_vectors

        distances_to_layer_meshes, obtuse_angles = self._dists_and_obtuse_angles(
            direction_vectors, has_hemispheres, thalamus_meshes_dir=thalamus_meshes_dir
        )
        L.info("Fixing disordered distances ...")
        fix_disordered_distances(distances_to_layer_meshes)

        return {
            "distances_to_layer_boundaries": self.annotation.voxel_dimensions[0]
            * distances_to_layer_meshes,
            "obtuse_angles": obtuse_angles,
        }


class ThalamusAtlas(MeshBasedLayeredAtlas):
    """
    Class holding the data of a two-layer atlas for the mouse thalamus.

    The second layer of the thalamus, that is, the complement of the reticular
    nucleus, cannot be defined via a simple regular expression because the
    thalamus (id = 549, non-leaf) has voxels with labels 549 in both AIBS CCFv2
    and CCFv3 mouse brain models.

    <original comment ends here> (AES, <2023-06-28 Wed>: I'm not sure the above
    comment applies anymore due to our recent, in-progress creation of
    leaf-only annotations, but nonetheless the thalamus is a special case for
    making its placement-hints.)

    If you're reading this, you're probably making new thalamus meshes because
    the annotation has changed or some other reason. Generating placement-hints
    for the thalamus has been changed, and now requires a manual step. You
    should do the following steps:

    1. Pass the argument '--thalamus-meshes-dir /your/folder/here' to the
    top-level CLI command 'atlas-placement-hints thalamus'. This will create
    the meshes, but NOT the placement-hints, and the program will exit.

    2. MANUALLY cut the reticular meshes into the 'top' and 'bottom' halves
    (aka the inner and outer halves if looking outwards from the center of the
    thalamus) using software like 'Blender'. Do this for each hemisphere. The
    RT region 'top' and 'bottom' halves are curvy and complex enough that it
    was decided that manual cutting was the most effective / efficient way to
    get a good mesh layer for the thalamus placement-hints, since the previous
    computational way included too many holes due to the curviness. To do the
    cutting in Blender, you can follow these instructions:

        A. Import each reticular mesh ('File > Import > Stl (.stl)').

        B. Click the dropdown menu in the upper left corner that says 'Object
        Mode' and change it to 'Edit Mode'.

        C. In the central window where the meshes are displayed, (you may have
        to zoom out) select either the 'inner' (bottom) or 'outer' (top) half
        of that hemisphere's reticular mesh.

        D. Click 'Mesh > Separate > Selection' (or press hotkey P then click
        Selection).

        E. In the 'Scene Collection' window to the top right, which lists all
        the meshes, you should now see a new entry that is named similarly to
        'reticular_nucleus_mesh_right_hemisphere_original.001', depending on
        your input file. Click on its entry in the 'Scene Collection' window to
        select it.

        F. Select 'File > Export > Stl (.stl)' to export this new mesh into its
        own file, making sure to click the box that says 'Selection Only' in
        the export prompt. Name the file appropriate to the hemisphere and
        top/bottom selection that you've just done. If you're unsure which mesh
        you have selected, you can click the 'Eye' symbols in the 'Scene
        Collection' window to toggle which meshes are shown in the main view.

        G. Since you have 'separated' your mesh into two, the original mesh
        object now consists of the remainder of the mesh which you did not
        select in the previous steps. In other words, if you previously
        selected, 'separated', and exported the 'bottom' of
        'reticular_nucleus_mesh_right_hemisphere_original', the entry in 'Scene
        Collection' for the mesh
        'reticular_nucleus_mesh_right_hemisphere_original' now only consists of
        the 'top' part of the mesh. Click the entry for it in 'Scene
        Collection', then 'File > Export' like above, making sure to select
        'Selection Only' and name it appropriately.

        H. Repeat the process for the other hemisphere.

    Note that this 'separation' is a different operation than 'splitting'.
    There are many tutorials on Youtube for how to do this. In the outgoing
    filename, change 'original' to 'handcut'. In the end, you should end up
    with four new files:
    'reticular_nucleus_mesh_left_hemisphere_bottom_handcut.stl',
    'reticular_nucleus_mesh_left_hemisphere_top_handcut.stl',
    'reticular_nucleus_mesh_right_hemisphere_bottom_handcut.stl', and
    'reticular_nucleus_mesh_right_hemisphere_top_handcut.stl'. After you have
    made the 4 new files, I recommend you view them individually (using
    software like Paraview) to double check that you named them correctly, etc.

    3. Re-run the top-level command 'atlas-placement-hints thalamus' but this
    time with both the argument '--thalamus-meshes-dir /your/folder/here' and
    the flag '--load-cut-thalamus-meshes'. This will NOT create the meshes, but
    WILL create the placement-hints using your newly handcut meshes! Note that
    this must be done using a compute node with at least as much RAM as that of
    'memory_bound' in
    'atlas-placement-hints/atlas_placement_hints/distances/
    utils.py:memory_efficient_intersection()',
    otherwise this WILL fail silently! With 'memory_bound' set to 300, if your
    meshes are not very optimized (e.g. using ultraliser's default 1
    '--optimization-iterations'), and have a similar number of faces to RT mask
    surface voxels, this should take approximately 1.5 hours for the whole
    thing. If your meshes are more optimized, this can significantly speed up
    the process.

    """

    def create_uncut_thalamus_meshes(self, thalamus_meshes_dir: str):
        """
        Create meshes representing the upper boundary of each layer of the thalamus atlas.

        Because the lower boundary of the thalamus is too irregular to obtain
        meaningful ray-mesh interesections, we consider instead its convex
        hull, which provides us with a smooth approximation. See pictures and
        discussion of https://bbpteam.epfl.ch/project/issues/browse/NSETM-1433

        <original comment ends here> AES <2023-06-28 Wed>: This now creates
        meshes that are expected to be manually cut using Blender/etc., as
        described in the 'ThalamusAtlas' class docstring. Also, note that this
        creates the 'upper boundary of each layer', but ALSO creates the
        bottom-most layer as well (see the docstring for
        'ThalamusAtlas.load_layer_meshes()' for details).
        """

        L.info(
            """Currently set to CREATE meshes, but NOT calculate thalamus
            placement-hints. See
            'atlas-placement-hints/atlas_placement_hints/layered_atlas.py:ThalamusAtlas'
            for details."""
        )
        hemisphere_volumes = split_into_halves(self.volume)
        for hemisphere in [LEFT, RIGHT]:
            if hemisphere == LEFT:
                hemisphere_string = "left"
            elif hemisphere == RIGHT:
                hemisphere_string = "right"
            L.info(
                "Creating uncut meshes for the %s hemisphere (hemisphere %d) of the %s region ...",
                hemisphere_string,
                hemisphere,
                self.metadata["region"]["name"],
            )

            # Because the lower boundary of the thalamus is too irregular to obtain meaningful
            # ray-mesh interesections, we consider instead its convex hull, which provides us with a
            # smooth approximation. See pictures and discussion of
            # https://bbpteam.epfl.ch/project/issues/browse/NSETM-1433
            thalamus_convex_hull_boundary = get_convex_hull_boundary(hemisphere_volumes[hemisphere])
            hull_mask = detailed_mesh_mask(
                thalamus_convex_hull_boundary, hemisphere_volumes[hemisphere].shape
            )
            reticular_nucleus_mesh = create_watertight_trimesh(hemisphere_volumes[hemisphere] == 1)
            reticular_nucleus_mesh_top = clip_mesh(reticular_nucleus_mesh, hull_mask)
            reticular_nucleus_mesh_bottom = clip_mesh(
                reticular_nucleus_mesh, hull_mask, remainder=True
            )
            reticular_nucleus_mesh_bottom.invert()
            overall_bottom = clip_mesh(
                thalamus_convex_hull_boundary,
                ~detailed_mesh_mask(reticular_nucleus_mesh, hemisphere_volumes[hemisphere].shape),
            )
            overall_bottom.fix_normals()
            overall_bottom.invert()

            thalamus_convex_hull_boundary.export(
                os.path.join(
                    thalamus_meshes_dir,
                    f"thalamus_convex_hull_boundary_{hemisphere_string}_hemisphere_original.stl",
                )
            )
            reticular_nucleus_mesh.export(
                os.path.join(
                    thalamus_meshes_dir,
                    f"reticular_nucleus_mesh_{hemisphere_string}_hemisphere_original.stl",
                )
            )
            reticular_nucleus_mesh_bottom.export(
                os.path.join(
                    thalamus_meshes_dir,
                    f"reticular_nucleus_mesh_{hemisphere_string}_hemisphere_bottom_original.stl",
                )
            )
            reticular_nucleus_mesh_top.export(
                os.path.join(
                    thalamus_meshes_dir,
                    f"reticular_nucleus_mesh_{hemisphere_string}_hemisphere_top_original.stl",
                )
            )
            overall_bottom.export(
                os.path.join(
                    thalamus_meshes_dir,
                    f"overall_bottom_{hemisphere_string}_hemisphere_original.stl",
                )
            )

        L.info("Finished creating and saving meshes for both hemispheres. Exiting.")
        sys.exit()

    # pylint: disable=W0613
    def load_layer_meshes(
        self, layered_volume: NDArray[np.integer], thalamus_meshes_dir: str, hemisphere=LEFT
    ) -> List["trimesh.Trimesh"]:
        """
        Load thalamus meshes meshes for bottom-most layer and upper boundary of other layers.

        Long-winded explanation by AES:

        Inside 'layered_volume', voxels marked 0 belong to 'outside' the
        relevant volume, and can be thought of as an unused 'Layer 0' that
        corresponds to the 'uppermost' space that is not considered a 'real'
        layer (like the 'pia' in cortex).

        Inside 'layered_volume', voxels marked 1 belong to the reticular
        nucleus (RT), and can be thought of as 'Layer 1' or the 'uppermost'
        layer we actually care about, similar to cortex L1. In this function,
        we seek to load a hand-cut 'upper boundary mesh' for this layer,
        corresponding to the outermost half (loosely speaking) of a mesh of the
        RT. This upper boundary mesh for Layer 1 (RT) is the first object in
        the list that is returned by the function.

        Inside 'layered_volume', voxels marked 2 belong to non-RT thalamus, and
        can be thought of as 'Layer 2' or the next-deeper layer, similar to
        cortex L2. In this function, we seek to load a hand-cut 'upper boundary
        mesh' for this layer, corresponding to the innermost half (loosely
        speaking) of a mesh of the RT (since we're assuming there's no gap
        between RT and nonRT, in general). This upper boundary mesh for Layer 2
        (nonRT) is the second object in the list that is returned by this
        function.

        The final object returned by this function is the 'lower boundary mesh'
        for the entire volume that we care about, aka the 'lowest boundary
        mesh'. (Using the previous upper boundaries and this single lower
        boundary is enough information to compute the distance from anywhere to
        the nearest upper and lower boundaries of each layer.) Confusingly, and
        unlike cortex, our lowest boundary is essentially the same as the
        boundary of Layer 0: the convex hull mesh of the entire thalamus. This
        is because all the direction vectors in nonRT (our Layer 2) point to RT
        (our Layer 1), and the origins of those direction vectors are
        distributed along almost a full half-sphere if looking from RT, at much
        wider angles than a more laminar structure like the cortex.

        Currently, since the lowest boundary mesh is actually the entire convex
        hull, that means there are some portions of the mesh where the 'bottom'
        layer boundary actually sits 'above' the uppermost boundary of the
        topmost layer! However, this shouldn't cause any problems.
        """
        if thalamus_meshes_dir == "":
            L.info(
                """\n --> ERROR: You must pass a directory containing the
                appropriate meshes to '--thalamus-meshes-dir'. See
                'atlas-placement-hints/atlas_placement_hints/layered_atlas.py:ThalamusAtlas'
                for details. Exiting.\n"""
            )
            sys.exit()

        if hemisphere == LEFT:
            hemisphere_string = "left"
        elif hemisphere == RIGHT:
            hemisphere_string = "right"

        L.info(
            """Loading hand-sliced thalamic reticular nucleus top and bottom
            meshes of %s hemisphere from '%s'""",
            hemisphere_string,
            thalamus_meshes_dir,
        )

        reticular_nucleus_mesh_top = trimesh.load_mesh(
            os.path.join(
                thalamus_meshes_dir,
                f"reticular_nucleus_mesh_{hemisphere_string}_hemisphere_top_handcut.stl",
            )
        )

        reticular_nucleus_mesh_bottom = trimesh.load_mesh(
            os.path.join(
                thalamus_meshes_dir,
                f"reticular_nucleus_mesh_{hemisphere_string}_hemisphere_bottom_handcut.stl",
            )
        )
        # We have to invert the normals of the bottom mesh faces after hand-cutting, since by
        # default they face "outward", not "inward" like we want (so they can align with the
        # direction vectors)
        reticular_nucleus_mesh_bottom.invert()  # type: ignore  # type: ignore

        overall_bottom = trimesh.load_mesh(
            os.path.join(
                thalamus_meshes_dir,
                f"overall_bottom_{hemisphere_string}_hemisphere_original.stl",
            )
        )

        return [
            reticular_nucleus_mesh_top,  # type: ignore  # type: ignore
            reticular_nucleus_mesh_bottom,  # type: ignore  # type: ignore
            overall_bottom,  # type: ignore  # type: ignore
        ]

    def _compute_dists_and_obtuse_angles(
        self, volume, direction_vectors, hemisphere=LEFT, thalamus_meshes_dir: str = ""
    ):
        # Note that this is LOADING meshes, not creating them!
        L.info(
            """Currently set to LOAD meshes and calculate thalamus
            placement-hints. See
            'atlas-placement-hints/atlas_placement_hints/layered_atlas.py:ThalamusAtlas'
            for details."""
        )
        layer_meshes = self.load_layer_meshes(volume, thalamus_meshes_dir, hemisphere)
        return distances_from_voxels_to_meshes_wrt_dir(volume, layer_meshes, direction_vectors)


class VoxelBasedLayeredAtlas(AbstractLayeredAtlas):
    """
    Class holding the data of a layered atlas, i. e., an atlas with well-defined layers.
    Distances to layer boundaries wrt to direction vectors are estimated using the identifiers
    of the voxels hit by the ray issued from a given voxel.

    Appropriate for the isocortex, CA1 and Thalamus regions.
    """

    def __init__(
        self,
        annotation: "VoxelData",
        region_map: "RegionMap",
        metadata: dict,
    ):
        """
        annotation: annotated volume enclosing the whole brain atlas.
        region_map: Object to navigate the brain regions hierarchy.
        metadata: dict, see json examples in app/data/metadata
        """

        AbstractLayeredAtlas.__init__(self, annotation, region_map, metadata)

    def compute_distances_to_layer_boundaries(
        self,
        direction_vectors: FloatArray,
        has_hemispheres: bool = True,
        flip_direction_vectors: bool = False,
    ) -> Dict[str, Union[FloatArray, BoolArray]]:
        """
        Compute distances from voxels to layers boundaries wrt to direction vectors.

        Args:
            direction_vectors: unit vector field of shape (W, H, D, 3)
                if `annotation.raw`is of shape (W, H, D).

        Returns: a 4D float array of shape (number of layers + 1, W, H, D) holding the distances
            from voxel centers to the upper boundaries of layers wrt to voxel direction vectors.
        """
        layer_count = len(self.metadata["layers"]["names"])
        volume = np.asarray(self.volume, dtype=np.uint8)
        L.info("Estimating layer thicknesses of %d layers based on ray traversals ...", layer_count)
        thicknesses = estimate_thicknesses(
            volume,
            np.asarray(direction_vectors, dtype=np.float32),
            np.asarray(self.annotation.offset, dtype=np.float32),
            np.asarray(self.annotation.voxel_dimensions, dtype=np.float32),
            layer_count,
            resolution=0.5,
        )

        L.info("Deriving distance estimates from layer thicknesses ...")
        assert thicknesses.shape == self.annotation.shape + (layer_count + 1,), (
            f"Expected shape: {self.annotation.shape + (layer_count + 1,)} distance arrays, "
            f"got {thicknesses.shape}"
        )

        # The array `thicknesses[..., 0]` holds the PHy values: non-negative distances from the
        # bottom of the region to each voxel. Layer `thicknesses[..., 1:]` are arranged from the
        # deepest layer to the shallowest, e.g., 6 to 1, following the flow of direction vectors.
        # We build a distances array in the same sense as in `distances.distance_to_meshes`:
        # a float 3D numpy array of shape (number_of_layers, region_shape) with
        # region_shape = `self.annotation.shape`, which holds the signed distance of every
        # voxel in the underlying volume (wrt to its direction vector) to a fixed layer boundary.
        # Signed distances are taken along the "curved y-axis", i.e. along the flow of unit
        # direction vectors.

        # Exclude PHy, reverse layer ordering before accumulating thicknesses
        distances = np.cumsum(thicknesses[..., 1:][..., ::-1], axis=-1)[..., ::-1]
        # Shape change: (region_shape, number_of_layers) -> (number_of_layers, region_shape)
        distances = np.moveaxis(distances, -1, 0)
        # Negative signed distance from each voxel to the region bottom (-PHy)
        thicknesses[..., 0] = -thicknesses[..., 0]
        # Signed distances to the top of layer i for every i in [0, number_of_layers - 1]
        for i in range(len(distances)):
            distances[i, ...] += thicknesses[..., 0]
        # Add distances to the bottom of the deepest layer, thought of as the
        # "top of the outside layer".
        distances = np.vstack([distances, thicknesses[np.newaxis, :, :, :, 0]])

        assert (
            len(distances) == layer_count + 1
        ), f"Expected {layer_count + 1} distance arrays, got {len(distances)} arrays."

        return {"distances_to_layer_boundaries": distances}


def save_problematic_voxel_mask(
    layered_atlas: AbstractLayeredAtlas, problems: dict, output_dir: str
):
    """
    Save the problematic voxel mask to file.

    The problematic volume is an array of the same shape as `layered_atlas.region`,
    i.e., (W, H, D). Its dtype is uint8.
    A voxel value equal to DistanceProblem.NO_PROBLEM indicates that no problem was detected.
    The value DistanceProblem.BEFORE_INTERPOLATION indicates that a problem was detected before
    interpolation of problematic distances by valid ones but not after.
    The value DistanceProblem.AFTER_INTERPOLATION indicates that a problem persists after
    interpolation.

    Args:
        layered_atlas: atlas for which distances computations have generated a problematic voxel
            mask.
        problems: dict returned by
            atlas_placement_hints.compute_placement_hints.compute_placement_hints.
            This dictionary contains in particular 3D binary masks corresponding to voxels
            with problematic placement hints.
        output_dir: directory in which to save the problematic volume as an nrrd file.
    """
    problematic_volume_path = os.path.join(
        output_dir, layered_atlas.metadata["region"]["name"] + "_problematic_voxel_mask.nrrd"
    )
    L.info(
        "Saving problematic volume of %s to file %s ...",
        layered_atlas.metadata["region"]["name"],
        problematic_volume_path,
    )
    before_voxel_mask = problems["before interpolation"]["volume"]
    problematic_volume = np.full(
        before_voxel_mask.shape, np.uint8(DistanceProblem.NO_PROBLEM.value)
    )
    problematic_volume[before_voxel_mask] = np.uint8(DistanceProblem.BEFORE_INTERPOLATION.value)
    after_voxel_mask = problems["after interpolation"]["volume"]
    problematic_volume[np.logical_and(after_voxel_mask, before_voxel_mask)] = np.uint8(
        DistanceProblem.PERSISTENT_AFTER_INTERPOLATION.value
    )
    problematic_volume[np.logical_and(after_voxel_mask, ~before_voxel_mask)] = np.uint8(
        DistanceProblem.NEW_AFTER_INTERPOLATION.value
    )

    layered_atlas.annotation.with_data(problematic_volume).save_nrrd(problematic_volume_path)
