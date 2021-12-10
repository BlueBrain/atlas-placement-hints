"""
Unit tests for the layered_atlas module
"""
import json
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore
import pytest
from voxcell import VoxelData  # type: ignore

import atlas_placement_hints.layered_atlas as tested
from atlas_placement_hints.layered_atlas import (  # type: ignore
    DistanceProblem,
    MeshBasedLayeredAtlas,
    ThalamusAtlas,
    VoxelBasedLayeredAtlas,
)
from tests.placement_hints.mocking_tools import IsocortexMock, ThalamusMock

TEST_PATH = Path(Path(__file__).parent.parent)
METADATA_PATH = TEST_PATH.parent / "atlas_placement_hints" / "app" / "metadata"


def test_assert_wrong_number_of_layers():
    isocortex_mock = IsocortexMock(padding=1, layer_thickness=1, x_thickness=2, z_thickness=1)
    # Merge the first and the last layer
    raw = isocortex_mock.annotation.raw
    for id_ in isocortex_mock.layer_ids[-1]:
        raw[raw == id_] = isocortex_mock.layer_ids[0][0]

    with open(METADATA_PATH / "isocortex_metadata.json", "r", encoding="utf-8") as file_:
        metadata = json.load(file_)

    layered_atlas = tested.MeshBasedLayeredAtlas(
        isocortex_mock.annotation, isocortex_mock.region_map, metadata
    )
    # There are 5 annotated layers whereas `metadata` refers to 6 layers
    with pytest.raises(AssertionError, match=".*layer indices.* layer strings.*"):
        layered_atlas.create_layer_meshes(layered_atlas.volume)


class Test_LayeredAtlas(unittest.TestCase):
    isocortex_mock: Optional[IsocortexMock] = None
    mesh_based_layered_atlas: Optional[MeshBasedLayeredAtlas] = None
    voxel_based_layered_atlas: Optional[VoxelBasedLayeredAtlas] = None

    @classmethod
    def setUpClass(cls):
        cls.isocortex_mock = IsocortexMock(
            padding=20, layer_thickness=15, x_thickness=35, z_thickness=25
        )
        with open(METADATA_PATH / "isocortex_metadata.json", "r", encoding="utf-8") as file_:
            metadata = json.load(file_)

        cls.mesh_based_layered_atlas = tested.MeshBasedLayeredAtlas(
            cls.isocortex_mock.annotation, cls.isocortex_mock.region_map, metadata
        )

        cls.voxel_based_layered_atlas = tested.VoxelBasedLayeredAtlas(
            cls.isocortex_mock.annotation, cls.isocortex_mock.region_map, metadata
        )

    def test_region(self):
        raw = self.mesh_based_layered_atlas.region.raw
        expected = self.isocortex_mock.annotation.raw.copy()
        expected[expected == self.isocortex_mock.background] = 0
        expected[expected > 0] = 1
        npt.assert_array_equal(raw, expected.astype(bool))

    def test_create_layered_volume(self):
        volume = self.mesh_based_layered_atlas.volume
        expected = self.isocortex_mock.annotation.raw.copy()
        for i, ids in enumerate(self.isocortex_mock.layer_ids):
            expected[np.isin(expected, ids)] = 6 - i
        expected[expected == 507] = 0
        npt.assert_array_equal(volume, expected)

    def test_create_layer_meshes(self):
        volume = self.mesh_based_layered_atlas.volume
        meshes = self.mesh_based_layered_atlas.create_layer_meshes(volume)
        for i, mesh in enumerate(meshes[:-1]):
            vertices = mesh.vertices
            assert np.all(vertices[:, 1] >= 0.8 * self.isocortex_mock.padding)
            assert np.all(
                vertices[:, 1]
                <= 1.2 * self.isocortex_mock.padding + (6 - i) * self.isocortex_mock.thickness
            )

    def test_compute_distances_to_layer_boundaries(self):
        direction_vectors = np.zeros(self.isocortex_mock.annotation.raw.shape + (3,), dtype=float)
        direction_vectors[self.isocortex_mock.annotation.raw > 0] = (0.0, 1.0, 0.0)
        voxel_size = self.isocortex_mock.annotation.voxel_dimensions[0]  # assumed to be uniform

        for atlas in [self.mesh_based_layered_atlas, self.voxel_based_layered_atlas]:
            distances = atlas.compute_distances_to_layer_boundaries(direction_vectors)
            dist_info = distances["distances_to_layer_boundaries"][:-1]
            for i, ids in enumerate(self.isocortex_mock.layer_ids):
                layer_mask = np.isin(atlas.annotation.raw, ids)
                layer_index = 6 - i
                for j, dist_to_upper_boundary in enumerate(dist_info):
                    boundary_index = j + 1
                    non_nan_mask = ~np.isnan(dist_to_upper_boundary)
                    # No more than 10% of NaNs
                    npt.assert_allclose(
                        np.count_nonzero(non_nan_mask), self.isocortex_mock.volume, rtol=0.1
                    )
                    mask = layer_mask & (~np.isnan(dist_to_upper_boundary))
                    valid = np.count_nonzero(
                        dist_to_upper_boundary[mask]
                        <= voxel_size
                        * (layer_index - boundary_index + 1)
                        * self.isocortex_mock.thickness
                    )
                    # Check that distances are respected for at least 65% of the voxel of each layer
                    npt.assert_allclose(valid, np.count_nonzero(mask), rtol=0.35)
                    valid = np.count_nonzero(
                        dist_to_upper_boundary[mask]
                        >= voxel_size
                        * (layer_index - boundary_index)
                        * self.isocortex_mock.thickness
                    )
                    npt.assert_allclose(valid, np.count_nonzero(mask), rtol=0.35)

    def test_save_problematic_voxel_mask(self):
        with tempfile.TemporaryDirectory() as tempdir:
            problematic_voxel_mask = np.zeros((2, 2, 2), dtype=bool)
            problematic_voxel_mask[0, 0, 0] = True
            problematic_voxel_mask[0, 1, 0] = True
            problems = {
                "before interpolation": {"volume": problematic_voxel_mask.copy()},
                "after interpolation": {"volume": problematic_voxel_mask},
            }
            problematic_voxel_mask[0, 1, 0] = False
            expected_voxel_mask = np.full((2, 2, 2), np.uint8(DistanceProblem.NO_PROBLEM.value))
            expected_voxel_mask[0, 0, 0] = np.uint8(
                DistanceProblem.PERSISTENT_AFTER_INTERPOLATION.value
            )
            expected_voxel_mask[0, 1, 0] = np.uint8(DistanceProblem.BEFORE_INTERPOLATION.value)

            tested.save_problematic_voxel_mask(self.mesh_based_layered_atlas, problems, tempdir)
            volume_path = Path(tempdir, "Isocortex_problematic_voxel_mask.nrrd")
            voxel_data = VoxelData.load_nrrd(volume_path)
            npt.assert_almost_equal(voxel_data.raw, expected_voxel_mask)


class Test_Thalamus_Atlas(unittest.TestCase):
    thalamus_mock: Optional[ThalamusMock] = None
    thalamus_atlas: Optional[ThalamusAtlas] = None

    @classmethod
    def setUpClass(cls):
        cls.thalamus_mock = ThalamusMock(padding=20, shape=(50, 50, 50), layer_thickness_ratio=0.2)
        with open(METADATA_PATH / "thalamus_metadata.json", "r", encoding="utf-8") as file_:
            metadata = json.load(file_)
        cls.thalamus_atlas = tested.ThalamusAtlas(
            cls.thalamus_mock.annotation, cls.thalamus_mock.region_map, metadata
        )

    def test_create_layered_volume(self):
        expected = self.thalamus_mock.annotation.raw.copy()
        expected[expected == 549] = 2
        expected[expected == 262] = 1
        npt.assert_array_equal(self.thalamus_atlas.volume, expected)

    def test_create_layer_meshes(self):
        meshes = self.thalamus_atlas.create_layer_meshes(self.thalamus_atlas.volume)

        # Check that the x-coordinates of the layer meshes vertices are reasonnable
        vertices = meshes[0].vertices
        assert np.all(vertices[:, 0] >= 0.8 * self.thalamus_mock.padding)
        assert np.all(
            vertices[:, 0]
            <= 1.2 * self.thalamus_mock.padding
            + self.thalamus_mock.layer_thickness_ratio * self.thalamus_mock.shape[0]
        )

        vertices = meshes[1].vertices
        assert np.all(
            vertices[:, 0]
            >= 0.8
            * (
                self.thalamus_mock.padding
                + self.thalamus_mock.layer_thickness_ratio * self.thalamus_mock.shape[0]
            )
        )
        assert np.all(
            vertices[:, 0] <= 1.2 * (self.thalamus_mock.padding + self.thalamus_mock.shape[0])
        )
