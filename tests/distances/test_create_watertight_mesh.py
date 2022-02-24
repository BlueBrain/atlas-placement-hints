"""
Unit tests of create_watertight_mesh
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

import atlas_placement_hints.distances.create_watertight_mesh as tested
from atlas_placement_hints.exceptions import AtlasPlacementHintsError


class Test_create_watertight_mesh(unittest.TestCase):
    ultra_volume_2_mesh_path = (
        "/gpfs/bbp.cscs.ch/apps/hpc/jenkins/deploy/applications/2018-12-19/"
        "linux-rhel7-x86_64/gcc-6.4.0/ultraliser-0.1.0-v4fncgcrft/bin"
        "/ultraVolume2Mesh"
    )

    def test_mean_min_dist(self):
        sample_size = 2500
        points1 = np.random.rand(sample_size, 3)
        points2 = points1 + 0.01 * np.random.rand(sample_size, 3)
        d = tested.mean_min_dist(points1, points2)
        assert d <= 0.01

    @patch("atlas_placement_hints.distances.create_watertight_mesh.L.info")
    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.mean_min_dist",
        return_value=0.015,
    )
    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.directed_hausdorff",
        return_value=[0.010],
    )
    def test_log_mesh_optimization_info(self, directed_hausdorff_mock, mean_min_dist_mock, L_mock):
        optimized_trimesh = trimesh.creation.annulus(1, 2, 1)
        unoptimized_trimesh = trimesh.creation.annulus(1, 2, 1)
        tested.log_mesh_optimization_info(optimized_trimesh, unoptimized_trimesh)
        assert mean_min_dist_mock.called
        assert directed_hausdorff_mock.called
        assert L_mock.call_args[0][1] == 0.015  # Check last argument of the last logger call

    def test_write_numpy_array_to_img_file(self):
        img_array = np.zeros((3, 4, 5))
        # Check if the .img and .hdr files have been created
        with tempfile.TemporaryDirectory() as tempdir:
            prev_dir = os.getcwd()
            os.chdir(tempdir)
            tested._write_numpy_array_to_img_file(img_array, "mask")
            assert Path("mask.img").exists()
            assert Path("mask.hdr").exists()
            with open("mask.hdr", "r", encoding="utf-8") as mask_header:
                content = mask_header.read()
                assert content == "3 4 5"
            os.chdir(prev_dir)

    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.find_executable",
        return_value="",
    )
    def test_get_ultra_volume_2_mesh_path_no_install(self, _):
        # Should raise if the ultraVolume2Mesh binary cannot be found.
        with pytest.raises(AtlasPlacementHintsError):
            assert tested._get_ultra_volume_2_mesh_path()

    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.find_executable",
        return_value=ultra_volume_2_mesh_path,
    )
    def test_get_ultra_volume_2_mesh_path_loaded(self, _):
        # Should succeed if ultraliser module is loaded
        assert (
            tested._get_ultra_volume_2_mesh_path()
            == Test_create_watertight_mesh.ultra_volume_2_mesh_path
        )

    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.ultra_volume_2_mesh",
        return_value="Ok",
    )
    def test_create_watertight_mesh_files_not_found(self, _):
        # Should raise if Ultraliser fails to generate meshes, i.e., .obj files
        with pytest.raises(AtlasPlacementHintsError):
            assert tested.create_watertight_trimesh(np.array([]))

    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.ultra_volume_2_mesh",
        return_value="Ok",
    )
    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.trimesh.load_mesh",
        return_value=trimesh.creation.box(),
    )
    @patch("atlas_placement_hints.distances.create_watertight_mesh.log_mesh_optimization_info")
    @patch(
        "atlas_placement_hints.distances.create_watertight_mesh.Path.exists",
        return_value=True,
    )
    # Should display mesh optimization info if requested
    def test_create_watertight_mesh_log_option(
        self,
        _,
        log_mesh_optimization_info_mock,
        trimesh_load_mock,
        ultra_volume_2_mesh_mock,
    ):
        mesh = tested.create_watertight_trimesh(np.array([]), optimization_info=True)
        assert log_mesh_optimization_info_mock.called
        assert isinstance(mesh, trimesh.base.Trimesh)
