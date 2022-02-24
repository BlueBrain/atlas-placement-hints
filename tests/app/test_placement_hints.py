"""test app/placement_hints"""
import json

import numpy as np
import numpy.testing as npt
from click.testing import CliRunner
from voxcell import VoxelData  # type: ignore

import atlas_placement_hints.app.placement_hints as tested
from tests.placement_hints.mocking_tools import Ca1Mock, IsocortexMock, ThalamusMock


def test_thalamus():
    runner = CliRunner()
    with runner.isolated_filesystem():
        thalamus_mock = ThalamusMock(padding=10, shape=(60, 50, 60), layer_thickness_ratio=0.15)
        direction_vectors = np.zeros(thalamus_mock.annotation.raw.shape + (3,), dtype=float)
        direction_vectors[thalamus_mock.annotation.raw > 0] = (-1.0, 0.0, 0.0)

        thalamus_mock.annotation.save_nrrd("annotation.nrrd")
        thalamus_mock.annotation.with_data(direction_vectors).save_nrrd("direction_vectors.nrrd")
        with open("hierarchy.json", "w", encoding="utf-8") as file_:
            json.dump(thalamus_mock.region_map_dict, file_)

        result = runner.invoke(
            tested.thalamus,
            [
                "--annotation-path",
                "annotation.nrrd",
                "--hierarchy-path",
                "hierarchy.json",
                "--direction-vectors-path",
                "direction_vectors.nrrd",
                "--output-dir",
                "placement_hints",
            ],
        )
        assert result.exit_code == 0, str(result.output)

        # The values selected below as upper bounds are surprisingly large, which can be explained
        # as follows. Due to the shape and the size of the simplified brain region under test,
        # voxels close to the boundary of the volume are problematic (rays issued from them
        # all miss the top surface meshes which have a too low resolution). This problem is
        # aggravated by the splitting into two hemispheres. Increasing the dimensions of the tested
        # volume reduces the number of problematic voxels but makes the test much longer.
        # See for a picture of the problematic voxel volume generated by this test.
        #
        # The testing issues reported here are very similar to those encountered with CA1 and
        # isocortex below. Renderings of the created surface meshes and of problematic volumes
        # indicate that the algorithms are working as expected.

        with open("placement_hints/distance_report.json", encoding="utf-8") as file_:
            report = json.load(file_)
            distances_report = report["before interpolation"]
            assert (
                distances_report[
                    "Proportion of voxels whose rays make an obtuse"
                    " angle with the mesh normal at the intersection point"
                ]
                <= 0.15
            )
            del distances_report[
                "Proportion of voxels whose rays make an obtuse"
                " angle with the mesh normal at the intersection point"
            ]

            assert (
                distances_report[
                    "Proportion of voxels with a NaN distance with respect to at least one layer"
                    " boundary distinct from the top and the bottom distances of the region"
                ]
                < 0.49
            )
            del distances_report[
                "Proportion of voxels with a NaN distance with respect to at least one layer"
                " boundary distinct from the top and the bottom distances of the region"
            ]

            assert (
                distances_report["Proportion of voxels with at least one distance-related problem"]
                < 0.5
            )
            del distances_report["Proportion of voxels with at least one distance-related problem"]

            for proportion in distances_report.values():
                assert proportion <= 0.06

        problematic_volume = VoxelData.load_nrrd(
            "placement_hints/Thalamus_problematic_voxel_mask.nrrd"
        )
        # Problems reported before interpolation of faulty values
        assert (
            np.count_nonzero(np.isin(problematic_volume.raw, [1, 2])) / thalamus_mock.volume < 0.5
        )

        # Problems which have persisted after interpolation and problems caused by interpolation
        assert (
            np.count_nonzero(np.isin(problematic_volume.raw, [2, 3])) / thalamus_mock.volume < 0.12
        )

        # Interpolation has caused no new problems
        assert np.count_nonzero(problematic_volume.raw == 3) / thalamus_mock.volume == 0.0

        ph_y = VoxelData.load_nrrd("placement_hints/[PH]y.nrrd")
        npt.assert_array_equal(ph_y.raw.shape, thalamus_mock.annotation.raw.shape)

        ph_th_no_rt = VoxelData.load_nrrd("placement_hints/[PH]THnotRT.nrrd")
        npt.assert_array_equal(ph_th_no_rt.raw.shape, thalamus_mock.annotation.raw.shape + (2,))


def test_ca1():
    runner = CliRunner()
    with runner.isolated_filesystem():
        ca1_mock = Ca1Mock(padding=10, layer_thickness=30, x_thickness=35, z_thickness=30)
        direction_vectors = np.zeros(ca1_mock.annotation.raw.shape + (3,), dtype=float)
        direction_vectors[ca1_mock.annotation.raw > 0] = (0.0, -1.0, 0.0)
        ca1_mock.annotation.save_nrrd("annotation.nrrd")
        ca1_mock.annotation.with_data(direction_vectors).save_nrrd("direction_vectors.nrrd")
        with open("hierarchy.json", "w", encoding="utf-8") as file_:
            json.dump(ca1_mock.region_map_dict, file_)

        result = runner.invoke(
            tested.ca1,
            [
                "--annotation-path",
                "annotation.nrrd",
                "--hierarchy-path",
                "hierarchy.json",
                "--direction-vectors-path",
                "direction_vectors.nrrd",
                "--output-dir",
                "placement_hints",
            ],
        )
        assert result.exit_code == 0, str(result.output)

        with open("placement_hints/distance_report.json", encoding="utf-8") as file_:
            report = json.load(file_)
            report_before = report["before interpolation"]
            # Because of the simplified ca1 model used for the tests,
            # bad voxels due to obtuse intersection angle are numerous.
            # Indeed, most of the voxels lying in the boundary with the ca1 complement
            # will issue rays that won't intersect with the target boundary meshes in the expected way.
            obtuse_angles = "Proportion of voxels whose rays make an obtuse angle with the mesh normal at the intersection point"
            assert report_before[obtuse_angles] <= 0.15

            at_least_one_problem = "Proportion of voxels with at least one distance-related problem"
            assert report_before[at_least_one_problem] <= 0.2

            for (description, proportion) in report_before.items():
                if description not in [obtuse_angles, at_least_one_problem]:
                    assert proportion <= 0.06

            # Inconsitencies report
            assert (
                report_before[
                    "Proportion of voxels whose distances to layer boundaries are not ordered consistently"
                ]
                <= 0.06
            )

            assert (
                report_before[
                    "Proportion of voxels for which the top layer has a non-positive thickness along their"
                    " direction vectors"
                ]
                <= 0.035
            )

            assert (
                report_before[
                    "Proportion of voxels whose distances to layer boundaries are inconsistent with their"
                    " actual layer location"
                ]
                <= 0.004
            )

            report_after = report["after interpolation"]
            assert not obtuse_angles in report_after
            for description in report_after:
                assert report_after[description] == 0.0

            problematic_volume = VoxelData.load_nrrd(
                "placement_hints/CA1_problematic_voxel_mask.nrrd"
            )
            assert (
                np.count_nonzero(np.isin(problematic_volume, [1, 2])) / ca1_mock.volume <= 0.07
            )  # before
            assert (
                np.count_nonzero(np.isin(problematic_volume, [2, 3])) / ca1_mock.volume <= 0.05
            )  # after (persistent or new)
            assert (
                np.count_nonzero(problematic_volume == 3) / ca1_mock.volume == 0.0
            )  # no new problems


def _write_isocortex_input_to_file(isocortex_mock):
    direction_vectors = np.zeros(isocortex_mock.annotation.raw.shape + (3,), dtype=float)
    direction_vectors[isocortex_mock.annotation.raw > 0] = (0.0, 1.0, 0.0)
    isocortex_mock.annotation.save_nrrd("annotation.nrrd")
    isocortex_mock.annotation.with_data(direction_vectors).save_nrrd("direction_vectors.nrrd")

    with open("hierarchy.json", "w", encoding="utf-8") as file_:
        json.dump(isocortex_mock.region_map_dict, file_)


def _get_isocortex_result(runner, algorithm="mesh-based"):
    options = [
        "--annotation-path",
        "annotation.nrrd",
        "--hierarchy-path",
        "hierarchy.json",
        "--direction-vectors-path",
        "direction_vectors.nrrd",
        "--output-dir",
        "placement_hints",
        "--algorithm",
        algorithm,
    ]

    return runner.invoke(tested.isocortex, options)


def test_isocortex():
    runner = CliRunner()
    with runner.isolated_filesystem():
        isocortex_mock = IsocortexMock(
            padding=10, layer_thickness=15, x_thickness=35, z_thickness=25
        )
        _write_isocortex_input_to_file(isocortex_mock)
        for algorithm in ["mesh-based"]:
            result = _get_isocortex_result(runner, algorithm)
            assert result.exit_code == 0, str(result.output)

            with open("placement_hints/distance_report.json", encoding="utf-8") as file_:
                report = json.load(file_)
                report_before = report["before interpolation"]
                # Because of the simplified isocortex model used for the tests,
                # bad voxels due to obtuse intersection angle are numerous.
                obtuse_angles = "Proportion of voxels whose rays make an obtuse angle with the mesh normal at the intersection point"
                if obtuse_angles in report_before:
                    assert report_before[obtuse_angles] <= 0.15

                at_least_one_problem = (
                    "Proportion of voxels with at least one distance-related problem"
                )
                assert report_before[at_least_one_problem] <= 0.2

                for (description, proportion) in report_before.items():
                    if description not in [obtuse_angles, at_least_one_problem]:
                        assert proportion <= 0.075

                # Inconsitencies report
                assert (
                    report_before[
                        "Proportion of voxels whose distances to layer boundaries are not ordered consistently"
                    ]
                    <= 0.08
                )

                assert (
                    report_before[
                        "Proportion of voxels for which the top layer has a non-positive thickness along their"
                        " direction vectors"
                    ]
                    <= 0.05
                )

                assert (
                    report_before[
                        "Proportion of voxels whose distances to layer boundaries are inconsistent with their"
                        " actual layer location"
                    ]
                    <= 0.004
                )

                report_after = report["after interpolation"]
                assert obtuse_angles not in report_after
                for description in report_after:
                    assert report_after[description] == 0.0

                problematic_volume = VoxelData.load_nrrd(
                    "placement_hints/Isocortex_problematic_voxel_mask.nrrd"
                )
                assert (
                    np.count_nonzero(np.isin(problematic_volume, [1, 2])) / isocortex_mock.volume
                    <= 0.18
                )  # before
                assert (
                    np.count_nonzero(np.isin(problematic_volume, [2, 3])) / isocortex_mock.volume
                    <= 0.14
                )  # after (persistent and new)
                assert (
                    np.count_nonzero(problematic_volume == 3) / isocortex_mock.volume == 0.0
                )  # no new problems


def test_exception_on_incorrect_input():
    runner = CliRunner()
    with runner.isolated_filesystem():
        isocortex_mock = IsocortexMock(padding=1, layer_thickness=1, x_thickness=5, z_thickness=5)
        # Direction vectors mistaken with orientation (field of 4D vectors)
        direction_vectors = np.zeros(isocortex_mock.annotation.raw.shape + (4,), dtype=float)
        isocortex_mock.annotation.save_nrrd("annotation.nrrd")
        isocortex_mock.annotation.with_data(direction_vectors).save_nrrd("direction_vectors.nrrd")
        with open("hierarchy.json", "w", encoding="utf-8") as file_:
            json.dump(isocortex_mock.region_map_dict, file_)

        result = runner.invoke(
            tested.isocortex,
            [
                "--annotation-path",
                "annotation.nrrd",
                "--hierarchy-path",
                "hierarchy.json",
                "--direction-vectors-path",
                "direction_vectors.nrrd",
                "--output-dir",
                "placement_hints",
            ],
        )
        assert result.exit_code == 1, str(result.output)
        assert "Direction vectors have dimension 4. Expected: 3." in str(result.exception)
