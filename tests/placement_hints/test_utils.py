"""
Unit tests of placement hints utils.
"""
from pathlib import Path

import numpy as np
import numpy.testing as npt
import trimesh

import atlas_placement_hints.utils as tested
from tests.mocking_tools import MockxelData


def test_is_obtuse_angle():
    vector_field_1 = np.array(
        [
            [
                [[1.0, 0.0, -1.0], [10.3, 5.6, 9.0]],
                [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
            ],
            [
                [[1.0, 2.0, -1.0], [-5.1, 1.0, 1.0]],
                [[1.0, 2.0, 3.0], [0.0, -1.0, -1.0]],
            ],
        ]
    )
    vector_field_2 = np.array(
        [
            [
                [[-1.0, 1.0, -1.0], [2.0, -2.0, 1.0]],
                [[-0.3, 0.1, -1.9], [1.0, -1.0, -1.0]],
            ],
            [
                [[-1.0, 2.0, -1.0], [5.1, 0.0, 26.0]],
                [[3.0, 2.0, -3.0], [-6.0, -1.0, -1.0]],
            ],
        ]
    )

    expected = [
        [[False, False], [False, False]],
        [[False, True], [True, False]],
    ]
    npt.assert_array_equal(tested.is_obtuse_angle(vector_field_1, vector_field_2), expected)


def test_centroid_outfacing_mesh():
    # The centroid lies outside of the mesh
    vertices = [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0.8]]
    faces = [[0, 1, 3], [0, 3, 2], [3, 1, 2], [0, 2, 4], [0, 4, 1], [1, 4, 2]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    expected_vertices = [[0, 0, 1], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
    expected_faces = [[2, 3, 0], [2, 0, 1], [0, 3, 1]]
    result = tested.centroid_outfacing_mesh(mesh)
    npt.assert_array_equal(result.vertices, expected_vertices)
    npt.assert_array_equal(result.faces, expected_faces)

    # The centroid lies inside of the mesh
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 0], [0, 3, 1]], dtype=float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    result = tested.centroid_outfacing_mesh(mesh)
    assert (
        len(set(tuple(v) for v in result.vertices).difference(set(tuple(v) for v in vertices))) == 0
    )
    assert len(result.faces) == len(faces)


def test_save_placement_hints():
    voxel_size = 10  # um
    saved_files_dict = {}
    voxel_data_for_saving = MockxelData(saved_files_dict, np.array([[[0]]]), (voxel_size,) * 3)

    tested.save_placement_hints(
        np.array([[[[2, 3, 4, 5]]], [[[1, 1, 1, 1]]], [[[-2, -1, 0, 1]]]]),
        "output_directory",
        voxel_data_for_saving,
        layer_names=["astring", 1],
    )

    expected_dict = {
        str(Path("output_directory", "[PH]y.nrrd")): -np.array([[[-2, -1, 0, 1]]]),
        str(Path("output_directory", "[PH]astring.nrrd")): np.array(
            [[[[3, 4], [2, 4], [1, 4], [0, 4]]]]
        ),
        str(Path("output_directory", "[PH]1.nrrd")): np.array([[[[0, 3], [0, 2], [0, 1], [0, 0]]]]),
    }

    for filename, value in expected_dict.items():
        npt.assert_array_equal(saved_files_dict[filename], value)


class Test_detailed_mesh_mask:
    def test_get_space_occupied_by_triangles(self):
        expected_mesh_mask = np.array(
            [[[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]], dtype=bool
        )

        vertices = np.array([[0, 0.5, 0.5], [0, 2.5, 0.5], [0, 2.5, 2.5]])
        faces = [[0, 1, 2]]
        test_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        npt.assert_array_equal(
            tested.detailed_mesh_mask(test_mesh, expected_mesh_mask.shape), expected_mesh_mask
        )
