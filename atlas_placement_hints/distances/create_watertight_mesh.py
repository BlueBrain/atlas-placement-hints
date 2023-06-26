"""
Main function to create a trimesh out of a voxellized 3D image
and its utilities.
Generating optimized meshes requires Ultraliser,
see https://bbpcode.epfl.ch/browse/code/viz/Ultraliser
"""
import logging
import os
import subprocess
import tempfile
from distutils.spawn import find_executable
from pathlib import Path

import nrrd  # type: ignore
import numpy as np
import trimesh  # type: ignore
from atlas_commons.typing import BoolArray, FloatArray
from scipy.spatial.distance import cdist, directed_hausdorff  # type: ignore

from atlas_placement_hints.exceptions import AtlasPlacementHintsError

L = logging.getLogger(__name__)
L.setLevel(logging.INFO)


def _write_numpy_array_to_img_file(img_array: BoolArray, filename: str) -> None:
    """
    Write the content of a 3D image provided as a numpy array
    to a file with extension .img

    Note: The output extension .img is required by ultraVolume2Mesh. The input name
        file for this function required to have no extension.

    Args:
        img_array(numpy.ndarray): numpy array holding the image content (voxels)
        filename(str): extension-free name of the file to write to.
    """
    # UltraVolume2Mesh expects a file name with .img extension,
    # see https://bbpcode.epfl.ch/browse/code/viz/Ultraliser/tree/ultraliser/common/Defines.h
    nrrd.write(
        filename + ".nrrd",
        np.uint8(img_array),
        header={"encoding": "raw"},
        detached_header=True,  # ultraVolume2Mesh cannot read embedded headers.
    )
    os.rename(filename + ".nrrd", filename + ".img")
    os.remove(filename + ".nhdr")
    # Create a two-line header file for ultraVolume2Mesh containing only the type and shape of the 3D array.
    header = filename + ".hdr"
    with open(header, "w", encoding="utf-8") as file_:
        file_.write("uchar\n")
        file_.write(" ".join([str(d) for d in img_array.shape]))


def _get_ultra_volume_2_mesh_path() -> str:
    """
    Get the path to the ultraVolume2Mesh executable file.

    The function checks if ultraVolume2Mesh is available by means of
    distutils.spawn.find_executable(). If yes, it returns this path as a string.
    Otherwise it raises an AtlasPlacementHintsError.

    Returns:
        ultra_volume_2_mesh_path: path to ultraVolume2Mesh.

    Raises:
        AtlasPlacementHintsError: if the executable file cannot be found.

    """
    ultra_volume_2_mesh_path = find_executable("ultraVolume2Mesh")
    if not ultra_volume_2_mesh_path:
        raise AtlasPlacementHintsError(
            "ultraVolume2Mesh was not found in this system.\n"
            "On BB5, you can load ultraliser with the command 'module load ultraliser'.\n"
            "You can also install ultraliser "
            "(see https://bbpcode.epfl.ch/browse/code/viz/Ultraliser/tree/ultraliser)"
            " and modify your PATH accordingly."
        )
    return ultra_volume_2_mesh_path


def ultra_volume_2_mesh(
    volume_path_stem: str,
    output_directory: str,
    smooth_iterations: int,
    iso_value_min: int = 1,
    iso_value_max: int = 256,
) -> None:
    """
    Calls Ultraliser/ultraVolume2Mesh with the option --optimize-mesh and some user-defined options.

    The executable ultraVolume2Mesh creates the boundary mesh of a 3D voxellized image
    using the Dual Marching Cubes algorithm.
    See https://bbpcode.epfl.ch/browse/code/viz/Ultraliser/tree/apps/ultraVolume2Mesh.

    Args:
        volume_path_stem: value of the --volume option, except for its .hdr extension
        joutput_directory: value of the --output-directory option (path to a directory).
        smooth_iterations: value of the --smooth-iterations option.
        iso_value_min: value of the minimum iso-value; defaults to 1.
        iso_value_max: value of the maximum iso-value; defaults to 256.
    Raises:
        UlraliserException if the executable ultraVolume2mesh cannot be found.
    """
    # Retrieve Ultraliser/ultraVolume2mesh path
    ultra_volume_2_mesh_path = _get_ultra_volume_2_mesh_path()

    # Arguments changed here to get working with Ultraliser 0.4.0 as of 2023-05-04, see
    # https://bbpteam.epfl.ch/project/issues/browse/VIZTM-1251
    # including that it now expects an .HDR file extension for the input file!
    subprocess.check_output(
        [
            ultra_volume_2_mesh_path,
            "--volume",
            str(volume_path_stem + ".hdr"),
            "--output-directory",
            output_directory,
            "--iso-option range",
            "--min-value",
            str(iso_value_min),
            "--max-value",
            str(iso_value_max),
            "--smooth-iterations",
            str(smooth_iterations),
            "--export-obj-mesh",
            "--project-xy",
            "--project-xz",
            "--project-zy",
            "--optimize-mesh",
        ]
    )


def mean_min_dist(points_1: FloatArray, points_2: FloatArray, sample_size: int = 1000) -> float:
    """
    Compute the mean of the minimum distance of
    3D points in a random sample of `points_1` to all points in `points_2`.

    Args:
        points_1: A 1D array of 3D points, i.e, an array of shape (N, 3).
        points_2: A 1D array of 3D points, i.e, an array of shape (N, 3).
        sample_size: size of the sample of pt1s for which distances are computed.

    Returns:
        mean of the distances of the points in the sample to `points_2`.
    """
    sampled_indices = np.random.choice(len(points_1), size=sample_size, replace=False)
    distances: FloatArray = np.min(cdist(points_1[sampled_indices], points_2), axis=1)

    return float(np.mean(distances))


def log_mesh_optimization_info(
    optimized_mesh: trimesh.base.Trimesh, unoptimized_mesh: trimesh.base.Trimesh
):
    """
    Log information about the optimized and unoptimized meshes generated by ultraVolume2Mesh.

    This function logs
      - the mean mininimum distance, see mean_min_dist()
      - the Hausdorff distance (https://en.wikipedia.org/wiki/Hausdorff_distance)
    The triangle centers are concatenated with the regular vertices for these two computations.

    Args:
        optimized_mesh: triangle mesh that has been optimized by ultraVolume2Mesh.
        unoptimized_mesh: triangle mesh obtained without optimization.
    """
    meshes = {"Optimized": optimized_mesh, "Unoptimized": unoptimized_mesh}
    for key, mesh in meshes.items():
        L.info("%s mesh: %d vertices, %d faces", key, len(mesh.vertices), len(mesh.faces))
    args = [
        np.concatenate((mesh.triangles_center, mesh.vertices), axis=0) for mesh in meshes.values()
    ]
    hausdorff = directed_hausdorff(args[0], args[1])[0]
    meandist = mean_min_dist(args[0], args[1])
    L.info("Hausdorff distance of optimization: %f", hausdorff)
    L.info("Mean distance of optimization: %f", meandist)


def create_watertight_trimesh(
    binary_image: BoolArray,
    optimization_info: bool = False,
) -> trimesh.base.Trimesh:
    """
    Create a watertight triangle mesh out of a 3D binary image.

    Relies on Ultraliser/ultraVolume2Mesh.

    Args:
        binary_image: 3D image to be processed for the creation of its boundary mesh.
        optimization_info: if True, compute and display optimization info.
            Otherwise no optimization info is computed.
    Returns:
        optimized_mesh: the optimized triangle mesh produced by ultraVolume2Mesh
         (Dual Marching Cubes algorithm).
    """
    optimized_mesh = None  # The mesh to be returned.
    unoptimized_mesh = None
    with tempfile.TemporaryDirectory() as tempdir:
        volume_path_stem = str(Path(tempdir, "binary_image"))
        # Write image to disk for later use by ultraliser.
        _write_numpy_array_to_img_file(binary_image, volume_path_stem)
        # ultraVolume2Mesh writes the resulting meshes to four output files in the following order:
        # 1. <volume path stem>-dmc.obj
        # 2. <volume path stem>-laplacian.obj
        # 3. <volume path stem>-optimized.obj
        # 4. <volume path stem>-watertight.obj
        iso_value_min = 1
        iso_value_max = 256
        ultra_volume_2_mesh(
            volume_path_stem=volume_path_stem,
            output_directory=tempdir,
            smooth_iterations=15,
            iso_value_min=iso_value_min,
            iso_value_max=iso_value_max,
        )
        # The format of the following filepaths is imposed by Ultraliser.
        output_mesh_stem = str(Path(tempdir, "meshes/", "binary_image"))
        output_filepath_opt = output_mesh_stem + "-" + "optimized.obj"
        output_filepath_unopt = output_mesh_stem + "-" + "laplacian.obj"
        output_filepath_wtrt = output_mesh_stem + "-" + "watertight.obj"
        for filepath in [output_filepath_opt, output_filepath_unopt, output_filepath_wtrt]:
            if not Path(filepath).exists():
                raise AtlasPlacementHintsError(f"Ultraliser failed to generate the mesh {filepath}")
        unoptimized_mesh = trimesh.load_mesh(output_filepath_unopt)
        optimized_mesh = trimesh.load_mesh(output_filepath_opt)
        watertight_mesh = trimesh.load_mesh(output_filepath_wtrt)

    if optimization_info:
        log_mesh_optimization_info(optimized_mesh, unoptimized_mesh)

    watertight_mesh.fix_normals()
    return watertight_mesh
