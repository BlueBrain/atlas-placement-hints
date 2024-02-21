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
from pathlib import Path
from shutil import which
from typing import Union

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
    # Create a two-line header file for ultraVolume2Mesh containing only the type and shape.
    header = filename + ".hdr"
    with open(header, "w", encoding="utf-8") as file_:
        file_.write("uchar\n")
        file_.write(" ".join([str(d) for d in img_array.shape]))


def _get_ultra_volume_2_mesh_path() -> str:
    """
    Get the path to the ultraVolume2Mesh executable file.

    The function checks if ultraVolume2Mesh is available by means of
    shutil.which. If yes, it returns this path as a string.
    Otherwise it raises an AtlasPlacementHintsError.

    Returns:
        ultra_volume_2_mesh_path: path to ultraVolume2Mesh.

    Raises:
        AtlasPlacementHintsError: if the executable file cannot be found.

    """
    ultra_volume_2_mesh_path = which("ultraVolume2Mesh")
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
    volume_path: str,
    output_directory: str,
    optimization_iterations: int,
) -> None:
    """
    Calls Ultraliser/ultraVolume2Mesh with the option --optimize-mesh and some user-defined options.

    The executable ultraVolume2Mesh creates the boundary mesh of a 3D voxellized image
    using the Dual Marching Cubes algorithm.
    See https://bbpcode.epfl.ch/browse/code/viz/Ultraliser/tree/apps/ultraVolume2Mesh.

    Args:
        volume_path: value of the --volume-path option (path to an .hdr file).
        output_directory: value of the --output-directory option (path to a directory).
        optimization_iteration: value of the --optimization-iterations option.
    Raises:
        UlraliserException if the executable ultraVolume2mesh cannot be found.
    """
    # Retrieve Ultraliser/ultraVolume2mesh path
    ultra_volume_2_mesh_path = _get_ultra_volume_2_mesh_path()
    with subprocess.Popen(
        [
            ultra_volume_2_mesh_path,
            "--volume",
            volume_path,
            "--iso-option",
            "value",
            "--iso-value",
            "1",
            "--export-obj-mesh",
            "--optimize-mesh",
            "--optimization-iterations",
            str(optimization_iterations),
            "--output-directory",
            output_directory,
        ],
        text=True,
    ) as proc:
        proc.communicate()


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
    binary_image: BoolArray, optimization_info: bool = False, mesh_name=None
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
    tempdir: Union[Path, str] = ""
    if mesh_name is not None:
        mesh_name = mesh_name.replace(" ", "_")
        tempdir = Path("meshes") / f"mesh_{mesh_name}"
        tempdir.mkdir(parents=True, exist_ok=True)
    else:
        tempdir_ctx = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        tempdir = tempdir_ctx.name

    # The format of the following filepaths is imposed by Ultraliser.
    volume_path = str(Path(tempdir, "binary_image"))
    _volume_path = Path(volume_path)
    output_filepath_opt = (
        _volume_path.parent / "meshes" / (str(_volume_path.stem) + "-watertight.obj")
    )
    output_filepath_unopt = _volume_path.parent / "meshes" / (str(_volume_path.stem) + "-dmc.obj")
    if not output_filepath_opt.exists():
        # ultraVolume2Mesh requires a name without file extension.
        # Write image to disk for later use by ultraliser.
        _write_numpy_array_to_img_file(binary_image, volume_path)
        ultra_volume_2_mesh(
            volume_path=volume_path + ".hdr",
            output_directory=str(tempdir),
            optimization_iterations=5,
        )

    for filepath in [output_filepath_unopt, output_filepath_opt]:
        if not Path(filepath).exists():
            raise AtlasPlacementHintsError(f"Ultraliser failed to generate the mesh {filepath}")

    optimized_mesh = trimesh.load_mesh(output_filepath_opt)

    if mesh_name is None:
        tempdir_ctx.cleanup()

    if optimization_info:
        unoptimized_mesh = trimesh.load_mesh(output_filepath_unopt)
        log_mesh_optimization_info(optimized_mesh, unoptimized_mesh)

    optimized_mesh.fix_normals()
    return optimized_mesh
