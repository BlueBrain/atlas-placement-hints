"""
 Mocking tools for the unit tests related to placement hints.
"""
from typing import Tuple

import numpy as np
from voxcell import RegionMap, VoxelData


class IsocortexMock:
    """
    Class to instantiate an annotated volume with some of the
    mouse isocortex features.

    The constructor allows to specify the dimensions of the volume and
    a uniform layer thickness along the y-axis.
    """

    def __init__(
        self,
        padding: int,
        layer_thickness: int,
        x_thickness: int,
        z_thickness: int,
        background: int = 507,
    ):
        self.thickness = layer_thickness  # layer thickness along the y-axis.
        self.padding = padding  # Regiodesics will crash with a padding <= 5 voxels
        y_width = 2 * padding + 6 * layer_thickness
        z_width = 3 * padding + 2 * z_thickness
        x_width = 2 * padding + x_thickness
        self.volume = (
            12 * z_thickness * layer_thickness * (x_width - 2 * padding)
        )  # volume of interest, in voxels.
        self.background = background
        raw = np.full(
            (x_width, y_width, z_width),
            background,  # This id should not belong to the region of interest.
        )
        self.layer_ids = [  # from layer 6 to layer 1
            [844, 526322264],
            [767, 12996],
            [12995, 12995],
            [11073, 11073],
            [1073, 12994],
            [68, 320],
        ]
        for i, ids in enumerate(self.layer_ids):
            y_inc = padding + layer_thickness * i
            # Left hemisphere
            raw[
                padding : (x_width - padding),
                y_inc : (y_inc + layer_thickness),
                padding : (padding + z_thickness // 2),
            ] = ids[0]
            raw[
                padding : (x_width - padding),
                y_inc : (y_inc + layer_thickness),
                (padding + z_thickness // 2) : (padding + z_thickness),
            ] = ids[1]
            # Right hemipshere
            raw[
                padding : (x_width - padding),
                y_inc : (y_inc + layer_thickness),
                (2 * padding + z_thickness) : (2 * padding + (3 * z_thickness) // 2),
            ] = ids[1]
            raw[
                padding : (x_width - padding),
                y_inc : (y_inc + layer_thickness),
                (2 * padding + (3 * z_thickness) // 2) : (2 * padding + 2 * z_thickness),
            ] = ids[0]
        self.annotation = VoxelData(raw, (10.0, 10.0, 10.0))
        self.region_map_dict = {
            "id": 0,
            "acronym": "root",
            "children": [
                {
                    "id": 315,
                    "acronym": "Isocortex",
                    "children": [
                        {
                            "id": 184,
                            "acronym": "FRP",
                            "children": [
                                {"id": 68, "acronym": "FRP1"},
                                {"id": 1073, "acronym": "FRP2"},
                                {"id": 11073, "acronym": "FRP3"},
                                {"id": 526322264, "acronym": "FRP6b"},
                            ],
                        },
                        {
                            "id": 500,
                            "acronym": "MO",
                            "children": [
                                {"id": 320, "acronym": "MOp1"},
                                {"id": 767, "acronym": "MOs5"},
                                {"id": 844, "acronym": "MOp6a"},
                            ],
                        },
                        {
                            "id": 453,
                            "acronym": "SS",
                            "children": [
                                {"id": 12995, "acronym": "SS4"},
                                {"id": 12994, "acronym": "SS2"},
                                {"id": 12996, "acronym": "SS5"},
                            ],
                        },
                    ],
                },
                {
                    "id": 698,
                    "acronym": "OLF",
                    "children": [{"id": 507, "acronym": "MOB"}],
                },
            ],
        }
        self.region_map = RegionMap.from_dict(self.region_map_dict)


class Ca1Mock:
    """
    Class to instantiate an annotated volume with some of the
    mouse CA1 features.

    The constructor allows to specify the dimensions of the volume and
    a uniform layer thickness along the y-axis.
    """

    def __init__(
        self,
        padding: int,
        layer_thickness: int,
        x_thickness: int,
        z_thickness: int,
        background: int = 0,
    ):
        self.thickness = layer_thickness  # layer thickness along the y-axis.
        self.padding = padding  # Regiodesics will crash with a padding <= 5 voxels
        y_width = 2 * padding + 4 * layer_thickness
        z_width = 2 * padding + z_thickness
        x_width = 2 * padding + x_thickness
        self.volume = (
            8 * z_thickness * layer_thickness * (x_width - 2 * padding)
        )  # volume of interest, in voxels.
        self.background = background
        raw = np.full(
            (x_width, y_width, z_width),
            background,  # This id should not belong to the region of interest.
        )
        self.layer_ids = [391, 415, 407, 399]  # from layer 4 to layer 1
        for i, id_ in enumerate(self.layer_ids):
            y_inc = padding + layer_thickness * i
            raw[
                padding : (x_width - padding),
                y_inc : (y_inc + layer_thickness),
                padding : (z_width - padding),
            ] = id_
        self.annotation = VoxelData(raw, (10.0, 10.0, 10.0))
        self.region_map_dict = {
            "id": 0,
            "acronym": "root",
            "children": [
                {
                    "id": 382,
                    "acronym": "CA1",
                    "children": [
                        {
                            "id": 399,
                            "acronym": "CA1so",
                        },
                        {
                            "id": 407,
                            "acronym": "CA1sp",
                        },
                        {
                            "id": 415,
                            "acronym": "CA1sr",
                        },
                        {
                            "id": 391,
                            "acronym": "CA1slm",
                        },
                    ],
                },
                {
                    "id": 698,
                    "acronym": "OLF",
                    "children": [{"id": 507, "acronym": "MOB"}],
                },
            ],
        }
        self.region_map = RegionMap.from_dict(self.region_map_dict)


class ThalamusMock:
    """
    Class to instantiate an annotated volume with some of the mouse thalamus features.

    The constructor allows to specify the dimensions of a box and a uniform padding,
    as well as the thickness ratio of the smallest layer (RT) wrt the largest (RT complement).
    """

    def __init__(self, padding: int, shape: Tuple[int, int, int], layer_thickness_ratio: float):
        self.padding = padding
        self.shape = shape
        self.layer_thickness_ratio = layer_thickness_ratio

        raw = np.zeros(shape, dtype=int)
        reticular_nucleus_thickness = int(layer_thickness_ratio * shape[0])
        raw[:reticular_nucleus_thickness, ...] = 262  # Region id of the reticular nucleus (RT)
        raw[reticular_nucleus_thickness:, ...] = 549  # Region id of the thalamus (TH)

        self.volume = shape[0] * shape[1] * shape[2]  # Number of voxels with positive labels

        raw = np.pad(raw, padding, "constant", constant_values=0)
        self.annotation = VoxelData(raw, (10.0, 10.0, 10.0))
        self.region_map_dict = {
            "id": 0,
            "acronym": "root",
            "children": [
                {
                    "id": 549,
                    "acronym": "TH",
                    "children": [
                        {
                            "id": 856,
                            "acronym": "DORpm",
                            "children": [{"acronym": "RT", "id": 262}],
                        },
                        {
                            "id": 864,
                            "acronym": "DORsm",
                        },
                    ],
                },
            ],
        }
        self.region_map = RegionMap.from_dict(self.region_map_dict)
