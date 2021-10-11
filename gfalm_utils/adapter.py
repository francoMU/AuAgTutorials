# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)

"""
Module containing adapters for the structure object of :py:class:`pymatgen.structure.Structure`
"""
import copy
import typing

import numpy as np
from pymatgen import Structure

from gfalm.params.parser.alloy import AlloySection
from gfalm.params.parser.structure import StructureSection


class PymatgenStructureAdapter(dict):
    """
    Adapter for the :py:class:`pymatgen.structure.Structure` to the GreenALM structure input
    """

    def __init__(
        self,
        structure: Structure,
        input_params: typing.Dict,
        volume: typing.Optional[float] = None,
        name: typing.Optional[str] = None,
    ):

        super().__init__()

        _input_params = copy.copy(input_params)
        self.update(**_input_params)

        alloy_input = {}

        frac_coords = []
        site_params_list = []

        default_alloy_params = get_default_alloy_params()

        default_structure_params = get_default_structure_params()

        for site in structure:

            frac_coords.append(site.frac_coords)

            site_params = {}

            for keys in default_structure_params.keys():
                try:
                    site_params[keys] = getattr(site, keys)
                except AttributeError:
                    site_params[keys] = default_structure_params[keys]

            if site.is_ordered:

                for param, value in site_params.items():
                    try:
                        if len(value) == 1:
                            site_params[param] = value[0]
                    except TypeError:
                        pass

                site_params["species"] = str(site.specie)

            else:
                components = []

                for keys in default_structure_params.keys():
                    site_params[keys] = default_structure_params[keys]

                for idx, specie in enumerate(site.species):
                    params = {"conc": site.species[specie]}

                    for keys in default_alloy_params.keys():
                        try:
                            params[keys] = getattr(site, keys)[idx]
                        except AttributeError:
                            params[keys] = default_alloy_params[keys]

                    components.append((str(specie), params))

                site_key = f"site{len(alloy_input)}"
                alloy_input[site_key] = components

                site_params["alloy_site"] = site_key

            site_params_list.append(site_params)

        struct_input = self.__getitem__("struct_input")
        struct_input["lattice"] = structure.lattice.matrix
        struct_input["sites_frac"] = (
            np.transpose(frac_coords),
            site_params_list,
        )

        if volume is not None:
            struct_input["lattice_scale"] = None
            ws_radius = np.cbrt(volume * 3 / (4 * np.pi * structure.num_sites))
            struct_input["ws_radius"] = (ws_radius, "A")

        self.__setitem__("alloy_input", alloy_input)

        if name is not None:
            ctrl_input = self.__getitem__("ctrl_input")
            ctrl_input["job_name"] = name


def get_default_alloy_params() -> typing.Dict:
    """Get default alloy parameters"""
    return {
        param: default[0] for param, default in AlloySection([]).site_params.items()
    }


def get_default_structure_params() -> typing.Dict:
    """Get default structure parameters"""
    return {
        param: default[0] for param, default in StructureSection([]).site_params.items()
    }
