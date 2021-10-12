from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(path.as_posix())

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from pymatgen import Structure

from thermal_exp.core.eos import EosFactory
from thermal_exp.core.quasiharmonic import QuasiharmonicApprox

from gfalm.main import run
from gfalm.mpi import is_master
from gfalm.utils import read_input_file

from gfalm_utils.adapter import PymatgenStructureAdapter

eq_volumes = []
eq_bulk_modulus = []

concentration = np.arange(0.2, 0.8, 0.2)

for idx, conc in enumerate(concentration):

    structure = Structure(
        lattice=[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
        species=[{"Au": conc, "Ag": 1 - conc}],
        coords=[[0.0, 0.0, 0.0]],
    )

    base_input_params = read_input_file("base.in")

    volumes = np.linspace(14, 19, 8)

    energies = []

    for volume in volumes:
        result = run(
            **PymatgenStructureAdapter(
                structure, base_input_params, volume=volume, name=f"{idx}-auag-{volume:.6f}"
            )
        )
        energy = result["general"].total_free_energy_xc3
        energies.append(energy)

    eos_no = EosFactory("birch_murnaghan", volumes, energies)

    debye_model = QuasiharmonicApprox(volumes, energies, structure)

    temperature = 300

    thermal_energies = debye_model.vibrational_free_energies(temperature, volumes)
    free_energies = thermal_energies + energies
    eos = EosFactory("birch_murnaghan", volumes, free_energies)
    volume_eq = eos.volume_eq

    eq_volumes.append(np.cbrt(volume_eq * 4))
    eq_bulk_modulus.append(eos.bulk_modulus_eq.to("GPa"))

pprint(concentration)
pprint(eq_volumes)
pprint(eq_bulk_modulus)
