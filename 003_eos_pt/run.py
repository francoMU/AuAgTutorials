from pathlib import Path
import sys

path = Path(__file__).parent.parent
sys.path.append(path.as_posix())

import matplotlib.pyplot as plt
import numpy as np
from pymatgen import Structure

from thermal_exp.core.eos import EosFactory
from thermal_exp.core.quasiharmonic import QuasiharmonicApprox

from gfalm.main import run
from gfalm.mpi import is_master
from gfalm.utils import read_input_file

from gfalm_utils.adapter import PymatgenStructureAdapter

experimental_a = 4.065

experimental_volume = experimental_a ** 3 / 4

structure = Structure(
    lattice=[[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
    species=["Pt"],
    coords=[[0.0, 0.0, 0.0]],
)

base_input_params = read_input_file("base.in")

volumes = np.linspace(14, 19, 8)

energies = []

for volume in volumes:
    result = run(
        **PymatgenStructureAdapter(
            structure, base_input_params, volume=volume, name=f"pt-{volume:.6f}"
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

print(volume_eq)
print(np.cbrt(volume_eq * 4))
print(eos.bulk_modulus_eq.to("GPa"))

