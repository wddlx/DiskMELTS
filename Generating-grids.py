"""Generate IRIS slab-model grids for the DiskMELTS molecules.

This file is designed to be run as a Jupyter notebook cell (or as a Python
script) on the HPC.  It never changes the working directory.  Grid files are
written below ``OUTPUT_ROOT/<molecule>/``.
"""

from pathlib import Path
import sys
from time import time

import astropy.constants as const
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from astropy.table import Table


# ---------------------------------------------------------------------------
# HPC configuration
# ---------------------------------------------------------------------------

# Add the directory that CONTAINS the ``iris`` package, not the package itself.
IRIS_REPOSITORY = Path("/home/u23/cyxie/iris")
HITRAN_DIRECTORY = IRIS_REPOSITORY / "HITRAN"
OBSERVED_SPECTRUM = Path(
    "/home/u23/cyxie/Usco/Consub_data/"
    "j16075796_v9.0_contsub_RVcorr.csv"
)

# Create one molecule directory directly below the directory in which the
# notebook was launched.  No os.chdir() is used anywhere in this workflow.
OUTPUT_ROOT = Path(".")

if str(IRIS_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(IRIS_REPOSITORY))

import iris  # noqa: E402  (import after adding the HPC repository to sys.path)


# Monoisotopic molecular masses in atomic mass units (u).  IRIS uses these
# values only for thermal broadening; explicit isotope masses are therefore
# used for 13CO2 and 13C12CH2.
MOLECULAR_WEIGHTS = {
    "H2O": 18.010565,
    "C2H2": 26.015650,
    "HCN": 27.010899,
    "CO2": 43.989830,
    "13CO2": 44.993185,
    "13C12CH2": 27.019005,
    "C4H2": 50.015650,
    "CH4": 16.031300,
    "C2H6": 30.046950,
    "HC3N": 51.010899,
    "OH": 17.002740,
    "CH3": 15.023475,
    "C2H4": 28.031300,
}

MOLECULES = list(MOLECULAR_WEIGHTS)

# Physical/model configuration.
DISTANCE_PC = 140.0
INCLINATION_DEG = 0.0  # retained for documentation; the slab API has no inc arg
FINE_DW_UM = 1.0e-5
LINE_FWHM_KMS = 0.0
LOG_A_FIXED = 0.0
LOG_N_GRID = np.linspace(13.0, 19.0, 49)
TEMPERATURE_GRID_K = np.linspace(50.0, 1400.0, 55)

# MIRI/MRS sub-band limits and R(lambda) coefficients from the original
# workflow (Klaus et al. 2023, FZ Tau, Table 3).
WBAND_HIGH = np.array(
    [5.74, 6.63, 7.65, 8.77, 10.13, 11.70,
     13.47, 15.57, 17.98, 20.95, 24.48, 28.10],
    dtype=float,
)
RESOLUTION_A = np.array(
    [-19.5, 2742, -543, 332, -331, -231,
     -5120, -1871, -2445, -2166, -1176, -3601],
    dtype=float,
)
RESOLUTION_B = np.array(
    [572, 150, 601, 400, 400, 264, 633, 317, 312, 225, 150, 216],
    dtype=float,
)

# Use non-overlapping partitions of the overlapping MIRI/MRS sub-bands.  Each
# internal boundary is the midpoint of one sub-band's upper limit and the next
# sub-band's lower limit.  This keeps the saved wavelength axis monotonic while
# allowing IRIS to use a different resolving power in every wavelength segment.
MIRI_SEGMENT_EDGES = np.array(
    [4.900, 5.700, 6.580, 7.580, 8.720, 10.075, 11.625,
     13.405, 15.490, 17.840, 20.820, 24.335, 27.000],
    dtype=float,
)
USE_WINDOWS = list(zip(MIRI_SEGMENT_EDGES[:-1], MIRI_SEGMENT_EDGES[1:]))


def load_observed_wavelengths(path):
    """Load and validate wavelength/flux columns from the reference CSV."""
    data = np.genfromtxt(path, skip_header=1, delimiter=",", usecols=(0, 1))
    data = np.atleast_2d(data)
    valid = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
    waves = data[valid, 0].astype(np.float64)
    fluxes = data[valid, 1].astype(np.float64)

    if waves.size == 0:
        raise ValueError(f"No finite wavelength/flux rows found in {path}")

    order = np.argsort(waves)
    waves = waves[order]
    fluxes = fluxes[order]
    if np.any(np.diff(waves) <= 0):
        raise ValueError("Observed wavelengths must be unique and increasing")
    return waves, fluxes


def resolving_power_at(wavelength_um):
    """Return the original piecewise MIRI resolving-power approximation."""
    band_index = int(np.searchsorted(WBAND_HIGH, wavelength_um, side="right"))
    if band_index >= len(WBAND_HIGH):
        raise ValueError(
            f"Wavelength {wavelength_um} um exceeds the calibrated upper "
            f"limit of {WBAND_HIGH[-1]} um"
        )
    return round(
        float(RESOLUTION_A[band_index] + RESOLUTION_B[band_index] * wavelength_um),
        1,
    )


def prepare_wavelength_windows(waves, fluxes, use_windows):
    """Select observed pixels and construct the fine IRIS wavelength grids."""
    selected_waves = []
    selected_fluxes = []
    fine_waves = []
    resolving_powers = []

    for window_index, (low, high) in enumerate(use_windows):
        if low >= high:
            raise ValueError(f"Invalid wavelength window {(low, high)}")
        # Internal segments are half-open so a pixel exactly on a shared
        # boundary is written once.  The final segment includes 27.0 um.
        if window_index < len(use_windows) - 1:
            mask = (waves >= low) & (waves < high)
        else:
            mask = (waves >= low) & (waves <= high)
        if not np.any(mask):
            raise ValueError(
                f"Wavelength window {(low, high)} selects no observed pixels"
            )

        selected_waves.append(waves[mask])
        selected_fluxes.append(fluxes[mask])
        fine_waves.append(np.arange(low - 0.1, high + 0.1, FINE_DW_UM))
        # One representative R per segment, evaluated at its central
        # wavelength using that MIRI/MRS sub-band's linear calibration.
        resolving_powers.append(resolving_power_at(0.5 * (low + high)))

    return selected_waves, selected_fluxes, fine_waves, resolving_powers


def build_forward_model(
    molecule,
    molecular_weight,
    selected_waves,
    fine_waves,
    resolving_powers,
):
    """Create molecule-specific IRIS slabs and a JIT-compiled forward model."""
    slabs = [
        iris.slab(
            molecules=[molecule],
            wlow=USE_WINDOWS[i][0] - 0.15,
            whigh=USE_WINDOWS[i][1] + 0.15,
            path_to_moldata=str(HITRAN_DIRECTORY),
        )
        for i in range(len(USE_WINDOWS))
    ]

    mass_u = jnp.asarray([[molecular_weight]], dtype=jnp.float64)
    proton_mass_cgs = const.m_p.cgs.value
    boltzmann_cgs = const.k_B.cgs.value

    def forward(log_t, log_n):
        # Preserve the (n_molecules, n_components) array convention expected
        # by IRIS even though each grid contains one molecule/component.
        temperature = jnp.asarray([[10.0**log_t]], dtype=jnp.float64)
        column_density = jnp.asarray([[10.0**log_n]], dtype=jnp.float64)
        emitting_area = jnp.asarray([[10.0**LOG_A_FIXED]], dtype=jnp.float64)

        thermal_dv = (
            jnp.sqrt(boltzmann_cgs * temperature / (mass_u * proton_mass_cgs))
            / 100000.0
        )
        total_dv = jnp.sqrt(thermal_dv**2 + LINE_FWHM_KMS**2)

        for i, slab in enumerate(slabs):
            slab.setup_disk(
                DISTANCE_PC,
                temperature,
                column_density,
                emitting_area,
                total_dv,
            )
            slab.setup_grid(fine_waves[i], selected_waves[i], resolving_powers[i])
            slab.simulate()

        return tuple(slab.downsampled_flux for slab in slabs)

    return jax.jit(forward)


def generate_molecule_grid(
    molecule,
    molecular_weight,
    selected_waves,
    fine_waves,
    resolving_powers,
):
    """Generate and save the complete (T, logN) grid for one molecule."""
    output_directory = OUTPUT_ROOT / molecule
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"\n--- {molecule}: molecular weight = {molecular_weight:.6f} u ---")
    print(f"Output directory: {output_directory.resolve()}")
    start = time()

    forward = build_forward_model(
        molecule,
        molecular_weight,
        selected_waves,
        fine_waves,
        resolving_powers,
    )
    output_wavelengths = np.concatenate(selected_waves)
    n_models = len(LOG_N_GRID) * len(TEMPERATURE_GRID_K)

    model_number = 0
    for log_n in LOG_N_GRID:
        for temperature in TEMPERATURE_GRID_K:
            model_number += 1
            model_segments = forward(
                jnp.log10(temperature),
                jnp.asarray(log_n, dtype=jnp.float64),
            )
            model_flux = np.concatenate(
                [np.asarray(segment, dtype=np.float64) for segment in model_segments]
            )

            if model_flux.shape != output_wavelengths.shape:
                raise RuntimeError(
                    f"IRIS returned {model_flux.size} flux values for "
                    f"{output_wavelengths.size} wavelength values ({molecule})"
                )

            peak_flux = float(np.nanmax(model_flux))
            if not np.isfinite(peak_flux):
                raise RuntimeError(
                    f"IRIS returned non-finite flux for {molecule}, "
                    f"T={temperature}, logN={log_n}"
                )
            threshold = max(peak_flux, 0.0) * 1.0e-4
            saved_flux = np.where(model_flux < threshold, 0.0, model_flux)

            filename = f"T{int(round(temperature))}N{log_n}.csv"
            table = Table({"wave": output_wavelengths, "Line": saved_flux})
            table.write(output_directory / filename, format="ascii.csv", overwrite=True)

            if model_number == 1 or model_number % 100 == 0:
                print(f"  {model_number:4d}/{n_models} models written")

    elapsed = time() - start
    print(f"Completed {molecule}: {n_models} models in {elapsed / 60.0:.1f} min")


# ---------------------------------------------------------------------------
# Sequential generation: one molecule is completed before the next begins.
# ---------------------------------------------------------------------------

print("-- Loading reference wavelength grid --")
observed_waves, observed_fluxes = load_observed_wavelengths(OBSERVED_SPECTRUM)
Xs, ys, fine_Xs, Rs = prepare_wavelength_windows(
    observed_waves,
    observed_fluxes,
    USE_WINDOWS,
)
print(f"Resolving powers: {Rs}")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for mol_name, mol_weight in MOLECULAR_WEIGHTS.items():
    generate_molecule_grid(
        mol_name,
        mol_weight,
        Xs,
        fine_Xs,
        Rs,
    )

print("\nAll molecular grids completed.")
