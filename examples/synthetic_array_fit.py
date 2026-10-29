"""Generate and fit HCN entirely from arrays with the current source checkout.

Run after ``python -m pip install -e .``::

    python examples/synthetic_array_fit.py --output-dir results/synthetic_hcn

The ``synthesize_spectrum`` helper and array form of ``fit_observation`` are
in the local 2.2.1 source checkout; they are not in published PyPI 2.2.0.
"""

import argparse
from pathlib import Path

import numpy as np

from diskmelts import fit_observation, synthesize_spectrum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('results/synthetic_hcn'))
    parser.add_argument('--sigma', type=int, choices=(0, 2), default=2)
    parser.add_argument('--n-samples', type=int, default=128)
    parser.add_argument('--no-errorbars', action='store_true',
                        help='Skip MCMC error bars and report the best fit only')
    args = parser.parse_args(argv)

    wavelength = np.linspace(13.8, 14.2, 121)
    wave, flux = synthesize_spectrum(
        'HCN', T=300, logN=18, A=1, sigma=args.sigma,
        wavelength=wavelength,
    )
    error = np.full_like(wave, 0.001)
    stages = [{'name': 'hcn', 'mol': 'HCN',
               'wavelength_range': (11.5, 17.0),
               'fit_ranges': (13.8, 14.2)}]
    result = fit_observation(
        stages=stages, obs_wav=wave, obs_flux=flux, obs_err=error,
        name='synthetic_hcn', sigma=args.sigma, output_dir=args.output_dir,
        detection_screening=False, n_samples=args.n_samples,
        n_refine=2, n_top=2, errorbars=not args.no_errorbars,
    )
    print('Fitted HCN:', result['fits']['hcn']['params']['HCN'])
    if result['fits']['hcn']['uncertainty']:
        print('Error bars:', result['fits']['hcn']['uncertainty']['HCN'])
    return result


if __name__ == '__main__':
    main()
