# Upper Sco raw spectra

These 14 CSV files are direct, row-for-row conversions of the local
`0-Consub/Data/USco/*_1d_v9.0.fits` spectra. Each CSV keeps the source name and
version and contains 10,856 rows with:

| CSV column | FITS extension 1 column | Unit |
| --- | --- | --- |
| `wave` | `wavelength` | µm |
| `flux` | `fluxdensity` | Jy |
| `err` | `fluxdensity_stddev` | Jy |

No continuum subtraction, wavelength trimming, rebinning, or velocity
correction was applied during conversion. The notebook
[`Example_Consub.ipynb`](../../notebooks/Example_Consub.ipynb) uses
`j16120505_1d_v9.0.csv` and selects 4.9–25 µm before estimating its
continuum. The original FITS files remain in the local `0-Consub/` directory.
