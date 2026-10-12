# Continuum subtraction

Use `diskmelts.consub.calculate()` to subtract a smooth continuum before fitting
molecular lines. It returns **five NumPy arrays** in this order:

```text
wav, consub, err, cont, contOrig
```

`cont` is the final continuum; `contOrig` is the baseline before correction using
line-free reference windows. `consub = raw_flux - cont`. Inputs must be matching
one-dimensional arrays: wavelength in µm and flux/error in the same units.
Wavelength must be positive, finite, unique, increasing, and aligned with the
reference templates' rest wavelengths. Use Jy for subsequent molecular fitting.
Nonfinite flux becomes NaN in the result.

## Read, calculate, and plot

This notebook example reads a CSV containing `wave,flux,err` columns. `err` may
be omitted from the file; in that case the helper returns `None` for error.

```python
import diskmelts

wav, flux, error = diskmelts.consub.read_spectrum("raw_spectrum.csv")
wav, consub, err, cont, contOrig = diskmelts.consub.calculate(
    wav, flux, error, preset="ttauri", exclude=[[12, 13]],
)
fig = diskmelts.consub.plot(
    wav, consub, err, cont, contOrig, ranges=[(5, 12), (12, 25)], savedir=None,
)
```

The FITS helper reads extension 1 columns `wavelength`, `fluxdensity`, and
`fluxdensity_stddev`, matching the supplied USco/PPan files. FITS reading requires
the optional `io` extra; array and CSV workflows do not. For other FITS layouts,
read arrays with your preferred tool and pass them directly to `calculate()`.

`ranges` changes only the plot. To fit a restricted wavelength range, slice
`wav`, `flux`, and `error` together before calling `calculate()`.

## Presets or custom windows

| `preset` | Candidate line-free windows |
| --- | --- |
| `"water"` | Avoid H2O emission |
| `"ttauri"` | Avoid H2O, C2H2, HCN, and CO2 |
| `"carbon_rich"` | Also avoid 13C12CH2, 13CO2, C4H2, HC3N, and CH4; exclude 12.5–16.5 µm |
| `"carbon_extended"` | Carbon-rich preset plus C2H4; same broad exclusion |
| `None` | Use `linefreefile`, or fit all nonexcluded pixels if no file is supplied |

Named presets bundle both model-based candidates and curated reference windows
for the correction. They need no local model grids. H2O templates use
**800 K, logN=18**, and carbon templates use **400 K, logN=16**. Their nominal
coverage is 4.9–27 µm. The choice of preset is yours; the code does not infer
disk chemistry from the source name.

With `preset=None`, supply a CSV with `xmin,xmax` columns to choose both fitting
and correction windows; an optional `lam` column is ignored. For example:

```text
xmin,xmax
9.0049,9.0184
9.0718,9.0886
```

These two rows illustrate the format; a real file should cover your fitting
range with enough clean pixels. Use it as follows:

```python
wav, consub, err, cont, contOrig = diskmelts.consub.calculate(
    wav, flux, error, preset=None, linefreefile="my_linefree.csv",
    exclude=[[12.5, 16.5]],
)
```

Exclusions always win over line-free windows. A nonempty `linefreefile` with a
named preset raises an error rather than silently choosing one.

To fit without reference windows:

```python
wav, consub, err, cont, contOrig = diskmelts.consub.calculate(
    wav, flux, error, preset=None, linefreefile=None, exclude=[[12.5, 16.5]],
)
```

Here the baseline uses all nonexcluded pixels with robust rejection. There is
no line-free correction, so `cont` and `contOrig` are identical. An empty string
for `linefreefile` has the same behavior as `None`.

## Smoothing controls

The default `method="robust"` fits local weighted straight lines, rejects line
excursions and negative defects, and interpolates the estimates with PCHIP.
The reference correction interpolates residual medians in 0.75 µm bins.
`method="arpls"` provides an alternative penalized smooth baseline.

| Argument | Setting |
| --- | --- |
| `boxsizes=None` | Automatic widths targeting about 31 clean pixels, bounded to 0.25–1 µm |
| `boxsizes=95` | Fixed window width in pixels |
| `boxsizes=[(4.9,9.2,131), (9.2,25,51)]` | Regional pixel widths; other regions remain automatic |
| `special=[(11,16)]` | Suppress the line-free correction inside these intervals, with smooth transitions outside |
| `shift=False` | Disable the correction everywhere |
| `correction_scale=0.75` | Correction-bin width in µm |

For arPLS, the window width sets the smoothing penalty. It is not a literal
moving box. The same exclusions and reference corrections apply to both methods.

## Plot or save

`plot()` accepts the five output arrays in their returned order. `error` and
`contOrig` may be `None`; `cont` is required. The raw spectrum is reconstructed
as `consub + cont`. For example, to omit the error band and unshifted curve:

```python
fig = diskmelts.consub.plot(wav, consub, cont=cont, ranges=(12, 17))
```

`savedir=None` writes no files. Supply a directory to save a figure:

```python
fig = diskmelts.consub.plot(
    wav, consub, err, cont, contOrig,
    savedir="figures/consub", filename="my_source.pdf",
)
```

The directory is created if needed. The default filename is `continuum.png`;
an existing file with the same name is replaced. The returned figure is shown
by Jupyter's Matplotlib backend. In a script, call `matplotlib.pyplot.show()`.

Save a spectrum in the column order expected by DiskMELTS fitting readers:

```python
import pandas as pd
pd.DataFrame({"wave": wav, "flux": consub, "err": err,
              "cont": cont, "contOrig": contOrig}).to_csv("consub.csv", index=False)
```

## Errors and limitations

`err` preserves valid supplied measurement errors. Missing or nonpositive errors
are estimated from local robust adjacent differences. These errors **do not
include correlated continuum uncertainty**. Broad excluded intervals, especially
12.5–16.5 µm, remain dependent on interpolation. The extended carbon mask removes
many anchors around the 10 µm dust feature; inspect it carefully. Neither method
corrects fringes or channel offsets, and one-temperature masks cannot guarantee
line-free windows for every source or spectral resolution. Very coarsely sampled
spectra may have too few pixels in the narrow preset windows, particularly for
`carbon_extended`; use a custom window CSV with broader clean intervals in that case.

For support masks and indicative continuum uncertainties, the advanced
`diskmelts.consub.subtract_continuum()` interface returns a diagnostic DataFrame;
`plot_consub()` shades its weakly supported regions. The five-array `plot()`
cannot reconstruct those masks. See {doc}`api/consub` for these helpers.

## Research and checks

The design draws on robust clipping in [STATCONT](https://arxiv.org/abs/1710.02419),
separation of smooth and narrow components in the
[AS 209 continuum study](https://arxiv.org/html/2402.00860v1#A1), and
[arPLS reweighting](https://pybaselines.readthedocs.io/en/latest/generated/api/pybaselines.Baseline.arpls.html).
No Gaussian-process implementation is included here.

The distributed `tests/test_consub_package.py` checks the public five-array
interface, bundled presets, custom windows, and plotting in a directory with
no `0-Consub/` inputs. Additional local tests and USco/PPan examples live in
the Git-ignored `0-Consub/consub_test.py`; that directory is not distributed or
required by the package.
Local synthetic benchmarks compare both methods against known
continua and different-temperature molecular spectra. The methods perform
differently across masked gaps; these tests do not establish absolute accuracy
on real observations.
