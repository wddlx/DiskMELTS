(unfitted-lines)=
# Unfitted lines

JWST spectra of disks also contain lines that DiskMELTS does not model: H I
recombination lines, H$_2$ and HD rotational lines, and atomic
fine-structure lines such as [Ne II], [Ne III], [Ar II], and [Ar III]. The
package bundles their vacuum wavelengths so you can mark them on fit plots.
Marked lines help you read the residuals and keep `fit_ranges` clear of
these features. The catalogue and the OH comparison below were added in 2.4.0.

## Plot the lines with a fit

`plot_saved_observation` and `plot_fit` take `show_lines=True`. Each line is
drawn as a gray dashed line (alpha 0.6) in both panels, with its name above
the top panel. Lines too close to label separately share one label.
`line_species` limits the marks to some species:

```python
from diskmelts import plot_saved_observation

figures = plot_saved_observation(
    result_dir='results/my_source', stages=stages, name='my_source',
    show=True, savefig=False,
    show_lines=True,                         # all bundled lines
    # line_species=['H2', 'HI', 'NeII'],     # or only some species
)
```

The default is `show_lines=False`, so existing plots are unchanged.

## Plot the lines on any axis

`plot_line_markers` draws the same markers on any Matplotlib axis with
wavelength in µm on x. By default it marks the lines inside the axis's
current x-limits, so set the limits first. `text_y=1.01` puts the names above
the axis; the default, 0.98, puts them inside it.

```python
import matplotlib.pyplot as plt
from diskmelts import plot_line_markers

fig, ax = plt.subplots()
ax.plot(wav, flux, color='k', lw=0.7)
ax.set_xlim(11.8, 17.5)
plot_line_markers(ax, text_y=1.01)
plot_line_markers(ax, species=['H2', 'NeII'])   # a subset
plot_line_markers(ax, category='HI', label=False)
```

The example below marks all bundled lines on the continuum-subtracted Upper
Sco J16120505 spectrum (`ttauri` preset).

```{image} _static/examples/unfitted_lines_markers.png
:alt: Continuum-subtracted J16120505 spectrum at 5-9 and 11.8-17.5 microns with gray dashed markers and names for H I, H2, HD, and atomic lines.
:width: 100%
```

(oh-model-overlay)=
## Compare OH models with the residuals

OH is not in the line list. Its mid-infrared emission in disks is usually not
in LTE (prompt emission after water photodissociation populates very high
rotational levels), so DiskMELTS does not recommend fitting it with an LTE
slab. You can still compare the residuals with LTE OH spectra at chosen
temperatures. Pass `oh_T` (one value or a list, 100–6000 K) to
`plot_saved_observation` or `plot_fit`:

```python
figures = plot_saved_observation(
    result_dir='results/my_source', stages=stages, name='my_source',
    show=True, savefig=False,
    oh_T=[1000, 4000],   # LTE OH models in the residual panels
    oh_sigma=0,          # the sigma used for the fit
)
```

Each OH model is scaled so its peak equals the peak of the plotted residual,
then shifted down by 0.4 × that peak, so it sits just below the residual.
Both peaks are taken over the plotted wavelength range. The column density
is `oh_logN=14` by default; there the emission is optically thin and the
normalized shape depends on temperature only. With the default
`oh_sigma=None`, `plot_fit` uses the sigma recorded in its results
(`fit_stage` and `fit_observation` record `grid_sigma`). Saved CSV results do
not record it, so pass `oh_sigma=2` to `plot_saved_observation` for a sigma=2
fit; its default is 0. Temperatures outside 100–6000 K raise an error before
any figure is drawn.

The example below shows the final residual of the Upper Sco J16120505 fit at
12.5–24.5 µm with 1000 K and 4000 K OH models.

```{image} _static/examples/oh_model_residuals.png
:alt: J16120505 fit at 12.5-24.5 microns; the residual panel shows 1000 K and 4000 K LTE OH models scaled to the residual peak and shifted below it.
:width: 100%
```

These models come from a separate OH model for 100–6000 K and logN 13–19.5,
with three overlapping wavelength tiles across 4.9–27 µm. Hot OH emits
strongly below 10 µm, so this model covers the whole MIRI range. Its
checkpoints are listed in `OH_BROAD_T_SPECS`, and their held-out accuracy is
in {ref}`oh-broad-t-bank`. Fits still use the
100–1400 K, 15–26 µm OH checkpoint in `MODEL_SPECS`; `load_fitting_models`
and `fit_observation` do not accept the broad model. To draw the overlay on
your own axis, or get the model flux:

```python
from diskmelts import oh_model_flux, plot_oh_model

flux = oh_model_flux(3000, wav, logN=14, sigma=0)  # Jy for A=1 au² at 140 pc
plot_oh_model(ax, wav, residual, [1000, 4000], offset=-0.4)
```

`oh_model_flux` returns zero outside 4.9–27 µm. To get the broad model on its
own wavelength axis, use `synthesize_spectrum('OH_broadT', T, logN)`.

## Read the catalogue

`load_line_list` returns the catalogue as a pandas DataFrame. Filter by
`species`, `category` (`'HI'`, `'molecular'`, or `'atomic'`), or `wav_range`:

```python
from diskmelts import load_line_list

lines = load_line_list(category='atomic', wav_range=(10, 20))
print(lines[['species', 'transition', 'wavelength_um']])
```

The table lives in the installed package at
`diskmelts/data/unfitted_lines.csv`. Its columns are `species`,
`transition`, `wavelength_um`, `label` (the plot text), `category`,
`reference`, and `source_url`.

## Sources

All wavelengths are vacuum values, between 4.8 and 28.6 µm, from the MPE
ISO line lists:

- H$_2$ pure-rotational (v=0-0 S) and HD (v=0-0 R) lines: the
  [molecular list](https://www.mpe.mpg.de/ir/ISO/linelists/Molecular.html)
  compiled by E. F. van Dishoeck (1994), from Jennings et al. (1987) and
  Ulivi et al. (1991).
- Atomic fine-structure lines: the
  [fine-structure list](https://www.mpe.mpg.de/ir/ISO/linelists/FSlines.html),
  from Feuchtgruber et al. (1997, ApJ 487, 962; 2001, ApJS 136, 221).
- H I: the [hydrogenic list](https://www.mpe.mpg.de/ir/ISO/linelists/Hydrogenic.html)
  by P. van Hoof, with wavelengths averaged over l sub-levels. Only
  transitions with $\Delta n \le 3$ and $n_\mathrm{upper} \le 14$ are included.

## Line list

### Molecular (H$_2$, HD)

| `species` | Transition | λ (µm, vacuum) | Label |
|---|---|---|---|
| `H2` | v=0-0 S(8) | 5.05303 | H$_2$ S(8) |
| `H2` | v=0-0 S(7) | 5.51116 | H$_2$ S(7) |
| `H2` | v=0-0 S(6) | 6.10856 | H$_2$ S(6) |
| `H2` | v=0-0 S(5) | 6.90952 | H$_2$ S(5) |
| `H2` | v=0-0 S(4) | 8.02505 | H$_2$ S(4) |
| `H2` | v=0-0 S(3) | 9.66491 | H$_2$ S(3) |
| `HD` | v=0-0 R(10) | 11.57346 | HD R(10) |
| `H2` | v=0-0 S(2) | 12.27861 | H$_2$ S(2) |
| `HD` | v=0-0 R(9) | 12.47181 | HD R(9) |
| `HD` | v=0-0 R(8) | 13.59265 | HD R(8) |
| `HD` | v=0-0 R(7) | 15.25104 | HD R(7) |
| `HD` | v=0-0 R(6) | 16.89381 | HD R(6) |
| `H2` | v=0-0 S(1) | 17.03483 | H$_2$ S(1) |
| `HD` | v=0-0 R(5) | 19.43100 | HD R(5) |
| `HD` | v=0-0 R(4) | 23.03376 | HD R(4) |
| `H2` | v=0-0 S(0) | 28.21883 | H$_2$ S(0) |
| `HD` | v=0-0 R(3) | 28.50197 | HD R(3) |

### H I recombination

| `species` | Transition | λ (µm, vacuum) | Label |
|---|---|---|---|
| `HI` | 9-6 | 5.908213 | H I 9-6 |
| `HI` | 6-5 | 7.459858 | H I 6-5 |
| `HI` | 8-6 | 7.502493 | H I 8-6 |
| `HI` | 10-7 | 8.760064 | H I 10-7 |
| `HI` | 9-7 | 11.308695 | H I 9-7 |
| `HI` | 7-6 | 12.371898 | H I 7-6 |
| `HI` | 11-8 | 12.387168 | H I 11-8 |
| `HI` | 10-8 | 16.209104 | H I 10-8 |
| `HI` | 12-9 | 16.880628 | H I 12-9 |
| `HI` | 8-7 | 19.061898 | H I 8-7 |
| `HI` | 13-10 | 22.331574 | H I 13-10 |
| `HI` | 11-9 | 22.340457 | H I 11-9 |
| `HI` | 9-8 | 27.803379 | H I 9-8 |

### Atomic fine-structure

| `species` | Transition | λ (µm, vacuum) | Label |
|---|---|---|---|
| `FeII` | a4F7/2-a6D7/2 | 4.889137 | [Fe II] |
| `FeII` | a4F5/2-a6D3/2 | 5.062350 | [Fe II] |
| `FeII` | a4F9/2-a6D9/2 | 5.340169 | [Fe II] |
| `FeII` | a4F7/2-a6D5/2 | 5.673905 | [Fe II] |
| `NiII` | 2D3/2-2D5/2 | 6.636000 | [Ni II] |
| `FeII` | a4F9/2-a6D7/2 | 6.721283 | [Fe II] |
| `ArII` | 2P1/2-2P3/2 | 6.985274 | [Ar II] |
| `ArIII` | 3P1-3P2 | 8.991380 | [Ar III] |
| `SIV` | 2P3/2-2P1/2 | 10.510500 | [S IV] |
| `NiII` | 4F7/2-4F9/2 | 10.682200 | [Ni II] |
| `ClI` | 2P1/2-2P3/2 | 11.333347 | [Cl I] |
| `NiII` | 4F5/2-4F7/2 | 12.728800 | [Ni II] |
| `NeII` | 2P1/2-2P3/2 | 12.813550 | [Ne II] |
| `NeIII` | 3P1-3P2 | 15.555100 | [Ne III] |
| `FeII` | a4F7/2-a4F9/2 | 17.935950 | [Fe II] |
| `NiII` | 4F3/2-4F5/2 | 18.240500 | [Ni II] |
| `SIII` | 3P2-3P1 | 18.713000 | [S III] |
| `ArIII` | 3P0-3P1 | 21.830200 | [Ar III] |
| `FeI` | a5D3-a5D4 | 24.042330 | [Fe I] |
| `FeII` | a4F5/2-a4F7/2 | 24.519250 | [Fe II] |
| `SI` | 3P1-3P2 | 25.249000 | [S I] |
| `FeII` | a6D7/2-a6D9/2 | 25.988290 | [Fe II] |
