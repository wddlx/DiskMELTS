# diskmelts.consub

For the recommended array workflow, see {doc}`../continuum`.

## Calculate and plot

```{eval-rst}
.. autofunction:: diskmelts.consub.calculate

.. autofunction:: diskmelts.consub.plot

.. autofunction:: diskmelts.consub.read_spectrum
```

## Reference windows

```{eval-rst}
.. autofunction:: diskmelts.consub.load_regions

.. autofunction:: diskmelts.consub.generate_line_free_regions
```

Generating new windows requires local model grids; using bundled presets does
not. The generator treats nonfinite template intervals as unknown. In the
supplied C2H6 400 K grid, 5.70–7.58 µm is nonfinite, so C2H6 is opt-in through
the generator's `molecules` argument. Save custom tables and use them with
`calculate(..., preset=None, linefreefile=...)`.

## Diagnostic DataFrame workflow

`subtract_continuum` accepts `line_free_regions` as pairs, a table, or a CSV path.
Its `reference_regions="curated"` default intersects those candidates with the
bundled curated windows when a preset is selected. Set `reference_regions=None`
to use all custom candidate windows for the correction.

The result includes `weak_support`, `reference_mask`, `fit_mask`, `cont_err`, and
`cont_sensitivity`. `cont_err` combines a reference-anchor estimate and smoothing
sensitivity; it is not a posterior uncertainty and is NaN where support is weak.
`cont_sensitivity` is the baseline change when doubling the smoothing width.
In `plot_consub`, green points are `reference_mask` calibration pixels, light red
marks wavelengths requested in `exclude`, and orange marks `weak_support` where
reference pixels are too distant or do not cover the wavelength. Red and orange
may overlap; orange signals less reliable continuum placement.

```{eval-rst}
.. autofunction:: diskmelts.consub.subtract_continuum

.. autofunction:: diskmelts.consub.plot_consub

.. autofunction:: diskmelts.consub.save_consub
```
