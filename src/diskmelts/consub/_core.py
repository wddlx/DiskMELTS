"""Continuum fitting internals and diagnostic helpers (wavelengths in microns).

Only NumPy, SciPy, and pandas are required for fitting. FITS I/O and plotting
import astropy and matplotlib lazily.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import maximum_filter1d
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import expit

HERE = Path(__file__).resolve().parent
PRESETS = {
    "water": ("H2O",),
    "ttauri": ("H2O", "C2H2", "HCN", "CO2"),
    "carbon_rich": ("H2O", "C2H2", "HCN", "CO2", "13C12CH2",
                    "13CO2", "C4H2", "HC3N", "CH4"),
    "carbon_extended": ("H2O", "C2H2", "HCN", "CO2", "13C12CH2",
                        "13CO2", "C4H2", "HC3N", "CH4", "C2H4"),
}
# C2H6 is opt-in: its local 400 K template is nonfinite at 5.70–7.58 µm.
# generate_line_free_regions(..., molecules=[...]) supports it conservatively.
__all__ = ["subtract_continuum", "generate_line_free_regions", "load_regions",
           "plot_consub", "read_spectrum", "save_consub"]


def _regions(regions):
    if regions is None:
        return np.empty((0, 2))
    if isinstance(regions, (str, Path)):
        regions = pd.read_csv(regions)
    if isinstance(regions, pd.DataFrame):
        if not {"xmin", "xmax"}.issubset(regions.columns):
            raise ValueError("Line-free CSV/table must contain xmin and xmax columns")
        regions = regions[["xmin", "xmax"]].to_numpy()
    a = np.asarray(regions, dtype=float)
    if a.size == 0:
        return np.empty((0, 2))
    if a.shape == (2,):
        a = a[np.newaxis]
    if a.ndim != 2 or a.shape[1] != 2 or not np.all(np.isfinite(a)) or np.any(a[:, 1] <= a[:, 0]):
        raise ValueError("Regions must be finite (xmin, xmax) pairs with xmin < xmax")
    return a


def _inside(x, regions):
    mask = np.zeros(len(x), dtype=bool)
    for low, high in _regions(regions):
        mask |= (x >= low) & (x <= high)
    return mask


def _intervals(x, mask, min_points=2):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    boundaries = np.r_[x[0], (x[:-1]+x[1:])/2, x[-1]]
    return pd.DataFrame([(boundaries[a], boundaries[b]) for a, b in edges.reshape(-1, 2)
                         if b-a >= min_points], columns=["xmin", "xmax"])


def load_regions(preset="ttauri"):
    """Read shipped candidate windows; these are references, not guarantees."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {tuple(PRESETS)}")
    return pd.read_csv(HERE / "regions" / f"{preset}.csv")


def generate_line_free_regions(grid_root, preset="ttauri", *, molecules=None,
                               threshold=0.03, padding_kms=40.0,
                               exclude=None, reference_file=None):
    """Find windows faint in every selected model, on the model wavelength axis.

    H2O: T=800 K, logN=18; all carbon species: T=400 K, logN=16.
    Threshold is a fraction of a rolling 0.5 micron peak, floored at 2% of
    the full model peak. Padding is a wavelength-registration margin; models
    already contain instrumental broadening. Outside model coverage is unknown.
    Optional reference_file intersects the candidates with curated windows.
    """
    if preset not in PRESETS or not 0 < threshold < 1 or padding_kms < 0:
        raise ValueError("Invalid preset, threshold, or padding_kms")
    species = tuple(PRESETS[preset] if molecules is None else molecules)
    if not species:
        raise ValueError("Select at least one molecule")
    x = None
    clean = None
    missing = {}
    for molecule in species:
        temperature, column = (800, 18) if molecule == "H2O" else (400, 16)
        path = Path(grid_root) / molecule / f"T{temperature}N{column}.0.csv"
        table = pd.read_csv(path)
        mx = table["wave"].to_numpy(float)
        my = table["Line"].to_numpy(float)
        if np.any(np.diff(mx) <= 0) or not np.all(np.isfinite(mx)) or not np.any(np.isfinite(my)) or np.nanmax(my) <= 0:
            raise ValueError(f"Invalid template: {path}")
        unknown = ~np.isfinite(my)
        missing[molecule] = int(unknown.sum())
        my = np.where(unknown, 0, my)
        if x is None:
            x, clean = mx, np.ones(len(mx), dtype=bool)
        window = max(3, int(0.5 / np.median(np.diff(mx))) | 1)
        envelope = np.maximum(maximum_filter1d(my, window), 0.02 * np.max(my))
        line = (my / envelope > threshold) | unknown
        # Pad intervals on the native axis before interpolating masks, so a
        # narrow line cannot disappear between samples of a coarser spectrum.
        intervals = _intervals(mx, line, min_points=1).to_numpy()
        if len(intervals):
            intervals[:, 0] *= 1 - padding_kms / 299792.458
            intervals[:, 1] *= 1 + padding_kms / 299792.458
        clean &= ~_inside(x, intervals)
        clean &= (x >= mx[0]) & (x <= mx[-1])
    exclusions = list(_regions(exclude))
    if preset.startswith("carbon_"):
        exclusions.append((12.5, 16.5))
    clean &= ~_inside(x, exclusions)
    if reference_file is not None:
        clean &= _inside(x, reference_file)
    result = _intervals(x, clean)
    result["lam"] = (result.xmin + result.xmax) / 2
    result.attrs.update(preset=preset, molecules=list(species), threshold=threshold,
                        padding_kms=padding_kms, unknown_template_pixels=missing)
    return result


def _mad(a):
    a = np.asarray(a)
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def _interp(x, xp, yp, linear_ends=False):
    if len(xp) == 1:
        return np.full(len(x), yp[0])
    # Constant end corrections avoid unstable polynomial extrapolation.
    out = PchipInterpolator(xp, yp)(np.clip(x, xp[0], xp[-1]))
    if linear_ends:
        # A sloped continuum beyond the last knot needs a slope, not a constant;
        # fit it over about 1 um of knots so one noisy knot cannot set it.
        xp, yp = np.asarray(xp), np.asarray(yp)
        for outside, end, near in ((x < xp[0], 0, xp <= xp[0] + 1.0),
                                   (x > xp[-1], -1, xp >= xp[-1] - 1.0)):
            if outside.any() and near.sum() >= 2:
                slope = np.polyfit(xp[near], yp[near], 1)[0]
                out[outside] = yp[end] + slope * (x[outside] - xp[end])
    return out


def _widths(x, good, boxsizes):
    dx = np.gradient(x)
    if boxsizes is None:
        # Expand to about 31 clean pixels; retain a physical lower limit so
        # the baseline does not track resolved molecular lines.
        n = np.convolve(good.astype(float), np.ones(min(101, len(x))), "same")
        fraction = np.maximum(n / min(101, len(x)), 0.05)
        return np.clip(31 * dx / fraction, 0.25, 1.0)
    if np.ndim(boxsizes) == 0:
        if not np.isfinite(boxsizes) or boxsizes < 5:
            raise ValueError("boxsizes must be at least 5 pixels")
        return dx * float(boxsizes)
    widths = _widths(x, good, None)
    for low, high, pixels in boxsizes:
        if not np.isfinite([low, high, pixels]).all() or high <= low or pixels < 5:
            raise ValueError("boxsizes entries must be (xmin, xmax, pixels>=5)")
        sel = (x >= low) & (x <= high)
        widths[sel] = dx[sel] * pixels
    return widths


def _local_baseline(x, y, noise, good, widths):
    positions, values = [], []
    # Local linear fits remove slope before clipping, unlike selecting minima.
    centers = np.arange(x[0], x[-1], max(0.04, np.median(widths) / 3))
    centers = np.unique(np.r_[x[0], centers, x[-1]])
    for center in centers:
        width = np.interp(center, x, widths)
        sel = good & (np.abs(x-center) <= width/2)
        if np.count_nonzero(sel) < 5:
            continue
        xx, yy, nn = x[sel]-center, y[sel], noise[sel]
        keep = np.ones(len(xx), dtype=bool)
        for _ in range(6):
            if np.sum(keep) < 4:
                break
            coef = np.polyfit(xx[keep], yy[keep], 1, w=1/nn[keep])
            residual = yy - np.polyval(coef, xx)
            scale = np.maximum(nn, _mad(residual[keep]))
            updated = (residual < 2.5*scale) & (residual > -4*scale)
            if np.array_equal(keep, updated):
                break
            keep = updated
        if np.sum(keep) < 4:
            continue
        # Place the knot at the actual support, not the nominal box center:
        # extrapolating a slope from a tiny off-center cluster creates spikes.
        location = np.median(xx[keep])
        positions.append(center+location)
        values.append(float(np.polyval(coef, location)))
    if len(positions) < 3:
        raise ValueError("Too few continuum windows; widen boxsizes or revise masks")
    knots = pd.DataFrame({"x": positions, "y": values}).groupby("x", sort=True).median()
    return _interp(x, knots.index.to_numpy(), knots.y.to_numpy(), linear_ends=True)


def _arpls(x, y, noise, good, widths):
    # Nonuniform-grid second derivatives, scaled to a median pixel spacing.
    h = np.diff(x) / np.median(np.diff(x))
    a = 2 / (h[:-1] * (h[:-1]+h[1:]))
    c = 2 / (h[1:] * (h[:-1]+h[1:]))
    derivative = diags([a, -a-c, c], [0, 1, 2], shape=(len(x)-2, len(x)))
    lam = (widths[1:-1] / (2*np.pi*np.median(np.diff(x))))**4
    penalty = derivative.T @ diags(lam) @ derivative
    base = good.astype(float) * (np.median(noise[good])/noise)**2
    weights = base.copy()
    converged = False
    for _ in range(40):
        z = spsolve((diags(weights) + penalty).tocsc(), weights*y)
        residual = y-z
        negative = residual[good & (residual < 0)]
        if len(negative) < 3:
            # Nothing left below the baseline to down-weight: a fixed point.
            converged = True
            break
        scale = max(np.std(negative), np.median(noise[good])*0.1)
        new = base * expit(-2*(residual - (2*scale-np.mean(negative)))/scale)
        # Strong negative defects must not pin the baseline downward.
        new[residual < -5*np.maximum(noise, scale)] = 0
        if np.linalg.norm(new-weights) / max(np.linalg.norm(weights), 1e-15) < 1e-3:
            converged = True
            break
        weights = new
    return z, converged


def subtract_continuum(wav, flux, *, error=None, exclude=None, boxsizes=None,
                       special=None, preset="ttauri", line_free_regions=None,
                       reference_regions="curated",
                       auto_mask=True, method="robust", shift=True,
                       correction_scale=0.75, special_blend=0.15):
    """Return a DataFrame containing the spectrum, continuum and diagnostics.

    boxsizes: None (automatic), a pixel count, or (xmin,xmax,pixels) rows.
    special: windows where the reference-window correction is suppressed.
    line_free_regions: pairs, CSV path, or xmin/xmax DataFrame; overrides preset.
    exclude always wins over references. Nonfinite flux is retained as NaN;
    wavelength must be finite, positive, unique, and strictly increasing.
    err is measurement/local-noise error; cont_err is a separate diagnostic
    uncertainty, not a calibrated posterior. With preset=None and no custom
    windows, all nonexcluded pixels are used and reference correction is off.
    """
    x, original = np.asarray(wav, float), np.asarray(flux, float)
    if x.ndim != 1 or original.shape != x.shape or len(x) < 15:
        raise ValueError("wav and flux must be matching 1D arrays with >=15 samples")
    if not np.all(np.isfinite(x)) or np.any(x <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("wav must be finite, positive, and strictly increasing")
    if method not in ("robust", "arpls") or correction_scale <= 0 or special_blend < 0:
        raise ValueError("Invalid method or correction/blend scale")
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"preset must be one of {tuple(PRESETS)}")
    finite = np.isfinite(original)
    if finite.sum() < 15:
        raise ValueError("At least 15 finite flux samples are required")
    original = original.copy()
    original[~finite] = np.nan
    unrestricted = preset is None and line_free_regions is None
    if unrestricted:
        refs = np.array([[x[0], x[-1]]])
        shift = False
    else:
        refs = _regions(load_regions(preset) if line_free_regions is None else line_free_regions)
    excluded = _inside(x, exclude)
    if preset is not None and preset.startswith("carbon_"):
        excluded |= _inside(x, [(12.5, 16.5)])
    reference = _inside(x, refs) & finite & ~excluded
    good = finite & ~excluded
    if auto_mask:
        good &= reference
    if isinstance(reference_regions, str) and reference_regions == "curated":
        if preset is not None:
            filename = "cont_ranges_PPan.csv" if preset.startswith("carbon_") else "cont_ranges.csv"
            reference &= _inside(x, HERE / filename)
    elif reference_regions is not None:
        reference &= _inside(x, reference_regions)
    if good.sum() < 15:
        raise ValueError("Fewer than 15 usable continuum pixels; revise regions")
    y = np.interp(x, x[finite], original[finite])
    # Adjacent differences estimate noise without a continuum-level assumption.
    noise_x, noise_y = [], []
    for lo in np.arange(x[0], x[-1], 0.75):
        adjacent = finite[:-1] & finite[1:] & (x[:-1] >= lo) & (x[1:] < lo+0.75)
        delta = np.diff(y)[adjacent]
        if len(delta) >= 5:
            noise_x.append(lo+0.375)
            noise_y.append(_mad(delta)/np.sqrt(2))
    floor = max(np.nanmedian(np.abs(y))*1e-10, np.finfo(float).eps)
    if not noise_x:
        noise_x, noise_y = [np.median(x)], [_mad(np.diff(y))/np.sqrt(2)]
    estimated_noise = np.maximum(_interp(x, noise_x, noise_y), floor)
    if error is None:
        measurement = estimated_noise.copy()
    else:
        measurement = np.asarray(error, float).copy()
        if measurement.shape != x.shape:
            raise ValueError("error must have the same shape as wav")
        invalid_error = ~np.isfinite(measurement) | (measurement <= 0)
        measurement[invalid_error] = estimated_noise[invalid_error]
    noise = np.maximum(measurement, estimated_noise)
    widths = _widths(x, good, boxsizes)
    # Remove isolated negative spikes from fitting, never from returned flux.
    neighbors = (np.roll(y, 1)+np.roll(y, -1))/2
    spikes = (neighbors-y > 6*noise)
    spikes[[0, -1]] = False
    good &= ~spikes
    reference &= ~spikes
    converged = True
    if method == "robust":
        baseline = _local_baseline(x, y, noise, good, widths)
    else:
        baseline, converged = _arpls(x, y, noise, good, widths)
        if not converged:
            warnings.warn("arPLS did not converge in 40 iterations", stacklevel=2)
    # Bin reference residuals before shape-preserving interpolation. This
    # prevents a dense cluster of narrow windows dominating a global polynomial.
    anchor_x, anchor_y, anchor_err = [], [], []
    for lo in np.arange(x[0], x[-1], correction_scale):
        sel = reference & (x >= lo) & (x < lo+correction_scale)
        if sel.sum() < 5:
            continue
        residual = y[sel]-baseline[sel]
        center = np.median(residual)
        scatter = max(_mad(residual), np.median(noise[sel]))
        accepted = np.abs(residual-center) < 3*scatter
        if accepted.sum() < 5:
            continue
        anchor_x.append(float(np.median(x[sel][accepted])))
        anchor_y.append(float(np.median(residual[accepted])))
        # Samples within one resolution element are correlated; this is still
        # only an indicative anchor error, supplemented by baseline sensitivity.
        anchor_err.append(float(1.253*scatter/np.sqrt(max(1, accepted.sum()/3))))
    if not anchor_x:
        raise ValueError("Too few reference anchors; supply broader line-free windows")
    correction = _interp(x, anchor_x, anchor_y) if shift else np.zeros(len(x))
    taper = np.ones(len(x))
    for lo, hi in _regions(special):
        if special_blend == 0:
            taper[(x >= lo) & (x <= hi)] = 0
        else:
            # Exact zero inside the special window, smooth transitions outside.
            left = np.clip((x-(lo-special_blend))/special_blend, 0, 1)
            right = np.clip(((hi+special_blend)-x)/special_blend, 0, 1)
            blend = (left*left*(3-2*left)) * (right*right*(3-2*right))
            taper *= 1-blend
    correction *= taper
    continuum = baseline+correction
    # Report baseline dependence on a factor-of-two smoothing change separately.
    if method == "robust":
        alternate = _local_baseline(x, y, noise, good, widths*2)
    else:
        alternate, _ = _arpls(x, y, noise, good, widths*2)
    sensitivity = np.abs(alternate-baseline)
    cont_error = np.hypot(_interp(x, anchor_x, anchor_err), sensitivity)
    # Distance to available reference pixels flags interpolation across holes.
    idx = np.searchsorted(x[reference], x)
    left = x[reference][np.clip(idx-1, 0, reference.sum()-1)]
    right = x[reference][np.clip(idx, 0, reference.sum()-1)]
    unsupported = np.minimum(np.abs(x-left), np.abs(x-right)) > np.maximum(widths, correction_scale)/2
    unsupported |= (x < x[reference][0]) | (x > x[reference][-1])
    cont_error[unsupported] = np.nan
    if unsupported.any():
        edges = np.flatnonzero(np.diff(np.r_[0, unsupported.astype(int), 0]))
        spans = ", ".join(f"{x[a]:.2f}-{x[b-1]:.2f}" for a, b in zip(edges[::2], edges[1::2]))
        warnings.warn(f"Continuum is interpolated or extrapolated without nearby reference "
                      f"pixels at {spans} um (weak_support column); inspect the plot before "
                      "using those regions", stacklevel=2)
    result = pd.DataFrame({"wave": x, "flux": original-continuum,
                           "err": measurement, "cont": continuum,
                           "cont_unshifted": baseline, "raw_flux": original,
                           "cont_err": cont_error, "cont_sensitivity": sensitivity,
                           "noise": noise, "correction": correction,
                           "fit_mask": good, "reference_mask": reference,
                           "excluded": excluded, "weak_support": unsupported,
                           "box_um": widths})
    result.attrs.update(preset=preset, method=method, converged=converged,
                        reference_source="unrestricted" if unrestricted else "windows",
                        anchor_wave=anchor_x, anchor_offset=anchor_y,
                        anchor_error=anchor_err, special=_regions(special).tolist())
    return result


def read_spectrum(path):
    """Read local JWST FITS or CSV with wave/flux[/err] columns."""
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit"):
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError('FITS reading requires astropy: pip install "diskmelts[io]"') from exc
        with fits.open(path) as hdus:
            data = hdus[1].data
            return tuple(np.asarray(data[key], float).copy() for key in
                         ("wavelength", "fluxdensity", "fluxdensity_stddev"))
    table = pd.read_csv(path)
    return (table.wave.to_numpy(), table.flux.to_numpy(),
            table.err.to_numpy() if "err" in table else None)


def save_consub(result, path):
    """Write all numeric diagnostics; first four columns match DiskMELTS I/O."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return path


def plot_consub(result, ranges=None, *, title=None, output=None):
    """Plot the continuum and residual with reference, exclusion, and support masks."""
    import matplotlib.pyplot as plt
    x = result.wave.to_numpy()
    if ranges is None:
        ranges = [(x[0], x[-1])]
    if np.asarray(ranges).shape == (2,):
        ranges = [ranges]
    ranges = _regions(ranges)
    if not len(ranges):
        raise ValueError("At least one plotting range is required")
    fig, axes = plt.subplots(2*len(ranges), 1, figsize=(12, 4*len(ranges)),
                             squeeze=False, constrained_layout=True)
    for i, (lo, hi) in enumerate(ranges):
        sel = (x >= lo) & (x <= hi)
        if not sel.any():
            plt.close(fig)
            raise ValueError(f"No data in plotting range {lo}, {hi}")
        a, b = axes[2*i, 0], axes[2*i+1, 0]
        a.plot(x[sel], result.raw_flux[sel], color="0.55", lw=0.6, label="Raw")
        if "cont_unshifted" in result:
            a.plot(x[sel], result.cont_unshifted[sel], color="darkorange", lw=1, label="Unshifted continuum")
        a.plot(x[sel], result.cont[sel], color="tab:blue", lw=1.2, label="Continuum")
        ref = sel & result.reference_mask.to_numpy()
        if ref.any():
            a.scatter(x[ref], result.raw_flux[ref], s=7, color="seagreen",
                      alpha=.8, zorder=4, label="Calibration pixels")
        b.plot(x[sel], result.flux[sel], color="0.2", lw=0.6)
        if "err" in result:
            b.fill_between(x[sel], -result.err[sel], result.err[sel], color="tab:blue", alpha=.15, label="Measurement/noise error")
        b.axhline(0, color="tab:blue", lw=.7)
        for axis in (a, b):
            if "excluded" in result:
                for j, row in _intervals(x, result.excluded.to_numpy()).iterrows():
                    axis.axvspan(row.xmin, row.xmax, color="lightcoral", alpha=.2,
                                 label="Excluded" if axis is a and j == 0 else None)
            for j, row in _intervals(x, result.weak_support.to_numpy()).iterrows():
                axis.axvspan(row.xmin, row.xmax, color="orange", alpha=.12,
                             label="Weak continuum support" if axis is a and j == 0 else None)
            axis.set_xlim(lo, hi)
            axis.set_ylabel("Flux density")
        a.legend(fontsize=8, ncol=4)
        b.set_xlabel("Wavelength (µm)")
    if title:
        fig.suptitle(title)
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    return fig
