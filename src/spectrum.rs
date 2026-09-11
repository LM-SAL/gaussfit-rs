use std::cell::RefCell;

use crate::gaussian::{fit_gaussian_bounded, FitConfig};
use crate::{
    FLAG_NO_CONVERGENCE, FLAG_NO_LOCAL_MAX, FLAG_SUCCESS, QUALITY_PEGGED, QUALITY_UNCONSTRAINED,
    QUALITY_ZERO_ERROR,
};

thread_local! {
    // Fit-window buffers (x, y, error), reused across the fits on a thread.
    static WINDOW: RefCell<[Vec<f32>; 3]> = const { RefCell::new([Vec::new(), Vec::new(), Vec::new()]) };
}

/// Internal result returned by `fit_single_spectrum_core`.
#[derive(Clone, Copy, Debug)]
pub struct FitSingleSpectrumResult {
    /// `[amp, vel, sigma, amp_err, vel_err, sig_err, reduced_chi2, flag]`
    pub(crate) fit_results: [f32; 8],
    pub(crate) i_left: i32,
    pub(crate) i_right: i32,
    /// Quality bitmask for successful fits; zero for failures. Exposed with `quality=True`.
    pub(crate) quality: u8,
    /// Solver iterations and model evaluations for successful fits; -1 otherwise. Exposed with
    /// `meta=True`.
    pub(crate) n_iter: i32,
    pub(crate) n_fev: i32,
}

fn result_with_flag(flag: f32, i_left: i32, i_right: i32) -> FitSingleSpectrumResult {
    let mut fit_results = [f32::NAN; 8];
    fit_results[7] = flag;
    FitSingleSpectrumResult {
        fit_results,
        i_left,
        i_right,
        quality: 0,
        n_iter: -1,
        n_fev: -1,
    }
}

/// Fit one spectrum: find the peak near `guide_velocity`, normalise the window
/// around it by the peak, run the bounded fit and rescale the amplitude back.
///
/// `spectrum`, `dopp_slit`, and `spec_noise` must all have the same length.
#[allow(clippy::too_many_arguments)]
pub fn fit_single_spectrum_core(
    spectrum: &[f32],
    dopp_slit: &[f32],
    spec_noise: &[f32],
    guide_velocity: f32,
    velocity_range: f32,
    npix: i32,
    npix_slack: i32,
    dv: f32,
    width_min: f32,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
    config: FitConfig,
) -> FitSingleSpectrumResult {
    let sg_xpixels = spectrum.len();
    let dv_slac = dv * npix_slack as f32;

    let mut imax = 0usize;
    let mut max_val = f32::NEG_INFINITY;

    // A non-finite guide makes every comparison false, so no peak is found
    // and FLAG_NO_LOCAL_MAX is returned, matching the C extension.
    for i in 0..sg_xpixels {
        let vel_abs = (dopp_slit[i] - guide_velocity).abs();
        if vel_abs <= velocity_range + dv_slac {
            let spec_val = spectrum[i];
            if spec_val > max_val {
                max_val = spec_val;
                imax = i;
            }
        }
    }

    // No finite, positive peak in the search window. Covers the all-NaN /
    // empty-window case (where `max_val` stays -inf) and non-positive peaks
    // that would otherwise invert the sign under normalisation by `max_val`.
    if max_val <= 0.0 {
        return result_with_flag(FLAG_NO_LOCAL_MAX, 0, 0);
    }

    // Match the original C extension: the slack-widened window is only used to
    // vet local maxima. A peak found in the slack band is rejected.
    if (dopp_slit[imax] - guide_velocity).abs() > velocity_range {
        return result_with_flag(FLAG_NO_LOCAL_MAX, 0, 0);
    }

    let npix = npix.max(0) as usize;
    let mut i_left = imax.saturating_sub(npix);
    let mut i_right = (imax + npix + 1).min(sg_xpixels);

    // Inherited from the original C extension: when the window is clamped at a
    // spectrum edge, only one extra pixel per side is recovered, so a peak
    // near the edge fits with fewer than `2 * npix + 1` samples. Kept so the
    // window selection stays compatible with the C code near edges.
    if i_right - i_left < (2 * npix + 1) {
        i_left = imax.saturating_sub(npix + 1);
        i_right = (imax + npix + 2).min(sg_xpixels);
    }

    let vel_center = dopp_slit[imax];
    let vel_half_range = dv * npix as f32;

    WINDOW.with_borrow_mut(|window| {
        let [xdata, ydata, edata] = window;
        xdata.clear();
        ydata.clear();
        edata.clear();
        for i in i_left..i_right {
            let x = dopp_slit[i];
            let y = spectrum[i] / max_val;
            let e = spec_noise[i] / max_val;
            // Skip masked / invalid samples: a single non-finite or
            // non-positive noise value would otherwise abort the whole fit.
            if x.is_finite() && y.is_finite() && e.is_finite() && e > 0.0 {
                xdata.push(x);
                ydata.push(y);
                edata.push(e);
            }
        }

        // A peak was found, but masking left too few valid samples to constrain a
        // 3-parameter fit. Reported as FLAG_NO_LOCAL_MAX to match the C extension.
        if xdata.len() < 3 {
            return result_with_flag(FLAG_NO_LOCAL_MAX, i_left as i32, i_right as i32);
        }

        let p0 = [1.0, vel_center, width_guess];
        let bounds = [
            [amplitude_rel_min, amplitude_rel_max],
            [vel_center - vel_half_range, vel_center + vel_half_range],
            [width_min, width_max],
        ];

        let Some(outcome) = fit_gaussian_bounded(xdata, ydata, edata, p0, bounds, config) else {
            return result_with_flag(FLAG_NO_CONVERGENCE, i_left as i32, i_right as i32);
        };

        let dof = (xdata.len() as i32 - 3).max(1) as f32;
        let fit_results = [
            outcome.params[0] * max_val,
            outcome.params[1],
            outcome.params[2],
            outcome.errors[0] * max_val,
            outcome.errors[1],
            outcome.errors[2],
            outcome.bestnorm / dof,
            FLAG_SUCCESS,
        ];

        // Compare errors with the velocity and width spans. The amplitude bounds
        // form a detection window; see the quality-bit rationale in docs/design-notes.rst.
        let mut quality = 0u8;
        if outcome.errors[1] >= 2.0 * vel_half_range || outcome.errors[2] >= (width_max - width_min)
        {
            quality |= QUALITY_UNCONSTRAINED;
        }
        if outcome.errors.contains(&0.0) {
            // This also occurs for exact fits because errors scale with residuals.
            quality |= QUALITY_ZERO_ERROR;
        }
        if outcome
            .params
            .iter()
            .zip(bounds.iter())
            .any(|(param, bound)| *param == bound[0] || *param == bound[1])
        {
            // rmpfit clamps a parameter onto the bound exactly, so equality is the
            // right test here.
            quality |= QUALITY_PEGGED;
        }

        FitSingleSpectrumResult {
            fit_results,
            i_left: i_left as i32,
            i_right: i_right as i32,
            quality,
            n_iter: outcome.n_iter as i32,
            n_fev: outcome.n_fev as i32,
        }
    })
}
