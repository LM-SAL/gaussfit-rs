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

#[derive(Clone, Copy, Debug)]
pub struct FitSingleSpectrumResult {
    /// `[amp, vel, sigma, amp_err, vel_err, sig_err, reduced_chi2, flag]`
    pub(crate) fit_results: [f32; 8],
    pub(crate) i_left: i32,
    pub(crate) i_right: i32,
    /// Quality bitmask for successful fits; zero for failures. The ninth result column.
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

    // Normalising by a negative peak would reverse the sign of the spectrum.
    if max_val <= 0.0 {
        return result_with_flag(FLAG_NO_LOCAL_MAX, 0, 0);
    }

    // A brighter peak in the slack band disqualifies the in-range peak.
    if (dopp_slit[imax] - guide_velocity).abs() > velocity_range {
        return result_with_flag(FLAG_NO_LOCAL_MAX, 0, 0);
    }

    let npix = npix.max(0) as usize;
    let mut i_left = imax.saturating_sub(npix);
    let mut i_right = (imax + npix + 1).min(sg_xpixels);

    // Match the C fitter at an edge by recovering at most one extra pixel per side.
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
            if x.is_finite() && y.is_finite() && e.is_finite() && e > 0.0 {
                xdata.push(x);
                ydata.push(y);
                edata.push(e);
            }
        }

        // The C fitter uses this status when too few samples survive masking.
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

        // The relative amplitude bounds are a detection window, so only velocity and width count.
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
            // rmpfit clamps to the exact bound.
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
