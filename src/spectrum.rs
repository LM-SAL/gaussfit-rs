use crate::gaussian::{fit_gaussian_bounded_with_config, FitConfig};
use crate::{FLAG_NO_CONVERGENCE, FLAG_NO_LOCAL_MAX, FLAG_SUCCESS};

/// Internal result returned by `fit_single_spectrum_core`.
#[derive(Clone, Copy, Debug)]
pub struct FitSingleSpectrumResult {
    /// `[amp, vel, sigma, amp_err, vel_err, sig_err, reduced_chi2, flag]`
    pub(crate) fit_results: [f32; 8],
    pub(crate) i_left: i32,
    pub(crate) i_right: i32,
    /// Negative on internal error; zero on success or expected failure.
    pub(crate) status: i32,
}

fn result_with_flag(flag: f32, i_left: i32, i_right: i32, status: i32) -> FitSingleSpectrumResult {
    let mut fit_results = [f32::NAN; 8];
    fit_results[7] = flag;
    FitSingleSpectrumResult {
        fit_results,
        i_left,
        i_right,
        status,
    }
}

/// Core fitting logic shared by both the single-spectrum and batch entry points.
///
/// `spectrum`, `dopp_slit`, and `spec_noise` must all have the same length
/// (`sg_xpixels`). The function normalises by the detected peak before fitting
/// and rescales the output parameters back to original units.
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

    // No finite, positive peak in the search window. Covers the all-NaN /
    // empty-window case (where `max_val` stays -inf) and non-positive peaks
    // that would otherwise invert the sign under normalisation by `max_val`.
    if max_val <= 0.0 {
        return result_with_flag(FLAG_NO_LOCAL_MAX, 0, 0, 0);
    }

    // Match the original C extension: the slack-widened window is only used to
    // vet local maxima. A peak found in the slack band is rejected.
    if (dopp_slit[imax] - guide_velocity).abs() > velocity_range {
        return result_with_flag(FLAG_NO_LOCAL_MAX, 0, 0, 0);
    }

    let npix = npix.max(0) as usize;
    let mut i_left = imax.saturating_sub(npix);
    let mut i_right = (imax + npix + 1).min(sg_xpixels);

    if i_right - i_left < (2 * npix + 1) {
        i_left = imax.saturating_sub(npix + 1);
        i_right = (imax + npix + 2).min(sg_xpixels);
    }

    // Common windows stay on the stack; unusually wide fits fall back to heap
    // buffers rather than reporting a misleading no-peak status.
    const MAX_STACK_WINDOW: usize = 512;
    let window_len = i_right - i_left;
    let vel_center = dopp_slit[imax];
    let vel_half_range = dv * npix as f32;

    if window_len <= MAX_STACK_WINDOW {
        let mut xbuf = [0.0f32; MAX_STACK_WINDOW];
        let mut ybuf = [0.0f32; MAX_STACK_WINDOW];
        let mut ebuf = [0.0f32; MAX_STACK_WINDOW];
        let mut count = 0usize;

        for i in i_left..i_right {
            let x = dopp_slit[i];
            let y = spectrum[i] / max_val;
            let e = spec_noise[i] / max_val;
            // Skip masked / invalid samples: a single non-finite or
            // non-positive noise value would otherwise abort the whole fit.
            if x.is_finite() && y.is_finite() && e.is_finite() && e > 0.0 {
                xbuf[count] = x;
                ybuf[count] = y;
                ebuf[count] = e;
                count += 1;
            }
        }

        return fit_prepared_spectrum_window(
            &xbuf[..count],
            &ybuf[..count],
            &ebuf[..count],
            max_val,
            i_left,
            i_right,
            vel_center,
            vel_half_range,
            width_min,
            amplitude_rel_min,
            amplitude_rel_max,
            width_max,
            width_guess,
            config,
        );
    }

    let mut xdata = Vec::with_capacity(window_len);
    let mut ydata = Vec::with_capacity(window_len);
    let mut edata = Vec::with_capacity(window_len);
    for i in i_left..i_right {
        let x = dopp_slit[i];
        let y = spectrum[i] / max_val;
        let e = spec_noise[i] / max_val;
        // Skip masked / invalid samples: a single non-finite or non-positive
        // noise value would otherwise abort the whole fit.
        if x.is_finite() && y.is_finite() && e.is_finite() && e > 0.0 {
            xdata.push(x);
            ydata.push(y);
            edata.push(e);
        }
    }

    fit_prepared_spectrum_window(
        &xdata,
        &ydata,
        &edata,
        max_val,
        i_left,
        i_right,
        vel_center,
        vel_half_range,
        width_min,
        amplitude_rel_min,
        amplitude_rel_max,
        width_max,
        width_guess,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn fit_prepared_spectrum_window(
    xdata: &[f32],
    ydata: &[f32],
    edata: &[f32],
    max_val: f32,
    i_left: usize,
    i_right: usize,
    vel_center: f32,
    vel_half_range: f32,
    width_min: f32,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
    config: FitConfig,
) -> FitSingleSpectrumResult {
    if xdata.len() < 3 {
        return result_with_flag(FLAG_NO_LOCAL_MAX, i_left as i32, i_right as i32, 0);
    }

    let p0 = [1.0, vel_center, width_guess];
    let bounds = [
        [amplitude_rel_min, amplitude_rel_max],
        [vel_center - vel_half_range, vel_center + vel_half_range],
        [width_min, width_max],
    ];

    let Some(outcome) = fit_gaussian_bounded_with_config(xdata, ydata, edata, p0, bounds, config)
    else {
        return result_with_flag(FLAG_NO_CONVERGENCE, i_left as i32, i_right as i32, 0);
    };

    let dof = (xdata.len() as i32 - 3).max(1) as f32;
    let mut fit_results = [0.0f32; 8];
    fit_results[0] = outcome.params[0] * max_val;
    fit_results[1] = outcome.params[1];
    fit_results[2] = outcome.params[2];
    fit_results[3] = outcome.errors[0] * max_val;
    fit_results[4] = outcome.errors[1];
    fit_results[5] = outcome.errors[2];
    fit_results[6] = outcome.bestnorm / dof;
    fit_results[7] = FLAG_SUCCESS;

    FitSingleSpectrumResult {
        fit_results,
        i_left: i_left as i32,
        i_right: i_right as i32,
        status: 0,
    }
}
