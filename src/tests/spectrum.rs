use crate::gaussian::FitConfig;
use crate::spectrum::{fit_single_spectrum_core, FitSingleSpectrumResult};
use crate::{FLAG_NO_CONVERGENCE, FLAG_NO_LOCAL_MAX, FLAG_SUCCESS};

const SIGMA_TRUE: f32 = 30.0;

fn velocity_grid(n: usize, start: f32, step: f32) -> Vec<f32> {
    (0..n).map(|i| start + step * i as f32).collect()
}

fn gaussian_spectrum(v: &[f32], amp: f32, vel: f32, sigma: f32) -> Vec<f32> {
    v.iter()
        .map(|&x| {
            let z = (x - vel) / sigma;
            amp * (-0.5 * z * z).exp()
        })
        .collect()
}

fn fit_clean_spectrum(v: &[f32], spectrum: &[f32]) -> FitSingleSpectrumResult {
    let noise = vec![0.01; v.len()];
    fit_clean_spectrum_with_noise(v, spectrum, &noise)
}

fn fit_clean_spectrum_with_noise(
    v: &[f32],
    spectrum: &[f32],
    noise: &[f32],
) -> FitSingleSpectrumResult {
    fit_single_spectrum_core(
        spectrum,
        v,
        noise,
        0.0,
        200.0,
        10,
        2,
        12.0,
        5.0,
        0.1,
        2.0,
        100.0,
        30.0,
        FitConfig::default(),
    )
}

#[test]
fn recovers_centered_clean_gaussian() {
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);

    let result = fit_clean_spectrum(&v, &spectrum);

    assert_eq!(result.fit_results[7], FLAG_SUCCESS);
    assert!(result.fit_results[1].abs() < 0.1);
    assert!((result.fit_results[2] - SIGMA_TRUE).abs() < 0.1);
    assert!(result.i_left < result.i_right);
}

#[test]
fn recovers_shifted_clean_gaussian() {
    let v = velocity_grid(80, -300.0, 600.0 / 79.0);
    let spectrum = gaussian_spectrum(&v, 2.5, 48.0, 22.0);
    let noise = vec![0.02; v.len()];

    let result = fit_single_spectrum_core(
        &spectrum,
        &v,
        &noise,
        50.0,
        120.0,
        12,
        2,
        600.0 / 79.0,
        5.0,
        0.1,
        2.0,
        80.0,
        25.0,
        FitConfig::default(),
    );

    assert_eq!(result.fit_results[7], FLAG_SUCCESS);
    assert!((result.fit_results[0] - 2.5).abs() < 0.01);
    assert!((result.fit_results[1] - 48.0).abs() < 0.1);
    assert!((result.fit_results[2] - 22.0).abs() < 0.1);
}

#[test]
fn returns_no_local_max_when_no_sample_in_search_window() {
    // guide_velocity is placed far beyond the grid so the search window is
    // empty (max_val stays -inf).
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);
    let noise = vec![0.02; v.len()];

    let result = fit_single_spectrum_core(
        &spectrum,
        &v,
        &noise,
        5000.0,
        5.0,
        10,
        2,
        12.0,
        5.0,
        0.1,
        2.0,
        100.0,
        30.0,
        FitConfig::default(),
    );

    assert_eq!(result.fit_results[7], FLAG_NO_LOCAL_MAX);
    assert!(result.fit_results[..7].iter().all(|value| value.is_nan()));
}

#[test]
fn large_window_uses_heap_fallback_and_still_fits() {
    let v = velocity_grid(700, -350.0, 1.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);
    let noise = vec![0.01; v.len()];

    let result = fit_single_spectrum_core(
        &spectrum,
        &v,
        &noise,
        0.0,
        50.0,
        300,
        0,
        1.0,
        5.0,
        0.1,
        2.0,
        100.0,
        30.0,
        FitConfig::default(),
    );

    assert!(result.i_right - result.i_left > 512);
    assert_eq!(result.fit_results[7], FLAG_SUCCESS);
    assert!(result.fit_results[1].abs() < 0.1);
    assert!((result.fit_results[2] - SIGMA_TRUE).abs() < 0.1);
}

#[test]
fn rejects_peak_inside_slack_zone() {
    // Match the original C extension: dv_slac =
    // dv * npix_slack = 12 * 2 = 24, so the search window reaches |v| <= 34.
    // The peak at vel = 20 is outside the strict velocity_range (10) but inside
    // the slack zone, and must be rejected as NO_LOCAL_MAX.
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 20.0, SIGMA_TRUE);
    let noise = vec![0.01; v.len()];

    let result = fit_single_spectrum_core(
        &spectrum,
        &v,
        &noise,
        0.0,
        10.0,
        10,
        2,
        12.0,
        5.0,
        0.1,
        2.0,
        100.0,
        30.0,
        FitConfig::default(),
    );

    assert_eq!(result.fit_results[7], FLAG_NO_LOCAL_MAX);
    assert!(result.fit_results[..7].iter().all(|value| value.is_nan()));
}

#[test]
fn skips_masked_noise_pixel_instead_of_failing() {
    // A single non-finite noise sample inside the window must be dropped, not
    // abort the whole fit (previously yielded FLAG_NO_CONVERGENCE).
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);
    let mut noise = vec![0.01; v.len()];
    noise[30] = f32::NAN; // near the central peak, inside the fit window

    let result = fit_clean_spectrum_with_noise(&v, &spectrum, &noise);

    assert_eq!(result.fit_results[7], FLAG_SUCCESS);
    assert!(result.fit_results[1].abs() < 0.5);
}

#[test]
fn sparse_window_reports_no_convergence() {
    // A peak exists, but masking leaves fewer than 3 valid samples, so the
    // failure is a fit failure (NO_CONVERGENCE), not a missing peak.
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);
    let noise = vec![f32::NAN; v.len()];

    let result = fit_clean_spectrum_with_noise(&v, &spectrum, &noise);

    assert_eq!(result.fit_results[7], FLAG_NO_CONVERGENCE);
}

#[test]
fn negative_peak_returns_no_local_max() {
    // An all-negative window has max_val <= 0 and must be rejected as
    // NO_LOCAL_MAX rather than sign-flipping under normalisation.
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, -1.0, 0.0, SIGMA_TRUE);
    let noise = vec![0.01; v.len()];

    let result = fit_clean_spectrum_with_noise(&v, &spectrum, &noise);

    assert_eq!(result.fit_results[7], FLAG_NO_LOCAL_MAX);
}
