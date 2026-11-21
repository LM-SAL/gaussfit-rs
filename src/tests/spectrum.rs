use crate::gaussian::FitConfig;
use crate::spectrum::{fit_single_spectrum_core, FitSingleSpectrumResult};
use crate::{
    FLAG_NO_LOCAL_MAX, FLAG_SUCCESS, QUALITY_PEGGED, QUALITY_UNCONSTRAINED, QUALITY_ZERO_ERROR,
};

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
fn large_window_still_fits() {
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
    // The 20 km/s line falls inside the 34 km/s search window but outside the 10 km/s limit.
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
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);
    let mut noise = vec![0.01; v.len()];
    noise[30] = f32::NAN; // near the central peak, inside the fit window

    let result = fit_clean_spectrum_with_noise(&v, &spectrum, &noise);

    assert_eq!(result.fit_results[7], FLAG_SUCCESS);
    assert!(result.fit_results[1].abs() < 0.5);
}

#[test]
fn sparse_window_reports_no_local_max() {
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE);
    let noise = vec![f32::NAN; v.len()];

    let result = fit_clean_spectrum_with_noise(&v, &spectrum, &noise);

    assert_eq!(result.fit_results[7], FLAG_NO_LOCAL_MAX);
}

#[test]
fn negative_peak_returns_no_local_max() {
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let spectrum = gaussian_spectrum(&v, -1.0, 0.0, SIGMA_TRUE);
    let noise = vec![0.01; v.len()];

    let result = fit_clean_spectrum_with_noise(&v, &spectrum, &noise);

    assert_eq!(result.fit_results[7], FLAG_NO_LOCAL_MAX);
}

#[test]
fn quality_bits_separate_constrained_unconstrained_zero_and_pegged_fits() {
    let v = velocity_grid(60, -300.0, 600.0 / 59.0);
    let clean = fit_clean_spectrum_with_noise(
        &v,
        &gaussian_spectrum(&v, 1.0, 0.0, SIGMA_TRUE),
        &vec![0.05; v.len()],
    );
    assert_eq!(clean.fit_results[7], FLAG_SUCCESS);
    assert_eq!(clean.quality, 0);

    // Alternating noise gives a converged fit whose errors exceed both allowed spans.
    let checker: Vec<f32> = (0..v.len())
        .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
        .collect();
    let faint: Vec<f32> = v
        .iter()
        .zip(&checker)
        .map(|(&x, &offset)| 0.5 * (-0.5 * (x / SIGMA_TRUE).powi(2)).exp() + offset)
        .collect();
    let unconstrained = fit_clean_spectrum_with_noise(&v, &faint, &vec![3.0; v.len()]);
    assert_eq!(unconstrained.fit_results[7], FLAG_SUCCESS);
    assert!(unconstrained.fit_results[4] >= 2.0 * 12.0 * 10.0);
    assert!(unconstrained.fit_results[5] >= 100.0 - 5.0);
    assert_ne!(unconstrained.quality & QUALITY_UNCONSTRAINED, 0);
    assert_ne!(unconstrained.quality & QUALITY_PEGGED, 0);

    // The faint line triggers the covariance determinant guard, yielding zero errors.
    let tiny = gaussian_spectrum(&v, 1.0e-8, 0.0, SIGMA_TRUE);
    let guarded = fit_clean_spectrum_with_noise(&v, &tiny, &vec![1.0; v.len()]);
    assert_eq!(guarded.fit_results[7], FLAG_SUCCESS);
    assert_eq!(&guarded.fit_results[3..6], &[0.0; 3]);
    assert_ne!(guarded.quality & QUALITY_ZERO_ERROR, 0);
    assert_eq!(guarded.quality & QUALITY_UNCONSTRAINED, 0); // zero errors stay under any span

    // A broad line legitimately saturates the upper width bound.
    let broad = fit_clean_spectrum_with_noise(
        &v,
        &gaussian_spectrum(&v, 1.0, 0.0, 300.0),
        &vec![0.05; v.len()],
    );
    assert_eq!(broad.fit_results[7], FLAG_SUCCESS);
    assert_eq!(broad.fit_results[2], 100.0);
    assert_ne!(broad.quality & QUALITY_PEGGED, 0);

    let failed = fit_clean_spectrum_with_noise(&v, &vec![-1.0; v.len()], &vec![0.05; v.len()]);
    assert_eq!(failed.fit_results[7], FLAG_NO_LOCAL_MAX);
    assert_eq!(failed.quality, 0);
}
