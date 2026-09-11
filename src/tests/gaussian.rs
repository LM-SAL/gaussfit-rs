use crate::gaussian::{fit_gaussian_bounded, FitConfig};

#[test]
fn fits_simple_gaussian() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = [
        (-0.5f32 * 4.0).exp(),
        (-0.5f32).exp(),
        1.0,
        (-0.5f32).exp(),
        (-0.5f32 * 4.0).exp(),
    ];
    let err = [0.1; 5];

    let outcome = fit_gaussian_bounded(
        &x,
        &y,
        &err,
        [0.9, 0.1, 0.8],
        [[0.1, 2.0], [-2.0, 2.0], [0.2, 2.0]],
        FitConfig::default(),
    )
    .expect("fit should converge");

    assert!((outcome.params[0] - 1.0).abs() < 1.0e-3);
    assert!(outcome.params[1].abs() < 1.0e-3);
    assert!((outcome.params[2] - 1.0).abs() < 1.0e-3);
}

#[test]
fn rejects_non_positive_sigma_lower_bound() {
    let x = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let y = [
        (-0.5f32 * 4.0).exp(),
        (-0.5f32).exp(),
        1.0,
        (-0.5f32).exp(),
        (-0.5f32 * 4.0).exp(),
    ];
    let err = [0.1; 5];

    let outcome = fit_gaussian_bounded(
        &x,
        &y,
        &err,
        [0.9, 0.1, 0.8],
        [[0.1, 2.0], [-2.0, 2.0], [0.0, 2.0]],
        FitConfig::default(),
    );

    assert!(outcome.is_none());
}

#[test]
fn bound_pinned_window_matches_c_reference() {
    // Regression guard for the MPFIT bound-snap stall fixed in
    // third_party/rmpfit (VENDORED.md): a bound-limited step landed sigma two
    // ULP above width_min, alpha then collapsed to ~1e-14 and stock rmpfit
    // reported `Par` after 4 iterations at reduced chi-square 56.66. Expected
    // values are MUSE's float32 C extension output for this corpus window
    // (parameter set "wide", seed 1, row 314), normalised by the peak.
    let x = [
        -238.98305, -228.81355, -218.64407, -208.47458, -198.30508, -188.13559, -177.9661,
        -167.79662, -157.62712, -147.45763, -137.28813,
    ];
    let y = [
        -0.02878176,
        0.0066301413,
        -0.02121532,
        -0.013344752,
        0.033882372,
        1.0,
        0.78892606,
        -0.051678322,
        -0.017880365,
        -0.007661589,
        0.011638442,
    ];
    let error = [0.03133689; 11];

    let outcome = fit_gaussian_bounded(
        &x,
        &y,
        &error,
        [1.0, -188.13559, 30.0],
        [[0.1, 2.0], [-238.98305, -137.28813], [5.0, 200.0]],
        FitConfig::default(),
    )
    .expect("fit should converge");

    assert!(
        (outcome.params[0] - 1.499_192_6).abs() < 1.0e-4,
        "amplitude {}",
        outcome.params[0]
    );
    assert!(
        (outcome.params[1] + 183.642_43).abs() < 1.0e-2,
        "velocity {}",
        outcome.params[1]
    );
    assert_eq!(outcome.params[2], 5.0, "sigma pinned at the lower bound");
    let reduced_chi2 = outcome.bestnorm / 8.0;
    assert!(
        (reduced_chi2 - 0.762_82).abs() < 1.0e-4,
        "reduced chi2 {reduced_chi2}"
    );
}
