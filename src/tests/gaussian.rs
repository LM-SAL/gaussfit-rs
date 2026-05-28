use crate::gaussian::{fit_gaussian_bounded_with_config, FitConfig};

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

    let outcome = fit_gaussian_bounded_with_config(
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
fn fits_simple_gaussian_f64() {
    let x: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let y: [f64; 5] = [
        (-0.5_f64 * 4.0).exp(),
        (-0.5_f64).exp(),
        1.0,
        (-0.5_f64).exp(),
        (-0.5_f64 * 4.0).exp(),
    ];
    let err = [0.1_f64; 5];

    let outcome = fit_gaussian_bounded_with_config::<f64>(
        &x,
        &y,
        &err,
        [0.9, 0.1, 0.8],
        [[0.1, 2.0], [-2.0, 2.0], [0.2, 2.0]],
        FitConfig::<f64>::default(),
    )
    .expect("f64 fit should converge");

    assert!((outcome.params[0] - 1.0).abs() < 1.0e-6);
    assert!(outcome.params[1].abs() < 1.0e-6);
    assert!((outcome.params[2] - 1.0).abs() < 1.0e-6);
}
