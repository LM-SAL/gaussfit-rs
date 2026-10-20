//! Pinned single-thread microbenchmark of the per-spectrum fit, run on demand:
//!
//! ```bash
//! taskset -c 2 cargo test --release bench_fit_window -- --ignored --nocapture
//! ```
//!
//! Same synthetic data as `benchmarks/throughput.py`, without Python, NumPy or Rayon
//! in the timed region. Prints the best-of-five nanoseconds per fit for 5-, 9- and
//! 21-sample windows. `BENCH_NPIX=2,10` selects the windows and `BENCH_ROUNDS=0`
//! skips the fits, which under `perf stat -e instructions:u` isolates their cost from
//! the data generation.

use std::time::Instant;

use crate::gaussian::FitConfig;
use crate::spectrum::fit_single_spectrum_core;

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    fn normal(&mut self) -> f32 {
        let u = self.uniform().max(1.0e-7);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

#[test]
#[ignore]
fn bench_fit_window() {
    let n_spectra = 20_000;
    let env = |name: &str| std::env::var(name).ok();
    let rounds: usize = env("BENCH_ROUNDS").map_or(5, |v| v.parse().unwrap());
    let npix_list: Vec<i32> = env("BENCH_NPIX").map_or(vec![2, 4, 10], |v| {
        v.split(',').map(|s| s.parse().unwrap()).collect()
    });
    for npix in npix_list {
        let window = 2 * npix + 1;
        let pixels = 60usize.max(4 * window as usize + 1);
        // 12 km/s pixels centred on zero, as benchmarks/throughput.py builds them.
        let velocity: Vec<f32> = (0..pixels)
            .map(|i| (i as f32 - (pixels - 1) as f32 / 2.0) * 12.0)
            .collect();
        let mut rng = Lcg(123);
        let spectra: Vec<Vec<f32>> = (0..n_spectra)
            .map(|_| {
                let amp = 0.2 + 1.8 * rng.uniform();
                let mean = -80.0 + 160.0 * rng.uniform();
                let sigma = 10.0 + 50.0 * rng.uniform();
                velocity
                    .iter()
                    .map(|v| {
                        amp * (-0.5 * ((v - mean) / sigma).powi(2)).exp() + 0.05 * rng.normal()
                    })
                    .collect()
            })
            .collect();
        let noise = vec![0.05f32; pixels];
        let mut best = f64::MAX;
        let mut checksum = 0.0f64;
        for _ in 0..rounds {
            let start = Instant::now();
            for spectrum in &spectra {
                let result = fit_single_spectrum_core(
                    spectrum,
                    &velocity,
                    &noise,
                    0.0,
                    200.0,
                    npix,
                    2,
                    12.0,
                    5.0,
                    0.1,
                    2.0,
                    100.0,
                    30.0,
                    FitConfig::default(),
                );
                checksum += f64::from(result.fit_results[0]);
            }
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!(
            "window={window:2} ns_per_fit={:.1} checksum={checksum:.6}",
            1e9 * best / n_spectra as f64
        );
    }
}
