use std::cell::RefCell;

use rmpfit::{MPConfig, MPFitter, MPPar, MPResult, MPSide, MPSuccess, MPWorkspace};

use crate::{FTOL, GTOL, MAX_ITER, XTOL};

thread_local! {
    // One solver workspace per thread (Rayon workers included): after the
    // first fit on a thread, a fit makes no heap allocations.
    static WORKSPACE: RefCell<MPWorkspace> = RefCell::new(MPWorkspace::default());
}

/// Convergence tolerances and iteration limit for the LM solver.
#[derive(Clone, Copy, Debug)]
pub struct FitConfig {
    /// Parameter step size tolerance.
    pub xtol: f32,
    /// Cost-function change tolerance.
    pub ftol: f32,
    /// Gradient norm tolerance.
    pub gtol: f32,
    /// Maximum outer LM iterations.
    pub max_iter: usize,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            xtol: XTOL,
            ftol: FTOL,
            gtol: GTOL,
            max_iter: MAX_ITER,
        }
    }
}

/// Fitted Gaussian parameters returned by the LM solver.
#[derive(Clone, Copy, Debug)]
pub struct FitOutcome {
    /// `[amplitude, mean, sigma]`
    pub(crate) params: [f32; 3],
    /// 1-sigma parameter errors `[amp_err, mean_err, sigma_err]`
    pub(crate) errors: [f32; 3],
    /// Sum of squared weighted residuals at the best-fit parameters.
    pub(crate) bestnorm: f32,
    /// Accepted Levenberg-Marquardt iterations.
    pub(crate) n_iter: usize,
    /// Model evaluations, Jacobian calls included.
    pub(crate) n_fev: usize,
}

/// Weighted-residual problem handed to the MPFIT (rmpfit) solver.
///
/// Residuals are `(y - amp * exp(-0.5 * ((x - mean) / sigma)^2)) / error`,
/// matching the MPFIT convention `(y - f(x)) / y_err`. Inputs are `f32`; all
/// arithmetic is done in `f64`.
struct GaussianProblem<'a> {
    x: &'a [f32],
    y: &'a [f32],
    error: &'a [f32],
    params: [MPPar; 3],
    config: MPConfig,
}

impl MPFitter for GaussianProblem<'_> {
    fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
        let amp = params[0];
        let mean = params[1];
        let sigma = params[2];
        for (i, deviate) in deviates.iter_mut().enumerate() {
            let z = (f64::from(self.x[i]) - mean) / sigma;
            let model = amp * (-0.5 * z * z).exp();
            *deviate = (f64::from(self.y[i]) - model) / f64::from(self.error[i]);
        }
        Ok(())
    }

    fn number_of_points(&self) -> usize {
        self.x.len()
    }

    fn config(&self) -> &MPConfig {
        &self.config
    }

    fn parameters(&self) -> &[MPPar] {
        &self.params
    }

    fn jacobian(
        &mut self,
        params: &[f64],
        deviates: &mut [f64],
        derivs: &mut [Option<Vec<f64>>],
    ) -> MPResult<()> {
        let amp = params[0];
        let mean = params[1];
        let sigma = params[2];
        for (i, deviate) in deviates.iter_mut().enumerate() {
            let inverse_error = 1.0 / f64::from(self.error[i]);
            let z = (f64::from(self.x[i]) - mean) / sigma;
            let expterm = (-0.5 * z * z).exp();
            *deviate = (f64::from(self.y[i]) - amp * expterm) * inverse_error;
            if let Some(column) = derivs[0].as_mut() {
                column[i] = -expterm * inverse_error;
            }
            if let Some(column) = derivs[1].as_mut() {
                column[i] = -amp * z * expterm * inverse_error / sigma;
            }
            if let Some(column) = derivs[2].as_mut() {
                column[i] = -amp * z * z * expterm * inverse_error / sigma;
            }
        }
        Ok(())
    }
}

/// Box-constrained Levenberg-Marquardt fit of a 3-parameter Gaussian.
///
/// `bounds[i] = [lower, upper]` for parameter `i` (amp, mean, sigma). Returns
/// `None` if the solver did not reach a convergence criterion (e.g. it hit the
/// iteration limit) or the inputs are invalid.
///
/// The fit is delegated to [`rmpfit`], a pure-Rust port of the CMPFIT/MINPACK
/// `mpfit` routine, so the convergence semantics match the original C
/// extension. The vendored copy in `third_party/rmpfit` carries local
/// patches (a bound-snap fix, the MINPACK `lmpar` clamp, and a reusable
/// per-thread workspace so a fit allocates nothing); see
/// `third_party/rmpfit/VENDORED.md`. Parameter errors use the full
/// three-parameter Hessian, including parameters at their bounds, matching
/// the C SciPy-style covariance calculation.
pub fn fit_gaussian_bounded(
    x: &[f32],
    y: &[f32],
    error: &[f32],
    initial: [f32; 3],
    bounds: [[f32; 2]; 3],
    config: FitConfig,
) -> Option<FitOutcome> {
    if x.len() != y.len() || x.len() != error.len() || x.len() < 3 {
        return None;
    }
    if !config.xtol.is_finite()
        || !config.ftol.is_finite()
        || !config.gtol.is_finite()
        || config.xtol <= 0.0
        || config.ftol <= 0.0
        || config.gtol <= 0.0
        || config.max_iter == 0
    {
        return None;
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return None;
    }
    for bound in bounds {
        if !bound[0].is_finite() || !bound[1].is_finite() || bound[0] >= bound[1] {
            return None;
        }
    }
    // The solver may evaluate the model anywhere inside the box, and `eval`
    // divides by sigma, so sigma = 0 must be unreachable.
    if bounds[2][0] <= 0.0 {
        return None;
    }

    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if error.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return None;
    }

    let limits = bounds.map(|bound| [f64::from(bound[0]), f64::from(bound[1])]);
    // Clamp the start point into the box; rmpfit rejects out-of-bounds starts
    // with MPError::InitBounds.
    let mut params = [
        clamp_f64(f64::from(initial[0]), limits[0]),
        clamp_f64(f64::from(initial[1]), limits[1]),
        clamp_f64(f64::from(initial[2]), limits[2]),
    ];

    let mp_par = |limit: [f64; 2]| MPPar {
        limited_low: true,
        limited_up: true,
        limit_low: limit[0],
        limit_up: limit[1],
        side: MPSide::User,
        ..MPPar::new()
    };
    let mp_config = MPConfig {
        ftol: f64::from(config.ftol),
        xtol: f64::from(config.xtol),
        gtol: f64::from(config.gtol),
        max_iter: config.max_iter,
        ..MPConfig::new()
    };

    let mut problem = GaussianProblem {
        x,
        y,
        error,
        params: [mp_par(limits[0]), mp_par(limits[1]), mp_par(limits[2])],
        config: mp_config,
    };

    let status = WORKSPACE
        .with_borrow_mut(|workspace| problem.mpfit_with_workspace(&mut params, workspace))
        .ok()?;

    // Chi/Par/Both/Dir are normal convergence; Ftol/Xtol/Gtol mean the solver
    // reached a rounding-limited minimum it cannot improve (still a usable fit).
    // MaxIter (and NotDone) are treated as non-convergence, matching the old
    // "ran out of iterations" -> FLAG_NO_CONVERGENCE behaviour.
    let converged = matches!(
        status.success,
        MPSuccess::Chi
            | MPSuccess::Par
            | MPSuccess::Both
            | MPSuccess::Dir
            | MPSuccess::Ftol
            | MPSuccess::Xtol
            | MPSuccess::Gtol
    );
    if !converged {
        return None;
    }

    let bestnorm = status.best_norm;
    let errors = scipy_style_errors(x, error, params, bestnorm);

    Some(FitOutcome {
        params: params.map(|value| value as f32),
        errors,
        bestnorm: bestnorm as f32,
        n_iter: status.n_iter,
        n_fev: status.n_fev,
    })
}

/// Return SciPy-style errors from the inverse full Hessian, including any
/// parameters pegged at their bounds. This mirrors `mp_xerror_scipy` in the C
/// backend rather than rmpfit's reduced covariance for free parameters only.
fn scipy_style_errors(x: &[f32], error: &[f32], params: [f64; 3], bestnorm: f64) -> [f32; 3] {
    let [amplitude, mean, sigma] = params;
    let mut hessian = [[0.0f64; 3]; 3];

    for (x_value, error_value) in x.iter().zip(error) {
        let inverse_error = 1.0 / f64::from(*error_value);
        let z = (f64::from(*x_value) - mean) / sigma;
        let expterm = (-0.5 * z * z).exp();
        let derivatives = [
            -expterm * inverse_error,
            -amplitude * z * expterm * inverse_error / sigma,
            -amplitude * z * z * expterm * inverse_error / sigma,
        ];
        for row in 0..3 {
            for column in row..3 {
                hessian[row][column] += derivatives[row] * derivatives[column];
            }
        }
    }

    let a = hessian[0][0];
    let b = hessian[0][1];
    let c = hessian[0][2];
    let d = hessian[1][1];
    let e = hessian[1][2];
    let f = hessian[2][2];
    let cofactors = [d * f - e * e, a * f - c * c, a * d - b * b];
    let cofactor_01 = -(b * f - e * c);
    let cofactor_02 = b * e - d * c;
    let determinant = a * cofactors[0] + b * cofactor_01 + c * cofactor_02;
    if !determinant.is_finite() || determinant.abs() < 1.0e-30 {
        return [0.0; 3];
    }

    let scale = if x.len() > 3 {
        (bestnorm / (x.len() - 3) as f64).sqrt()
    } else {
        1.0
    };
    cofactors.map(|cofactor| {
        let variance = cofactor / determinant;
        if variance > 0.0 {
            (variance.sqrt() * scale) as f32
        } else {
            0.0
        }
    })
}

fn clamp_f64(value: f64, limit: [f64; 2]) -> f64 {
    value.max(limit[0]).min(limit[1])
}

#[cfg(test)]
mod derivative_tests {
    use super::*;

    #[test]
    fn analytical_jacobian_matches_central_differences() {
        let mut problem = GaussianProblem {
            x: &[-1.0, 0.5, 2.0],
            y: &[0.2, 1.1, 0.4],
            error: &[0.1, 0.2, 0.3],
            params: std::array::from_fn(|_| MPPar::new()),
            config: MPConfig::new(),
        };
        let params = [1.2, 0.3, 0.8];
        let mut residuals = [0.0; 3];
        let mut derivatives = vec![Some(vec![0.0; 3]); 3];
        problem
            .jacobian(&params, &mut residuals, &mut derivatives)
            .unwrap();

        let step = 1.0e-6;
        for parameter in 0..3 {
            let mut left = params;
            let mut right = params;
            left[parameter] -= step;
            right[parameter] += step;
            let mut left_residuals = [0.0; 3];
            let mut right_residuals = [0.0; 3];
            problem.eval(&left, &mut left_residuals).unwrap();
            problem.eval(&right, &mut right_residuals).unwrap();

            for sample in 0..3 {
                let numerical = (right_residuals[sample] - left_residuals[sample]) / (2.0 * step);
                let analytical = derivatives[parameter].as_ref().unwrap()[sample];
                assert!((analytical - numerical).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn full_hessian_errors_include_parameter_at_bound() {
        let x = [-81.215_41, -40.607_704, 0.0, 40.607_704, 81.215_41];
        let y = [
            0.899_082_66,
            0.949_152_9,
            0.999_999_8,
            0.988_445_2,
            0.952_372_6,
        ];
        let error = [1.0; 5];
        let outcome = fit_gaussian_bounded(
            &x,
            &y,
            &error,
            [1.0, 0.0, 69.137_89],
            [[0.9, 1.1], [-81.215_41, 81.215_41], [59.137_894, 200.0]],
            FitConfig::default(),
        )
        .unwrap();

        // Reference values from MUSE's float32 C extension. In particular,
        // sigma is at its upper bound but must retain a non-zero uncertainty.
        let expected = [0.008_243_047, 4.667_220_6, 16.366_814];
        for (actual, expected) in outcome.errors.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= expected * 2.0e-3);
        }
    }
}
