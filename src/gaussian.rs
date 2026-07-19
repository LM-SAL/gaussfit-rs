use num_traits::Float;
use rmpfit::{MPConfig, MPFitter, MPPar, MPResult, MPSide, MPSuccess};

use crate::{FTOL, GTOL, MAX_ITER, XTOL};

/// Convergence tolerances and iteration limit for the LM solver.
#[derive(Clone, Copy, Debug)]
pub struct FitConfig<F: Float = f32> {
    /// Parameter step size tolerance.
    pub xtol: F,
    /// Cost-function change tolerance.
    pub ftol: F,
    /// Gradient norm tolerance.
    pub gtol: F,
    /// Maximum outer LM iterations.
    pub max_iter: usize,
}

impl Default for FitConfig<f32> {
    fn default() -> Self {
        Self {
            xtol: XTOL,
            ftol: FTOL,
            gtol: GTOL,
            max_iter: MAX_ITER,
        }
    }
}

// Tighter than the f32 default to exploit the extra precision; matches the
// defaults of the Python `fit_gaussian_f64` wrapper.
impl Default for FitConfig<f64> {
    fn default() -> Self {
        Self {
            xtol: 1.0e-10,
            ftol: 1.0e-10,
            gtol: 1.0e-10,
            max_iter: MAX_ITER,
        }
    }
}

/// Fitted Gaussian parameters returned by the LM solver.
#[derive(Clone, Copy, Debug)]
pub struct FitOutcome<F: Float = f32> {
    /// `[amplitude, mean, sigma]`
    pub(crate) params: [F; 3],
    /// 1-sigma parameter errors `[amp_err, mean_err, sigma_err]`
    pub(crate) errors: [F; 3],
    /// Sum of squared weighted residuals at the best-fit parameters.
    pub(crate) bestnorm: F,
}

/// Weighted-residual problem handed to the MPFIT (rmpfit) solver.
///
/// Residuals are `(y - amp * exp(-0.5 * ((x - mean) / sigma)^2)) / error`,
/// matching the MPFIT convention `(y - f(x)) / y_err`. All arithmetic is done
/// in `f64`; callers using `f32` convert in and out.
struct GaussianProblem<'a, F: Float> {
    x: &'a [F],
    y: &'a [F],
    error: &'a [F],
    params: [MPPar; 3],
    config: MPConfig,
}

impl<F: Float> MPFitter for GaussianProblem<'_, F> {
    fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
        let amp = params[0];
        let mean = params[1];
        let sigma = params[2];
        for (i, deviate) in deviates.iter_mut().enumerate() {
            let z = (to_f64(&self.x[i]) - mean) / sigma;
            let model = amp * (-0.5 * z * z).exp();
            *deviate = (to_f64(&self.y[i]) - model) / to_f64(&self.error[i]);
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
            let inverse_error = 1.0 / to_f64(&self.error[i]);
            let z = (to_f64(&self.x[i]) - mean) / sigma;
            let expterm = (-0.5 * z * z).exp();
            *deviate = (to_f64(&self.y[i]) - amp * expterm) * inverse_error;
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
/// `mpfit` routine, so the convergence and bound-handling semantics match the
/// original C extension. Parameter errors use the full three-parameter
/// Hessian, including parameters at their bounds, matching its SciPy-style
/// covariance calculation.
pub fn fit_gaussian_bounded_with_config<F: Float>(
    x: &[F],
    y: &[F],
    error: &[F],
    initial: [F; 3],
    bounds: [[F; 2]; 3],
    config: FitConfig<F>,
) -> Option<FitOutcome<F>> {
    let zero = F::zero();
    if x.len() != y.len() || x.len() != error.len() || x.len() < 3 {
        return None;
    }
    if !config.xtol.is_finite()
        || !config.ftol.is_finite()
        || !config.gtol.is_finite()
        || config.xtol <= zero
        || config.ftol <= zero
        || config.gtol <= zero
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
    if bounds[2][0] <= zero {
        return None;
    }

    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if error.iter().any(|v| !v.is_finite() || *v <= zero) {
        return None;
    }

    let limits = [
        [to_f64(&bounds[0][0]), to_f64(&bounds[0][1])],
        [to_f64(&bounds[1][0]), to_f64(&bounds[1][1])],
        [to_f64(&bounds[2][0]), to_f64(&bounds[2][1])],
    ];
    // Clamp the start point into the box; rmpfit rejects out-of-bounds starts
    // with MPError::InitBounds.
    let mut params = [
        clamp_f64(to_f64(&initial[0]), limits[0]),
        clamp_f64(to_f64(&initial[1]), limits[1]),
        clamp_f64(to_f64(&initial[2]), limits[2]),
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
        ftol: to_f64(&config.ftol),
        xtol: to_f64(&config.xtol),
        gtol: to_f64(&config.gtol),
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

    let status = problem.mpfit(&mut params).ok()?;

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
        params: [
            from_f64(params[0])?,
            from_f64(params[1])?,
            from_f64(params[2])?,
        ],
        errors,
        bestnorm: from_f64(bestnorm)?,
    })
}

/// Return SciPy-style errors from the inverse full Hessian, including any
/// parameters pegged at their bounds. This mirrors `mp_xerror_scipy` in the C
/// backend rather than rmpfit's reduced covariance for free parameters only.
fn scipy_style_errors<F: Float>(x: &[F], error: &[F], params: [f64; 3], bestnorm: f64) -> [F; 3] {
    let [amplitude, mean, sigma] = params;
    let mut hessian = [[0.0f64; 3]; 3];

    for (x_value, error_value) in x.iter().zip(error) {
        let inverse_error = 1.0 / to_f64(error_value);
        let z = (to_f64(x_value) - mean) / sigma;
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
        return [F::zero(); 3];
    }

    let scale = if x.len() > 3 {
        (bestnorm / (x.len() - 3) as f64).sqrt()
    } else {
        1.0
    };
    std::array::from_fn(|index| {
        let variance = cofactors[index] / determinant;
        let error = if variance > 0.0 {
            variance.sqrt() * scale
        } else {
            0.0
        };
        F::from(error).unwrap_or_else(F::zero)
    })
}

fn to_f64<F: Float>(value: &F) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn from_f64<F: Float>(value: f64) -> Option<F> {
    F::from(value)
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
        let outcome = fit_gaussian_bounded_with_config(
            &x,
            &y,
            &error,
            [1.0, 0.0, 69.137_89],
            [[0.9, 1.1], [-81.215_41, 81.215_41], [59.137_894, 200.0]],
            FitConfig::<f32>::default(),
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
