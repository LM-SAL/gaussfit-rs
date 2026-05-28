use num_traits::Float;

use crate::linalg::{invert_3x3, solve_3x3};
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

impl Default for FitConfig<f64> {
    fn default() -> Self {
        Self {
            xtol: XTOL as f64,
            ftol: FTOL as f64,
            gtol: GTOL as f64,
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

/// Levenberg-Marquardt solver for a bounded 3-parameter Gaussian.
///
/// `bounds[i] = [lower, upper]` for parameter `i` (amp, mean, sigma).
/// Returns `None` if the solver did not reach a convergence criterion.
/// Error estimates are derived from the diagonal of the inverse Hessian
/// scaled by `sqrt(chi2 / dof)`.
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

    let mut params = [
        clamp(initial[0], bounds[0][0], bounds[0][1]),
        clamp(initial[1], bounds[1][0], bounds[1][1]),
        clamp(initial[2], bounds[2][0], bounds[2][1]),
    ];

    let mut lambda = F::from(1.0e-3_f64).unwrap();
    let scale_down = F::from(0.1_f64).unwrap();
    let scale_up = F::from(10.0_f64).unwrap();
    let lambda_min = F::from(1.0e-12_f64).unwrap();
    let one = F::one();
    let mut converged = false;
    let mut accepted_any_step = false;

    for _ in 0..config.max_iter {
        let (cost, gradient, hessian) = gaussian_normal_equations(x, y, error, params)?;

        // Absolute "fit is good enough" floor: when the chi-squared cost itself
        // drops to/below `ftol` the model already matches the data within
        // tolerance. This also guards near-perfect fits where f32 rounding stops
        // the line search from finding a strictly-smaller trial cost, which would
        // otherwise be misreported as non-convergence.
        if cost <= config.ftol {
            converged = true;
            break;
        }
        if max_abs(&gradient) <= config.gtol {
            converged = true;
            break;
        }

        let mut accepted = false;
        let mut step_converged = false;
        for _ in 0..24 {
            let mut damped = hessian;
            for i in 0..3 {
                damped[i][i] = damped[i][i] + lambda * Float::max(hessian[i][i].abs(), one);
            }

            let rhs = [-gradient[0], -gradient[1], -gradient[2]];
            let Some(step) = solve_3x3(damped, rhs) else {
                lambda = lambda * scale_up;
                continue;
            };

            let trial = [
                clamp(params[0] + step[0], bounds[0][0], bounds[0][1]),
                clamp(params[1] + step[1], bounds[1][0], bounds[1][1]),
                clamp(params[2] + step[2], bounds[2][0], bounds[2][1]),
            ];

            let trial_cost = gaussian_cost(x, y, error, trial)?;
            if trial_cost.is_finite() && trial_cost < cost {
                let step_norm = norm3([
                    trial[0] - params[0],
                    trial[1] - params[1],
                    trial[2] - params[2],
                ]);
                let param_norm = norm3(params);
                let cost_delta = (cost - trial_cost).abs();

                params = trial;
                lambda = Float::max(lambda * scale_down, lambda_min);
                accepted = true;
                accepted_any_step = true;

                step_converged = cost_delta <= config.ftol * (cost.abs() + config.ftol)
                    || step_norm <= config.xtol * (param_norm + config.xtol);
                break;
            }

            lambda = lambda * scale_up;
        }

        if !accepted {
            // MPFIT reports convergence when it reaches a local minimum and no
            // further damped step improves the cost. Preserve explicit failure
            // for fits that never accepted a step.
            if accepted_any_step {
                converged = true;
            }
            break;
        }
        if step_converged {
            converged = true;
            break;
        }
    }

    if !converged {
        return None;
    }

    let (bestnorm, _, hessian) = gaussian_normal_equations(x, y, error, params)?;
    let dof = F::from((x.len() as i32 - 3).max(1)).unwrap();
    let scale = (bestnorm / dof).sqrt();
    let inverse = invert_3x3(hessian);
    let zero = F::zero();
    let mut errors = [zero; 3];
    if let Some(inverse) = inverse {
        for i in 0..3 {
            let variance = inverse[i][i];
            errors[i] = if variance > zero && variance.is_finite() {
                variance.sqrt() * scale
            } else {
                zero
            };
        }
    }

    Some(FitOutcome {
        params,
        errors,
        bestnorm,
    })
}

/// Generic chi-squared cost for a Gaussian model (no gradient/Hessian).
/// Used in the LM inner trial-step loop where derivatives are not needed.
fn gaussian_cost<F: Float>(x: &[F], y: &[F], error: &[F], params: [F; 3]) -> Option<F> {
    let zero = F::zero();
    let neg_half = F::from(-0.5_f64).unwrap();
    let amp = params[0];
    let mean = params[1];
    let sigma = params[2];
    if sigma == zero || !sigma.is_finite() {
        return None;
    }
    let mut cost = zero;
    for i in 0..x.len() {
        let err = error[i];
        if err == zero || !err.is_finite() {
            return None;
        }
        let z = (x[i] - mean) / sigma;
        let model = amp * (neg_half * z * z).exp();
        let residual = (y[i] - model) / err;
        if !residual.is_finite() {
            return None;
        }
        cost = cost + residual * residual;
    }
    Some(cost)
}

/// Generic chi-squared cost, gradient, and approximate Hessian in one pass.
/// Returns `None` if any noise value is zero/non-finite or any residual overflows.
#[allow(clippy::needless_range_loop)]
fn gaussian_normal_equations<F: Float>(
    x: &[F],
    y: &[F],
    error: &[F],
    params: [F; 3],
) -> Option<(F, [F; 3], [[F; 3]; 3])> {
    let zero = F::zero();
    let neg_half = F::from(-0.5_f64).unwrap();
    let amp = params[0];
    let mean = params[1];
    let sigma = params[2];
    if sigma == zero || !sigma.is_finite() {
        return None;
    }

    let mut cost = zero;
    let mut gradient = [zero; 3];
    let mut hessian = [[zero; 3]; 3];

    for i in 0..x.len() {
        let err = error[i];
        if err == zero || !err.is_finite() {
            return None;
        }

        let z = (x[i] - mean) / sigma;
        let expterm = (neg_half * z * z).exp();
        let model = amp * expterm;
        let residual = (y[i] - model) / err;
        if !residual.is_finite() {
            return None;
        }

        let derivs = [
            -expterm / err,
            -amp * z * expterm / (sigma * err),
            -amp * z * z * expterm / (sigma * err),
        ];

        cost = cost + residual * residual;
        for row in 0..3 {
            gradient[row] = gradient[row] + derivs[row] * residual;
            for col in 0..=row {
                hessian[row][col] = hessian[row][col] + derivs[row] * derivs[col];
            }
        }
    }

    for row in 0..3 {
        for col in 0..row {
            hessian[col][row] = hessian[row][col];
        }
    }

    Some((cost, gradient, hessian))
}

fn clamp<F: Float>(value: F, low: F, high: F) -> F {
    Float::min(Float::max(value, low), high)
}

fn max_abs<F: Float>(values: &[F; 3]) -> F {
    values
        .iter()
        .fold(F::zero(), |acc, &v| Float::max(acc, v.abs()))
}

fn norm3<F: Float>(values: [F; 3]) -> F {
    (values[0] * values[0] + values[1] * values[1] + values[2] * values[2]).sqrt()
}
