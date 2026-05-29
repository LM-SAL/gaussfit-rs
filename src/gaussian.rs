use num_traits::Float;
use rmpfit::{MPConfig, MPFitter, MPPar, MPResult, MPSuccess};

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

/// Weighted-residual problem handed to the MPFIT (rmpfit) solver.
///
/// Residuals are `(y - amp * exp(-0.5 * ((x - mean) / sigma)^2)) / error`,
/// matching the MPFIT convention `(y - f(x)) / y_err`. All arithmetic is done
/// in `f64`; callers using `f32` convert in and out.
struct GaussianProblem {
    x: Vec<f64>,
    y: Vec<f64>,
    error: Vec<f64>,
    params: [MPPar; 3],
    config: MPConfig,
}

impl MPFitter for GaussianProblem {
    fn eval(&mut self, params: &[f64], deviates: &mut [f64]) -> MPResult<()> {
        let amp = params[0];
        let mean = params[1];
        let sigma = params[2];
        for (i, deviate) in deviates.iter_mut().enumerate() {
            let z = (self.x[i] - mean) / sigma;
            let model = amp * (-0.5 * z * z).exp();
            *deviate = (self.y[i] - model) / self.error[i];
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
}

/// Box-constrained Levenberg-Marquardt fit of a 3-parameter Gaussian.
///
/// `bounds[i] = [lower, upper]` for parameter `i` (amp, mean, sigma). Returns
/// `None` if the solver did not reach a convergence criterion (e.g. it hit the
/// iteration limit) or the inputs are invalid.
///
/// The fit is delegated to [`rmpfit`], a pure-Rust port of the CMPFIT/MINPACK
/// `mpfit` routine, so the convergence and bound-handling semantics match the
/// original C extension. Parameter errors are the 1-sigma uncertainties from
/// the covariance diagonal, scaled by `sqrt(chi2 / dof)`.
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

    // Convert to f64 (rmpfit operates in double precision) and validate the
    // sample data the way the old solver did via its per-point checks.
    let xf: Vec<f64> = x.iter().map(to_f64).collect();
    let yf: Vec<f64> = y.iter().map(to_f64).collect();
    let ef: Vec<f64> = error.iter().map(to_f64).collect();
    if xf.iter().any(|v| !v.is_finite()) || yf.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if ef.iter().any(|v| !v.is_finite() || *v <= 0.0) {
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
        x: xf,
        y: yf,
        error: ef,
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
    let dof = ((x.len() as i32 - 3).max(1)) as f64;
    let scale = (bestnorm / dof).sqrt();

    let mut errors = [zero; 3];
    for (i, slot) in errors.iter_mut().enumerate() {
        let err = status.xerror.get(i).copied().unwrap_or(0.0) * scale;
        *slot = if err.is_finite() {
            F::from(err).unwrap_or(zero)
        } else {
            zero
        };
    }

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

fn to_f64<F: Float>(value: &F) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn from_f64<F: Float>(value: f64) -> Option<F> {
    F::from(value)
}

fn clamp_f64(value: f64, limit: [f64; 2]) -> f64 {
    value.max(limit[0]).min(limit[1])
}
