/*
 * gaussian_deviate.c - Unified Gaussian model deviate function for MPFIT
 *
 * Computes weighted residuals and analytical derivatives for Gaussian model
 * using a standard C implementation with cached exponential values.
 *
 * This file supports both float32 and float64 precision via preprocessor macros.
 * When MPFIT_FLOAT is defined, float (32-bit) precision is used; otherwise
 * double (64-bit) precision is used. This mirrors the unified mpfit.h approach.
 *
 * Includes analytical derivatives for amplitude, mean, and sigma parameters.
 */

#include <stdlib.h>
#include <math.h>

/*
 * Precision-dependent type and function definitions.
 * When MPFIT_FLOAT is defined: use float, expf, suffix _f32
 * Otherwise: use double, exp, no suffix (or _f64 for clarity)
 */
#ifdef MPFIT_FLOAT
typedef float gd_real;
#define GD_EXP(x) expf(x)
#define GD_HALF 0.5f
#define GD_STRUCT_NAME gaussian_private_data_f32
#define GD_FUNC_NAME myfunct_gaussian_deviates_with_derivatives_f32
#else
typedef double gd_real;
#define GD_EXP(x) exp(x)
#define GD_HALF 0.5
#define GD_STRUCT_NAME gaussian_private_data
#define GD_FUNC_NAME myfunct_gaussian_deviates_with_derivatives
#endif

/* Private data structure for passing x, y, error to user function */
struct GD_STRUCT_NAME
{
  const gd_real *x;     /* Independent variable */
  const gd_real *y;     /* Measured values */
  const gd_real *error; /* Measurement uncertainties */
};

/*
 * User function: compute Gaussian deviates (y - f(x))/error
 * For Gaussian model: f(x) = p[0] * exp(-0.5 * ((x - p[1]) / p[2])^2)
 * Parameters: p[0] = amplitude, p[1] = mean, p[2] = sigma
 */
int GD_FUNC_NAME(int m, int n, gd_real *p, gd_real *deviates,
                 gd_real **derivs, void *private_data)
{
  struct GD_STRUCT_NAME *pdata = (struct GD_STRUCT_NAME *)private_data;
  int i;
  gd_real amp, mean, sigma;
  (void)n; /* Suppress unused parameter warning */

  /* Extract parameters */
  amp = p[0];
  mean = p[1];
  sigma = p[2];

  /* Compute weighted deviates and cache exp() values for derivatives */
  /* Allocate temporary storage for exp terms if derivatives are needed */
  gd_real *expterms = NULL;
  gd_real *z_values = NULL;

  if (derivs)
  {
    expterms = (gd_real *)malloc(m * sizeof(gd_real));
    z_values = (gd_real *)malloc(m * sizeof(gd_real));
  }

  for (i = 0; i < m; i++)
  {
    gd_real x_i = pdata->x[i];
    gd_real y_i = pdata->y[i];
    gd_real err_i = pdata->error[i];
    gd_real z = (x_i - mean) / sigma;
    gd_real expterm = GD_EXP(-GD_HALF * z * z);
    gd_real model = amp * expterm;

    deviates[i] = (y_i - model) / err_i;

    /* Cache values for derivative calculation */
    if (derivs)
    {
      expterms[i] = expterm;
      z_values[i] = z;
    }
  }

  /* If derivs is non-zero, compute analytical derivatives using cached values */
  /* Since deviate = (y - model)/error, d(deviate)/d(p) = -d(model)/d(p)/error */
  if (derivs)
  {
    for (i = 0; i < m; i++)
    {
      gd_real err_i = pdata->error[i];
      gd_real expterm = expterms[i];
      gd_real z = z_values[i];

      /* d(deviate)/d(amp) = -d(model)/d(amp)/error = -exp(-0.5*z^2) / error */
      if (derivs[0])
        derivs[0][i] = -expterm / err_i;

      /* d(deviate)/d(mean) = -d(model)/d(mean)/error = -amp * z * exp(-0.5*z^2) / (sigma * error) */
      if (derivs[1])
        derivs[1][i] = -amp * z * expterm / (sigma * err_i);

      /* d(deviate)/d(sigma) = -d(model)/d(sigma)/error = -amp * z^2 * exp(-0.5*z^2) / (sigma * error) */
      if (derivs[2])
        derivs[2][i] = -amp * z * z * expterm / (sigma * err_i);
    }

    /* Free temporary storage */
    free(expterms);
    free(z_values);
  }

  return 0;
}
