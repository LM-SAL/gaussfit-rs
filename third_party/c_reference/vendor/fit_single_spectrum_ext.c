/*
 * fit_single_spectrum_ext.c - Python C extension for single spectrum Gaussian fitting
 *
 * Implements _fit_single_spectrum_c() which replicates the Python fit_single_spectrum_fallback()
 * function for improved performance.
 *
 * Build: Included in pyproject.toml as an extension module
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Float32 precision - MPFIT_FLOAT defined by setup.py */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Path to cmpfit library - from ftools package */
#include "cmpfit-1.5/mpfit.h"

/* Include Gaussian deviate computation (float32 version due to MPFIT_FLOAT) */
/* gaussian_deviate.c from ftools via include_dirs */
#include "gaussian_deviate.c"

/* Flag values matching Python implementation */
#define FLAG_SUCCESS 0
#define FLAG_NO_LOCAL_MAX 1
#define FLAG_NO_CONVERGENCE 2

/*
 * fit_single_spectrum_c - Core C implementation
 *
 * Parameters:
 *   spectrum: float32 array of flux values (SG_xpixels,)
 *   dopp_slit: float32 array of velocity values (SG_xpixels,)
 *   spec_noise: float32 array of noise values (SG_xpixels,)
 *   guide_velocity: guide velocity for masking centre
 *   velocity_range: velocity range for masking
 *   npix: number of pixels for fitting window
 *   npix_slack: slack pixels for pix_slac vetting
 *   dv: velocity spacing (median gradient)
 *   width_min: minimum line width
 *   SG_xpixels: total number of spectral pixels
 *
 * Outputs:
 *   fit_results: 8-element float32 array [amplitude, velocity, linewidth,
 *                error_amplitude, error_velocity, error_linewidth, redchisq, flag]
 *   i_left, i_right: mask range indices (returned via pointers)
 *
 * Returns: 0 on success, non-zero on error
 */
static int fit_single_spectrum_c(
    const float *spectrum,
    const float *dopp_slit,
    const float *spec_noise,
    float guide_velocity,
    float velocity_range,
    int npix,
    int npix_slack,
    float dv,
    float width_min,
    int SG_xpixels,
    float amplitude_rel_min,
    float amplitude_rel_max,
    float width_max,
    float width_guess,
    float *fit_results,
    int *out_i_left,
    int *out_i_right)
{
    int i, imax, i_left, i_right, n_pix_available, n_valid, npar;
    float dv_slac, max_val;
    float vel_center, vel_half_range;
    float *xdata = NULL, *ydata = NULL, *noise_win = NULL;
    float *spectrum_win = NULL, *dopp_win = NULL;
    mp_par *pars = NULL;
    mp_config config;
    mp_result result;
    struct gaussian_private_data_f32 private_data;
    float best_params[3];
    float *resid = NULL, *xerror = NULL, *xerror_scipy = NULL, *covar = NULL;
    int status, dof;

    /* Initialize outputs */
    for (i = 0; i < 8; i++)
    {
        fit_results[i] = NAN;
    }
    *out_i_left = 0;
    *out_i_right = 0;

    /* Step 1: Apply velocity range mask (enlarged by dv_slac for pix_slac vetting) */
    dv_slac = dv * npix_slack;

    /* Find maximum in masked spectrum */
    imax = 0;
    max_val = -INFINITY;

    for (i = 0; i < SG_xpixels; i++)
    {
        float vel_abs = fabsf(dopp_slit[i] - guide_velocity);
        if (vel_abs <= (velocity_range + dv_slac))
        {
            float spec_val = spectrum[i];
            if (spec_val > max_val)
            {
                max_val = spec_val;
                imax = i;
            }
        }
    }

    /* Step 2: Check if valid maximum found */
    if (max_val == -INFINITY)
    {
        fit_results[7] = FLAG_NO_LOCAL_MAX;
        return 0;
    }

    if (max_val == 0.0f)
    {
        fit_results[7] = FLAG_NO_LOCAL_MAX;
        return 0;
    }

    /* pix_slac vetting: reject if max is in the outer dv_slac band */
    if (fabsf(dopp_slit[imax] - guide_velocity) > velocity_range)
    {
        fit_results[7] = FLAG_NO_LOCAL_MAX;
        return 0;
    }

    /* Step 3: Extract window around maximum (npix pixels on each side) */
    i_left = imax - npix;
    if (i_left < 0)
        i_left = 0;

    i_right = imax + npix + 1;
    if (i_right > SG_xpixels)
        i_right = SG_xpixels;

    /* Ensure we have at least 2*npix+1 pixels if possible */
    n_pix_available = i_right - i_left;
    if (n_pix_available < (2 * npix + 1))
    {
        i_left = imax - npix - 1;
        if (i_left < 0)
            i_left = 0;
        i_right = imax + npix + 2;
        if (i_right > SG_xpixels)
            i_right = SG_xpixels;
    }

    *out_i_left = i_left;
    *out_i_right = i_right;

    n_pix_available = i_right - i_left;

    /* Extract window and normalize */
    spectrum_win = (float *)malloc(n_pix_available * sizeof(float));
    dopp_win = (float *)malloc(n_pix_available * sizeof(float));
    if (!spectrum_win || !dopp_win)
    {
        free(spectrum_win);
        free(dopp_win);
        fit_results[7] = FLAG_NO_CONVERGENCE;
        return -1;
    }

    /* Step 4: Normalize by maximum and count valid (non-NaN) points */
    n_valid = 0;
    for (i = 0; i < n_pix_available; i++)
    {
        spectrum_win[i] = spectrum[i_left + i] / max_val;
        dopp_win[i] = dopp_slit[i_left + i];
        if (!isnan(spectrum_win[i]))
        {
            n_valid++;
        }
    }

    /* Step 5: Need at least 3 points for fitting */
    if (n_valid < 3)
    {
        free(spectrum_win);
        free(dopp_win);
        fit_results[7] = FLAG_NO_LOCAL_MAX;
        return 0;
    }

    /* Allocate arrays for valid data only */
    xdata = (float *)malloc(n_valid * sizeof(float));
    ydata = (float *)malloc(n_valid * sizeof(float));
    noise_win = (float *)malloc(n_valid * sizeof(float));
    if (!xdata || !ydata || !noise_win)
    {
        free(spectrum_win);
        free(dopp_win);
        free(xdata);
        free(ydata);
        free(noise_win);
        fit_results[7] = FLAG_NO_CONVERGENCE;
        return -1;
    }

    /* Copy valid data */
    n_valid = 0;
    for (i = 0; i < n_pix_available; i++)
    {
        if (!isnan(spectrum_win[i]))
        {
            xdata[n_valid] = dopp_win[i];
            ydata[n_valid] = spectrum_win[i];
            noise_win[n_valid] = spec_noise[i_left + i] / max_val;
            n_valid++;
        }
    }

    free(spectrum_win);
    free(dopp_win);

    /* Step 6: Set up fitting parameters */
    npar = 3;
    vel_center = dopp_slit[imax];
    vel_half_range = dv * npix;

    /* Initial guesses: [amplitude=1.0, velocity=vel_center, linewidth=width_guess] */
    best_params[0] = 1.0f;
    best_params[1] = vel_center;
    best_params[2] = width_guess;

    /* Setup parameter constraints */
    pars = (mp_par *)calloc(npar, sizeof(mp_par));
    if (!pars)
    {
        free(xdata);
        free(ydata);
        free(noise_win);
        fit_results[7] = FLAG_NO_CONVERGENCE;
        return -1;
    }

    /* Amplitude bounds: amplitude_rel_min to amplitude_rel_max */
    pars[0].limited[0] = 1;
    pars[0].limits[0] = amplitude_rel_min;
    pars[0].limited[1] = 1;
    pars[0].limits[1] = amplitude_rel_max;
    pars[0].side = 3; /* Analytical derivatives */

    /* Velocity bounds: vel_center ± vel_half_range */
    pars[1].limited[0] = 1;
    pars[1].limits[0] = vel_center - vel_half_range;
    pars[1].limited[1] = 1;
    pars[1].limits[1] = vel_center + vel_half_range;
    pars[1].side = 3;

    /* Linewidth bounds: width_min to width_max */
    pars[2].limited[0] = 1;
    pars[2].limits[0] = width_min;
    pars[2].limited[1] = 1;
    pars[2].limits[1] = width_max;
    pars[2].side = 3;

    /* Configure MPFIT */
    memset(&config, 0, sizeof(config));
    config.ftol = 1.0e-6f;
    config.xtol = 1.0e-6f;
    config.gtol = 1.0e-6f;
    config.maxiter = 2000;

    /* Allocate result arrays */
    resid = (float *)malloc(n_valid * sizeof(float));
    xerror = (float *)malloc(npar * sizeof(float));
    xerror_scipy = (float *)malloc(npar * sizeof(float));
    covar = (float *)malloc(npar * npar * sizeof(float));
    if (!resid || !xerror || !xerror_scipy || !covar)
    {
        free(pars);
        free(xdata);
        free(ydata);
        free(noise_win);
        free(resid);
        free(xerror);
        free(xerror_scipy);
        free(covar);
        fit_results[7] = FLAG_NO_CONVERGENCE;
        return -1;
    }

    /* Setup result structure */
    memset(&result, 0, sizeof(result));
    result.resid = resid;
    result.xerror = xerror;
    result.covar = covar;
    result.xerror_scipy = xerror_scipy;

    /* Setup private data for Gaussian model */
    private_data.x = xdata;
    private_data.y = ydata;
    private_data.error = noise_win;

    /* Step 7: Call MPFIT */
    status = mpfit(myfunct_gaussian_deviates_with_derivatives_f32,
                   n_valid, npar, best_params, pars, &config,
                   (void *)&private_data, &result);

    if (status <= 0)
    {
        /* Fit failed */
        free(pars);
        free(xdata);
        free(ydata);
        free(noise_win);
        free(resid);
        free(xerror);
        free(xerror_scipy);
        free(covar);
        fit_results[7] = FLAG_NO_CONVERGENCE;
        return 0;
    }

    /* Step 8: Store results (scale back to original units) */
    dof = n_valid - npar;
    if (dof <= 0)
        dof = 1;

    fit_results[0] = best_params[0] * max_val;     /* amplitude */
    fit_results[1] = best_params[1];               /* velocity */
    fit_results[2] = best_params[2];               /* linewidth */
    fit_results[3] = xerror_scipy[0] * max_val;    /* error_amplitude */
    fit_results[4] = xerror_scipy[1];              /* error_velocity */
    fit_results[5] = xerror_scipy[2];              /* error_linewidth */
    fit_results[6] = result.bestnorm / (float)dof; /* reduced chi-square */
    fit_results[7] = FLAG_SUCCESS;                 /* flag */

    /* Cleanup */
    free(pars);
    free(xdata);
    free(ydata);
    free(noise_win);
    free(resid);
    free(xerror);
    free(xerror_scipy);
    free(covar);

    return 0;
}

/*
 * Python wrapper function: _fit_single_spectrum_c
 *
 * Arguments:
 *   spectrum: 1D float32 array
 *   dopp_slit: 1D float32 array
 *   spec_noise: 1D float32 array
 *   guide_velocity: float
 *   velocity_range: float
 *   npix: int
 *   npix_slack: int
 *   dv: float
 *   width_min: float
 *   SG_xpixels: int
 *   amplitude_rel_min: float (lower bound for amplitude, relative to max)
 *   amplitude_rel_max: float (upper bound for amplitude, relative to max)
 *   width_max: float (upper bound for linewidth)
 *   width_guess: float (initial guess for linewidth)
 *
 * Returns: tuple (fit_results, (i_left, i_right))
 */
static PyObject *py_fit_single_spectrum_c(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject *spectrum_obj, *dopp_slit_obj, *spec_noise_obj;
    float guide_velocity, velocity_range, dv, width_min;
    float amplitude_rel_min, amplitude_rel_max, width_max, width_guess;
    int npix, npix_slack, SG_xpixels;

    static char *kwlist[] = {"spectrum", "dopp_slit", "spec_noise",
                             "guide_velocity", "velocity_range",
                             "npix", "npix_slack", "dv", "width_min",
                             "SG_xpixels",
                             "amplitude_rel_min", "amplitude_rel_max",
                             "width_max", "width_guess", NULL};

    (void)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOffiiffiffff", kwlist,
                                     &spectrum_obj, &dopp_slit_obj,
                                     &spec_noise_obj,
                                     &guide_velocity, &velocity_range,
                                     &npix, &npix_slack,
                                     &dv, &width_min, &SG_xpixels,
                                     &amplitude_rel_min, &amplitude_rel_max,
                                     &width_max, &width_guess))
    {
        return NULL;
    }

    /* Convert to contiguous float32 arrays */
    PyArrayObject *spectrum_arr = (PyArrayObject *)PyArray_FROM_OTF(
        spectrum_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *dopp_slit_arr = (PyArrayObject *)PyArray_FROM_OTF(
        dopp_slit_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *spec_noise_arr = (PyArrayObject *)PyArray_FROM_OTF(
        spec_noise_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!spectrum_arr || !dopp_slit_arr || !spec_noise_arr)
    {
        Py_XDECREF(spectrum_arr);
        Py_XDECREF(dopp_slit_arr);
        Py_XDECREF(spec_noise_arr);
        PyErr_SetString(PyExc_ValueError, "Failed to convert input arrays to float32");
        return NULL;
    }

    /* Get data pointers */
    float *spectrum = (float *)PyArray_DATA(spectrum_arr);
    float *dopp_slit = (float *)PyArray_DATA(dopp_slit_arr);
    float *spec_noise = (float *)PyArray_DATA(spec_noise_arr);

    /* Allocate output array */
    npy_intp dims[1] = {8};
    PyArrayObject *fit_results_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!fit_results_arr)
    {
        Py_DECREF(spectrum_arr);
        Py_DECREF(dopp_slit_arr);
        Py_DECREF(spec_noise_arr);
        return NULL;
    }
    float *fit_results = (float *)PyArray_DATA(fit_results_arr);

    int i_left, i_right;

    /* Call C implementation */
    int ret = fit_single_spectrum_c(
        spectrum, dopp_slit, spec_noise, guide_velocity,
        velocity_range, npix, npix_slack,
        dv, width_min, SG_xpixels,
        amplitude_rel_min, amplitude_rel_max, width_max, width_guess,
        fit_results, &i_left, &i_right);

    Py_DECREF(spectrum_arr);
    Py_DECREF(dopp_slit_arr);
    Py_DECREF(spec_noise_arr);

    if (ret < 0)
    {
        Py_DECREF(fit_results_arr);
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed in fit_single_spectrum_c");
        return NULL;
    }

    /* Build return tuple: (fit_results, (i_left, i_right)) */
    PyObject *mask_range = Py_BuildValue("(ii)", i_left, i_right);
    PyObject *result = PyTuple_Pack(2, (PyObject *)fit_results_arr, mask_range);

    Py_DECREF(fit_results_arr);
    Py_DECREF(mask_range);

    return result;
}

/* Module method definitions */
static PyMethodDef module_methods[] = {
    {"_fit_single_spectrum_c", (PyCFunction)py_fit_single_spectrum_c,
     METH_VARARGS | METH_KEYWORDS,
     "Fit a single spectrum with Gaussian model (C implementation).\n\n"
     "Parameters\n"
     "----------\n"
     "spectrum : ndarray (float32)\n"
     "    1D flux array\n"
     "dopp_slit : ndarray (float32)\n"
     "    1D Doppler/velocity axis\n"
     "spec_noise : ndarray (float32)\n"
     "    1D noise array\n"
     "guide_velocity : float\n"
     "    Guide velocity for masking centre\n"
     "velocity_range : float\n"
     "    Velocity range for masking\n"
     "npix : int\n"
     "    Number of pixels for fitting window\n"
     "npix_slack : int\n"
     "    Slack pixels for pix_slac vetting\n"
     "dv : float\n"
     "    Velocity spacing\n"
     "width_min : float\n"
     "    Minimum line width\n"
     "SG_xpixels : int\n"
     "    Total number of spectral pixels\n"
     "amplitude_rel_min : float\n"
     "    Lower bound for amplitude (relative to max)\n"
     "amplitude_rel_max : float\n"
     "    Upper bound for amplitude (relative to max)\n"
     "width_max : float\n"
     "    Upper bound for linewidth (km/s)\n"
     "width_guess : float\n"
     "    Initial guess for linewidth (km/s)\n\n"
     "Returns\n"
     "-------\n"
     "tuple\n"
     "    (fit_results, (i_left, i_right)) where fit_results is 8-element array"},
    {NULL, NULL, 0, NULL}};

/* Module definition */
static struct PyModuleDef fit_single_spectrum_ext_module = {
    PyModuleDef_HEAD_INIT,
    "fit_single_spectrum_ext",
    "C extension for single spectrum Gaussian fitting",
    -1,
    module_methods};

/* Module initialization */
PyMODINIT_FUNC PyInit_fit_single_spectrum_ext(void)
{
    import_array(); /* Initialize NumPy */
    return PyModule_Create(&fit_single_spectrum_ext_module);
}
