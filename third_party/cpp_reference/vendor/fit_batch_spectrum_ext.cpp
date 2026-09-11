/*
 * fit_batch_spectrum_ext.cpp
 *
 * Batch Gaussian fitter — Python C extension + OpenMP core.
 *
 * Faithful port of the v1 sandbox approach: uses the original cmpfit-1.5
 * mpfit library (same as fit_single_spectrum_ext.cpp), with a per-thread
 * FitWorkspace to avoid allocations in the hot loop. Parallelised with
 * OpenMP; pass n_threads=1 for serial behaviour.
 *
 * Python module name: fit_batch_spectrum_ext
 * Exported function:  fit_spectra_batch(...)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#if defined(_OPENMP)
#include <omp.h>
#endif

/* Original cmpfit C library */
#define MPFIT_FLOAT 1
extern "C" {
#include "cmpfit-1.5/mpfit.h"
int mpfit(mp_func funct, int m, int npar, float *xall, mp_par *pars,
          mp_config *config, void *private_data, mp_result *result);
}

/* ------------------------------------------------------------------ */
/* Gaussian deviate — identical to fit_single_spectrum_ext.cpp (v1)   */
/* ------------------------------------------------------------------ */
struct gaussian_private_data_f32 {
    const float* x;
    const float* y;
    const float* error;
    float* expterms;
    float* z_values;
};

static int myfunct_gaussian(int m, int /*n*/, float* p, float* deviates,
                             float** derivs, void* private_data)
{
    gaussian_private_data_f32* pd = (gaussian_private_data_f32*)private_data;
    float amp = p[0], mean = p[1], sigma = p[2];
    for (int i = 0; i < m; i++) {
        float z  = (pd->x[i] - mean) / sigma;
        float ex = expf(-0.5f * z * z);
        deviates[i] = (pd->y[i] - amp * ex) / pd->error[i];
        if (derivs) { pd->expterms[i] = ex; pd->z_values[i] = z; }
    }
    if (derivs) {
        for (int i = 0; i < m; i++) {
            float e = pd->error[i], ex = pd->expterms[i], z = pd->z_values[i];
            if (derivs[0]) derivs[0][i] = -ex / e;
            if (derivs[1]) derivs[1][i] = -amp * z * ex / (sigma * e);
            if (derivs[2]) derivs[2][i] = -amp * z * z * ex / (sigma * e);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-thread workspace (same as v1 sandbox)                           */
/* ------------------------------------------------------------------ */
struct FitWorkspace {
    std::vector<float>  spectrum_win, dopp_win, xdata, ydata, noise_win;
    std::vector<mp_par> pars;
    std::vector<float>  resid, xerror, xerror_scipy, covar, expterms, z_values;

    void ensure_capacity(int n) {
        if ((int)spectrum_win.size() < n) {
            spectrum_win.resize(n); dopp_win.resize(n);
            xdata.resize(n); ydata.resize(n); noise_win.resize(n);
            resid.resize(n); expterms.resize(n); z_values.resize(n);
        }
        if (pars.size() < 3) {
            pars.resize(3); xerror.resize(3); xerror_scipy.resize(3); covar.resize(9);
        }
    }
};

static const float FLAG_SUCCESS        = 0.0f;
static const float FLAG_NO_LOCAL_MAX   = 1.0f;
static const float FLAG_NO_CONVERGENCE = 2.0f;

/* ------------------------------------------------------------------ */
/* Core single-spectrum fit (mirrors fit_single_spectrum_ext.cpp)      */
/* ------------------------------------------------------------------ */
static inline void fit_single_core(
    const float* spectrum, const float* dopp_slit, const float* spec_noise,
    float guide_velocity, float velocity_range,
    int npix, int npix_slack, float dv, float width_min, int SG_xpixels,
    float amplitude_rel_min, float amplitude_rel_max,
    float width_max, float width_guess,
    float* fit_results, uint8_t* gfit_mask_row,
    FitWorkspace& ws)
{
    for (int i = 0; i < 8; i++) fit_results[i] = std::numeric_limits<float>::quiet_NaN();
    std::memset(gfit_mask_row, 0, SG_xpixels);

    float dv_slac = dv * npix_slack;
    int imax = 0; float max_val = -1e30f; bool found = false;
    for (int i = 0; i < SG_xpixels; i++) {
        if (std::fabs(dopp_slit[i] - guide_velocity) <= (velocity_range + dv_slac)) {
            if (spectrum[i] > max_val) { max_val = spectrum[i]; imax = i; found = true; }
        }
    }
    if (!found || max_val <= 0.0f) { fit_results[7] = FLAG_NO_LOCAL_MAX; return; }
    if (std::fabs(dopp_slit[imax] - guide_velocity) > velocity_range) {
        fit_results[7] = FLAG_NO_LOCAL_MAX; return;
    }

    int i_left  = imax - npix;     if (i_left  < 0)          i_left  = 0;
    int i_right = imax + npix + 1; if (i_right > SG_xpixels) i_right = SG_xpixels;
    int n_pix   = i_right - i_left;
    if (n_pix < (2 * npix + 1)) {
        i_left  = imax - npix - 1; if (i_left  < 0)          i_left  = 0;
        i_right = imax + npix + 2; if (i_right > SG_xpixels) i_right = SG_xpixels;
        n_pix   = i_right - i_left;
    }

    ws.ensure_capacity(n_pix);

    int n_valid = 0;
    for (int i = 0; i < n_pix; i++) {
        ws.spectrum_win[i] = spectrum[i_left + i] / max_val;
        ws.dopp_win[i]     = dopp_slit[i_left + i];
        if (!std::isnan(ws.spectrum_win[i])) n_valid++;
    }
    if (n_valid < 3) { fit_results[7] = FLAG_NO_LOCAL_MAX; return; }

    int idx = 0;
    for (int i = 0; i < n_pix; i++) {
        if (!std::isnan(ws.spectrum_win[i])) {
            ws.xdata[idx]     = ws.dopp_win[i];
            ws.ydata[idx]     = ws.spectrum_win[i];
            ws.noise_win[idx] = spec_noise[i_left + i] / max_val;
            idx++;
        }
    }

    int   npar = 3;
    float vel_center = dopp_slit[imax], vel_hr = dv * npix;
    float best_params[3] = {1.0f, vel_center, width_guess};

    std::memset(ws.pars.data(), 0, npar * sizeof(mp_par));
    ws.pars[0].limited[0]=1; ws.pars[0].limits[0]=amplitude_rel_min;
    ws.pars[0].limited[1]=1; ws.pars[0].limits[1]=amplitude_rel_max; ws.pars[0].side=3;
    ws.pars[1].limited[0]=1; ws.pars[1].limits[0]=vel_center - vel_hr;
    ws.pars[1].limited[1]=1; ws.pars[1].limits[1]=vel_center + vel_hr; ws.pars[1].side=3;
    ws.pars[2].limited[0]=1; ws.pars[2].limits[0]=width_min;
    ws.pars[2].limited[1]=1; ws.pars[2].limits[1]=width_max; ws.pars[2].side=3;

    mp_config config; std::memset(&config, 0, sizeof(config));
    config.ftol=1.0e-6f; config.xtol=1.0e-6f; config.gtol=1.0e-6f; config.maxiter=2000;

    mp_result result; std::memset(&result, 0, sizeof(result));
    result.resid        = ws.resid.data();
    result.xerror       = ws.xerror.data();
    result.covar        = ws.covar.data();
    result.xerror_scipy = ws.xerror_scipy.data();

    gaussian_private_data_f32 pdata;
    pdata.x=ws.xdata.data(); pdata.y=ws.ydata.data(); pdata.error=ws.noise_win.data();
    pdata.expterms=ws.expterms.data(); pdata.z_values=ws.z_values.data();

    int status = mpfit(myfunct_gaussian, n_valid, npar, best_params,
                       ws.pars.data(), &config, (void*)&pdata, &result);
    if (status <= 0) { fit_results[7] = FLAG_NO_CONVERGENCE; return; }

    int dof = n_valid - npar; if (dof <= 0) dof = 1;
    fit_results[0] = best_params[0] * max_val;
    fit_results[1] = best_params[1];
    fit_results[2] = best_params[2];
    fit_results[3] = ws.xerror_scipy[0] * max_val;
    fit_results[4] = ws.xerror_scipy[1];
    fit_results[5] = ws.xerror_scipy[2];
    fit_results[6] = result.bestnorm / (float)dof;
    fit_results[7] = FLAG_SUCCESS;

    /* Write fit window mask */
    std::memset(gfit_mask_row + i_left, 1, n_pix);
}

/* ------------------------------------------------------------------ */
/* OpenMP batch core                                                   */
/* ------------------------------------------------------------------ */
static void fit_spectra_batch_core(
    const float* spectra,          /* [B, SG_xpixels] */
    const float* dopp_slits,       /* [n_slit, SG_xpixels] */
    int   n_slit,
    const float* spec_noises,      /* [n_shapes, SG_xpixels] or [B, SG_xpixels] */
    int   noise_is_shared,         /* 1 if noise lacks slit dim */
    const float* guide_velocities, /* [B] */
    int   guide_stride,            /* 0=shared constant, 1=per-spectrum */
    float velocity_range,
    int   npix,
    int   npix_slack,
    float dv,
    float width_min,
    int   SG_xpixels,
    float amplitude_rel_min,
    float amplitude_rel_max,
    float width_max,
    float width_guess,
    float*   fit_results,          /* [B, 8] */
    uint8_t* gfit_mask,            /* [B, SG_xpixels] */
    int   B,
    int   n_threads)
{
#if defined(_OPENMP)
    if (n_threads > 0) omp_set_num_threads(n_threads);
#endif
#ifndef OMP_BLOCK_SIZE
#define OMP_BLOCK_SIZE 32
#endif
    #pragma omp parallel
    {
        FitWorkspace ws;
        ws.ensure_capacity(2 * npix + 4);

        #pragma omp for schedule(dynamic, OMP_BLOCK_SIZE)
        for (int b = 0; b < B; ++b) {
            const float* spectrum      = &spectra[b * SG_xpixels];
            const float* dopp_slit     = &dopp_slits[(b % n_slit) * SG_xpixels];
            const float* spec_noise    = noise_is_shared
                ? &spec_noises[(b / n_slit) * SG_xpixels]
                : &spec_noises[b * SG_xpixels];
            float guide_velocity       = guide_velocities[b * guide_stride];
            uint8_t* mask_row          = &gfit_mask[b * SG_xpixels];

            fit_single_core(
                spectrum, dopp_slit, spec_noise, guide_velocity, velocity_range,
                npix, npix_slack, dv, width_min, SG_xpixels,
                amplitude_rel_min, amplitude_rel_max, width_max, width_guess,
                &fit_results[b * 8], mask_row, ws);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Python C-API wrapper                                               */
/* ------------------------------------------------------------------ */
static PyObject *py_fit_spectra_batch(PyObject *self, PyObject *args, PyObject *keywds)
{
    (void)self;
    PyObject *spectra_obj, *dopp_slits_obj, *spec_noises_obj;
    PyObject *guide_velocities_obj, *gfit_mask_obj;
    int n_slit, noise_is_shared, guide_stride, npix, npix_slack, SG_xpixels, B;
    int n_threads = 0;
    float velocity_range, dv, width_min, amplitude_rel_min, amplitude_rel_max;
    float width_max, width_guess;

    static const char *kwlist[] = {
        "spectra","dopp_slits","n_slit","spec_noises","noise_is_shared",
        "guide_velocities","guide_stride","velocity_range","npix","npix_slack",
        "dv","width_min","SG_xpixels","amplitude_rel_min","amplitude_rel_max",
        "width_max","width_guess","gfit_mask","B","n_threads",NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, keywds, "OOiOiOifiiffiffffOii", const_cast<char**>(kwlist),
            &spectra_obj, &dopp_slits_obj, &n_slit, &spec_noises_obj, &noise_is_shared,
            &guide_velocities_obj, &guide_stride, &velocity_range, &npix, &npix_slack,
            &dv, &width_min, &SG_xpixels, &amplitude_rel_min, &amplitude_rel_max,
            &width_max, &width_guess, &gfit_mask_obj, &B, &n_threads))
        return NULL;

    PyArrayObject *sa = (PyArrayObject*)PyArray_FROM_OTF(spectra_obj,         NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *da = (PyArrayObject*)PyArray_FROM_OTF(dopp_slits_obj,      NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *na = (PyArrayObject*)PyArray_FROM_OTF(spec_noises_obj,     NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ga = (PyArrayObject*)PyArray_FROM_OTF(guide_velocities_obj,NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *ma = (PyArrayObject*)PyArray_FROM_OTF(gfit_mask_obj,       NPY_BOOL,    NPY_ARRAY_INOUT_ARRAY);

    if (!sa||!da||!na||!ga||!ma) {
        Py_XDECREF(sa); Py_XDECREF(da); Py_XDECREF(na);
        Py_XDECREF(ga); Py_XDECREF(ma);
        PyErr_SetString(PyExc_ValueError,"Failed to convert input arrays"); return NULL;
    }

    npy_intp dims[2] = {B, 8};
    PyArrayObject *ra = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!ra) {
        Py_DECREF(sa); Py_DECREF(da); Py_DECREF(na); Py_DECREF(ga); Py_DECREF(ma);
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS
    fit_spectra_batch_core(
        (float*)PyArray_DATA(sa), (float*)PyArray_DATA(da), n_slit,
        (float*)PyArray_DATA(na), noise_is_shared,
        (float*)PyArray_DATA(ga), guide_stride,
        velocity_range, npix, npix_slack, dv, width_min, SG_xpixels,
        amplitude_rel_min, amplitude_rel_max, width_max, width_guess,
        (float*)PyArray_DATA(ra), (uint8_t*)PyArray_DATA(ma), B, n_threads);
    Py_END_ALLOW_THREADS

    Py_DECREF(sa); Py_DECREF(da); Py_DECREF(na); Py_DECREF(ga);
    PyArray_ResolveWritebackIfCopy(ma);
    Py_DECREF(ma);

    return (PyObject*)ra;
}

static PyMethodDef module_methods[] = {
    {"fit_spectra_batch", (PyCFunction)py_fit_spectra_batch,
     METH_VARARGS|METH_KEYWORDS,
     "Batch Gaussian fitter (v1-faithful cmpfit + OpenMP). "
     "n_threads=0 uses all cores, n_threads=1 for serial."},
    {NULL,NULL,0,NULL}};

static struct PyModuleDef mod = {
    PyModuleDef_HEAD_INIT, "fit_batch_spectrum_ext",
    "Batch Gaussian fitting extension (v1-faithful, cmpfit + OpenMP)", -1,
    module_methods};

PyMODINIT_FUNC PyInit_fit_batch_spectrum_ext(void)
{ import_array(); return PyModule_Create(&mod); }
