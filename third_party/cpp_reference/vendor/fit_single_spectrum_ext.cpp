/*
 * fit_single_spectrum_ext.cpp
 *
 * Faithful C++ port of fastfit/fit_single_spectrum_ext.c (original C
 * reference), using the same cmpfit-1.5 mpfit library and identical config.
 * No extra maxfev cap, no NaN guard in deviate function.
 * Equivalent to sandbox v1 - numerically matches the C implementation.
 *
 * Batch path uses OpenMP per-thread workspace (same as v1).
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
#if defined(_OPENMP)
#include <omp.h>
#endif

/* Use original cmpfit C library - same as fastfit/fit_single_spectrum_ext.c */
#define MPFIT_FLOAT 1
extern "C" {
#include "cmpfit-1.5/mpfit.h"
int mpfit(mp_func funct, int m, int npar, float *xall, mp_par *pars,
          mp_config *config, void *private_data, mp_result *result);
}

/* ------------------------------------------------------------------ */
/* Gaussian deviate - no NaN guard, analytical derivatives            */
/* Identical to C original                                            */
/* ------------------------------------------------------------------ */
struct gaussian_private_data_f32 {
    const float* x;
    const float* y;
    const float* error;
    float* expterms;
    float* z_values;
};

extern "C" int myfunct_gaussian_deviates_with_derivatives_f32_opt(
    int m, int /*n*/, float* p, float* deviates,
    float** derivs, void* private_data)
{
    gaussian_private_data_f32* pdata = (gaussian_private_data_f32*)private_data;
    float amp   = p[0];
    float mean  = p[1];
    float sigma = p[2];
    float* expterms = pdata->expterms;
    float* z_values = pdata->z_values;
    for (int i = 0; i < m; i++) {
        float z = (pdata->x[i] - mean) / sigma;
        float ex = expf(-0.5f * z * z);
        deviates[i] = (pdata->y[i] - amp * ex) / pdata->error[i];
        if (derivs) { expterms[i] = ex; z_values[i] = z; }
    }
    if (derivs) {
        for (int i = 0; i < m; i++) {
            float e = pdata->error[i], ex = expterms[i], z = z_values[i];
            if (derivs[0]) derivs[0][i] = -ex / e;
            if (derivs[1]) derivs[1][i] = -amp * z * ex / (sigma * e);
            if (derivs[2]) derivs[2][i] = -amp * z * z * ex / (sigma * e);
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Per-thread workspace (std::vector, same as v1)                     */
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
/* Core fitter - step-by-step mirror of the C original                */
/* ------------------------------------------------------------------ */
inline int fit_single_spectrum_cpp_core(
    const float* spectrum, const float* dopp_slit, const float* spec_noise,
    float guide_velocity, float velocity_range,
    int npix, int npix_slack, float dv, float width_min, int SG_xpixels,
    float amplitude_rel_min, float amplitude_rel_max,
    float width_max, float width_guess,
    float* fit_results, int& out_i_left, int& out_i_right,
    FitWorkspace& ws)
{
    for (int i = 0; i < 8; i++) fit_results[i] = std::nanf("");
    out_i_left = out_i_right = 0;

    float dv_slac = dv * npix_slack;
    int imax = 0; float max_val = -1e30f; bool found = false;
    for (int i = 0; i < SG_xpixels; i++) {
        if (std::fabs(dopp_slit[i] - guide_velocity) <= (velocity_range + dv_slac)) {
            if (spectrum[i] > max_val) { max_val = spectrum[i]; imax = i; found = true; }
        }
    }
    if (!found || max_val <= 0.0f) { fit_results[7] = FLAG_NO_LOCAL_MAX;  return 0; }
    if (std::fabs(dopp_slit[imax] - guide_velocity) > velocity_range) {
        fit_results[7] = FLAG_NO_LOCAL_MAX; return 0;
    }

    int i_left  = imax - npix;     if (i_left  < 0)          i_left  = 0;
    int i_right = imax + npix + 1; if (i_right > SG_xpixels) i_right = SG_xpixels;
    int n_pix   = i_right - i_left;
    if (n_pix < (2 * npix + 1)) {
        i_left  = imax - npix - 1; if (i_left  < 0)          i_left  = 0;
        i_right = imax + npix + 2; if (i_right > SG_xpixels) i_right = SG_xpixels;
        n_pix   = i_right - i_left;
    }
    out_i_left = i_left; out_i_right = i_right;
    ws.ensure_capacity(n_pix);

    int n_valid = 0;
    for (int i = 0; i < n_pix; i++) {
        ws.spectrum_win[i] = spectrum[i_left + i] / max_val;
        ws.dopp_win[i]     = dopp_slit[i_left + i];
        if (!std::isnan(ws.spectrum_win[i])) n_valid++;
    }
    if (n_valid < 3) { fit_results[7] = FLAG_NO_LOCAL_MAX; return 0; }

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

    /* config identical to C - no maxfev cap */
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

    int status = mpfit(myfunct_gaussian_deviates_with_derivatives_f32_opt,
                       n_valid, npar, best_params, ws.pars.data(), &config,
                       (void*)&pdata, &result);
    if (status <= 0) { fit_results[7] = FLAG_NO_CONVERGENCE; return 0; }

    int dof = n_valid - npar; if (dof <= 0) dof = 1;
    fit_results[0] = best_params[0] * max_val;
    fit_results[1] = best_params[1];
    fit_results[2] = best_params[2];
    fit_results[3] = ws.xerror_scipy[0] * max_val;
    fit_results[4] = ws.xerror_scipy[1];
    fit_results[5] = ws.xerror_scipy[2];
    fit_results[6] = result.bestnorm / (float)dof;
    fit_results[7] = FLAG_SUCCESS;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Batch C export (OpenMP)                                            */
/* ------------------------------------------------------------------ */
extern "C" {
void fit_spectra_batch_cpp(
    const float* spectra, const float* dopp_slits, int dopp_stride,
    const float* spec_noises, int noise_stride,
    const float* guide_velocities, int guide_stride,
    float velocity_range, int npix, int npix_slack, float dv, float width_min,
    int SG_xpixels, float amplitude_rel_min, float amplitude_rel_max,
    float width_max, float width_guess,
    float* fit_results, int* out_i_left, int* out_i_right, int B)
{
    #pragma omp parallel
    {
        FitWorkspace ws; ws.ensure_capacity(2 * npix + 2);
        #pragma omp for schedule(dynamic, 16)
        for (int b = 0; b < B; ++b) {
            int il=0, ir=0;
            fit_single_spectrum_cpp_core(
                &spectra[b*SG_xpixels], &dopp_slits[b*dopp_stride],
                &spec_noises[b*noise_stride], guide_velocities[b*guide_stride],
                velocity_range, npix, npix_slack, dv, width_min, SG_xpixels,
                amplitude_rel_min, amplitude_rel_max, width_max, width_guess,
                &fit_results[b*8], il, ir, ws);
            out_i_left[b]=il; out_i_right[b]=ir;
        }
    }
}
} /* extern "C" */

/* ------------------------------------------------------------------ */
/* Python C-API wrapper                                               */
/* ------------------------------------------------------------------ */
static PyObject *py_fit_single_spectrum_c(PyObject *self, PyObject *args, PyObject *keywds)
{
    (void)self;
    PyObject *so, *do_, *no_;
    float gv, vr, dv, wmin, armin, armax, wmax, wg;
    int npix, nslack, sgx;
    static const char *kw[] = {
        "spectrum","dopp_slit","spec_noise","guide_velocity","velocity_range",
        "npix","npix_slack","dv","width_min","SG_xpixels",
        "amplitude_rel_min","amplitude_rel_max","width_max","width_guess",NULL};
    if (!PyArg_ParseTupleAndKeywords(args,keywds,"OOOffiiffiffff",
            const_cast<char**>(kw),&so,&do_,&no_,&gv,&vr,&npix,&nslack,
            &dv,&wmin,&sgx,&armin,&armax,&wmax,&wg)) return NULL;

    PyArrayObject *sa=(PyArrayObject*)PyArray_FROM_OTF(so, NPY_FLOAT32,NPY_ARRAY_IN_ARRAY);
    PyArrayObject *da=(PyArrayObject*)PyArray_FROM_OTF(do_,NPY_FLOAT32,NPY_ARRAY_IN_ARRAY);
    PyArrayObject *na=(PyArrayObject*)PyArray_FROM_OTF(no_,NPY_FLOAT32,NPY_ARRAY_IN_ARRAY);
    if (!sa||!da||!na) {
        Py_XDECREF(sa); Py_XDECREF(da); Py_XDECREF(na);
        PyErr_SetString(PyExc_ValueError,"Failed to convert arrays to float32"); return NULL;
    }
    npy_intp dims[1]={8};
    PyArrayObject *ra=(PyArrayObject*)PyArray_SimpleNew(1,dims,NPY_FLOAT32);
    if (!ra) { Py_DECREF(sa); Py_DECREF(da); Py_DECREF(na); return NULL; }
    float *fr=(float*)PyArray_DATA(ra);
    int il=0, ir=0;
    FitWorkspace ws;
    fit_single_spectrum_cpp_core(
        (float*)PyArray_DATA(sa),(float*)PyArray_DATA(da),(float*)PyArray_DATA(na),
        gv,vr,npix,nslack,dv,wmin,sgx,armin,armax,wmax,wg,fr,il,ir,ws);
    Py_DECREF(sa); Py_DECREF(da); Py_DECREF(na);
    PyObject *mr=Py_BuildValue("(ii)",il,ir);
    PyObject *ret=PyTuple_Pack(2,(PyObject*)ra,mr);
    Py_DECREF(ra); Py_DECREF(mr);
    return ret;
}

static PyMethodDef module_methods[] = {
    {"_fit_single_spectrum_c",(PyCFunction)py_fit_single_spectrum_c,
     METH_VARARGS|METH_KEYWORDS,"Fit a single spectrum (v1-faithful, cmpfit)."},
    {NULL,NULL,0,NULL}};

static struct PyModuleDef mod = {
    PyModuleDef_HEAD_INIT,"fit_single_spectrum_ext",
    "C++ extension for Gaussian fitting (v1-faithful, cmpfit)",-1,module_methods};

PyMODINIT_FUNC PyInit_fit_single_spectrum_ext(void)
{ import_array(); return PyModule_Create(&mod); }
