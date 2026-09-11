/*
 * gaussfit-rs diagnostic, not part of the upstream fastfit2 sources.
 *
 * Exposes mpfit's iteration and evaluation counts for the unmodified C++
 * single-spectrum core. The vendored fit_single_spectrum_ext.cpp is included
 * with `mpfit` renamed, so the core's one solver call lands in the shim below,
 * which records mp_result.niter and mp_result.nfev before returning. Every
 * other line the fit executes is the vendored one, compiled with the same
 * flags, so the counts are those of the reference, not of a copy.
 */
#define mpfit mpfit_counted
#include "fit_single_spectrum_ext.cpp"
#undef mpfit

extern "C" int mpfit(mp_func funct, int m, int npar, float *xall, mp_par *pars,
                     mp_config *config, void *private_data, mp_result *result);

static thread_local int last_niter = -1;
static thread_local int last_nfev = -1;

extern "C" int mpfit_counted(mp_func funct, int m, int npar, float *xall, mp_par *pars,
                             mp_config *config, void *private_data, mp_result *result)
{
    int status = mpfit(funct, m, npar, xall, pars, config, private_data, result);
    last_niter = result->niter;
    last_nfev = result->nfev;
    return status;
}

/* Same arguments and first two return values as _fit_single_spectrum_c, plus
 * (niter, nfev); both are -1 when the core returned before calling mpfit. */
static PyObject *py_fit_single_spectrum_counts(PyObject *self, PyObject *args, PyObject *keywds)
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
    last_niter = last_nfev = -1;
    fit_single_spectrum_cpp_core(
        (float*)PyArray_DATA(sa),(float*)PyArray_DATA(da),(float*)PyArray_DATA(na),
        gv,vr,npix,nslack,dv,wmin,sgx,armin,armax,wmax,wg,fr,il,ir,ws);
    Py_DECREF(sa); Py_DECREF(da); Py_DECREF(na);
    PyObject *ret=Py_BuildValue("(N(ii)(ii))",(PyObject*)ra,il,ir,last_niter,last_nfev);
    return ret;
}

static PyMethodDef counts_methods[] = {
    {"_fit_single_spectrum_counts",(PyCFunction)py_fit_single_spectrum_counts,
     METH_VARARGS|METH_KEYWORDS,
     "Fit a single spectrum with the vendored C++ core; also returns (niter, nfev)."},
    {NULL,NULL,0,NULL}};

static struct PyModuleDef counts_mod = {
    PyModuleDef_HEAD_INIT,"fit_counts_ext",
    "gaussfit-rs diagnostic: fastfit2 single-spectrum fit with solver counts",-1,counts_methods};

PyMODINIT_FUNC PyInit_fit_counts_ext(void)
{ import_array(); return PyModule_Create(&counts_mod); }
