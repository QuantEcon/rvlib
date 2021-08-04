
# ifdef _WIN32
#   define __USE_MINGW_ANSI_STDIO 1
# endif

#include <Rmath.h>
// for F77_NAME() :
#include <R_ext/RS.h>

// for SEXP:
#include <Rinternals.h>

/* Instead of inside R
 * #include "nmath.h"
 ------------------- */
// MM_R our substitute for the above header , from R's source
// -------- excerpt from  nmath.h ---------------------------------
#include <R_ext/Error.h>
# define MATHLIB_ERROR(fmt,x)		error(fmt,x);
# define MATHLIB_WARNING(fmt,x)		warning(fmt,x)
# define MATHLIB_WARNING2(fmt,x,x2)	warning(fmt,x,x2)
# define MATHLIB_WARNING3(fmt,x,x2,x3)	warning(fmt,x,x2,x3)
# define MATHLIB_WARNING4(fmt,x,x2,x3,x4) warning(fmt,x,x2,x3,x4)
# define MATHLIB_WARNING5(fmt,x,x2,x3,x4,x5) warning(fmt,x,x2,x3,x4,x5)

#include <R_ext/Arith.h>
#define ML_POSINF	R_PosInf
#define ML_NEGINF	R_NegInf
#define ML_NAN		R_NaN

#ifdef ENABLE_NLS
#include <libintl.h>
#define _(String) gettext (String)
#else
#define _(String) (String)
#endif


#define ML_VALID(x)	(!ISNAN(x))

#define ME_NONE		0
/*	no error */
#define ME_DOMAIN	1
/*	argument out of domain */
#define ME_RANGE	2
/*	value out of range */
#define ME_NOCONV	4
/*	process did not converge */
#define ME_PRECISION	8
/*	does not have "full" precision */
#define ME_UNDERFLOW	16
/*	and underflow occured (important for IEEE)*/

/* FIXME: 'ERR' and 'ERROR' below are misnomers, both in R and stand-alone Rmathlib:
   -----  they only *warn* after all; they are private headers, so can change anytime
*/

#define ML_ERR_return_NAN { ML_ERROR(ME_DOMAIN, ""); return ML_NAN; }

/* For a long time prior to R 2.3.0 ML_ERROR did nothing.
   We don't report ME_DOMAIN errors as the callers collect ML_NANs into
   a single warning.
 */
#define ML_ERROR(x, s) { \
   if(x > ME_DOMAIN) { \
       char *msg = ""; \
       switch(x) { \
       case ME_DOMAIN: \
	   msg = _("argument out of domain in '%s'\n");	\
	   break; \
       case ME_RANGE: \
	   msg = _("value out of range in '%s'\n");	\
	   break; \
       case ME_NOCONV: \
	   msg = _("convergence failed in '%s'\n");	\
	   break; \
       case ME_PRECISION: \
	   msg = _("full precision may not have been achieved in '%s'\n"); \
	   break; \
       case ME_UNDERFLOW: \
	   msg = _("underflow occurred in '%s'\n");	\
	   break; \
       } \
       MATHLIB_WARNING(msg, s); \
   } \
}



#define ML_WARN_return_NAN { ML_WARNING(ME_DOMAIN, ""); return ML_NAN; }

/* For a long time prior to R 2.3.0 ML_WARNING did nothing.
   We don't report ME_DOMAIN errors as the callers collect ML_NANs into
   a single warning.
 */
#define ML_WARNING(x, s) { \
   if(x > ME_DOMAIN) { \
       char *msg = ""; \
       switch(x) { \
       case ME_DOMAIN: \
	   msg = _("argument out of domain in '%s'\n");	\
	   break; \
       case ME_RANGE: \
	   msg = _("value out of range in '%s'\n");	\
	   break; \
       case ME_NOCONV: \
	   msg = _("convergence failed in '%s'\n");	\
	   break; \
       case ME_PRECISION: \
	   msg = _("full precision may not have been achieved in '%s'\n"); \
	   break; \
       case ME_UNDERFLOW: \
	   msg = _("underflow occurred in '%s'\n");	\
	   break; \
       } \
       MATHLIB_WARNING(msg, s); \
   } \
}

// --------- MM_R end_of { #include <nmath.h> substitute } ----------------------


/* R's  #include <config.h> typically defines this
 *                          (it may be very slow on Solaris):
 *
 * Define if you wish to use the 'long double' type.
 */
#define HAVE_LONG_DOUBLE 1
/* no longer needed :------
#ifndef _WIN32
#define HAVE_LONG_DOUBLE 1
#else // Windows:
#  ifdef _WIN64
#    define HAVE_LONG_DOUBLE 1
#  else // Windows 32 bit
#    define HAVE_LONG_DOUBLE 1
#  endif
#endif
*/

/* Required by C99, but might be slow */
#ifdef HAVE_LONG_DOUBLE
# define LDOUBLE long double
#else
# define LDOUBLE double
#endif

#ifdef HAVE_LONG_DOUBLE
# define EXP expl
# define EXPm1 expm1l
# define FABS fabsl
# define LOG logl
# define LOG1p log1pl
// Rmpfr: log(mpfr(2, 130)) {130 bits is "more than enough": most long_double are just 80 bits!}
# define M_LN2_ 0.6931471805599453094172321214581765680755L
# define PR_g_ "Lg"
# ifdef _WIN32 // all of Windows (such that "%Lg" works)
#   define __USE_MINGW_ANSI_STDIO 1
# endif

#else //--------------------

# define EXP exp
# define EXPm1 expm1
# define FABS fabs
# define LOG log
# define LOG1p log1p
# define M_LN2_ M_LN2
# define PR_g_ "g"

#endif



#include "dpq.h"
//        =====

/* Type 'logical':
   1) use for .C() called functions for clarity
   2) remember which were Fortran logicals (now in C)
*/
typedef int logical;


// qchisq_appr.c : -------------------------------------------------------------

void qchisq_appr_v(double *P, int *n, double *nu, double *tol,
		   logical *lower_tail, logical *log_p,
		   /* result: */ double *q)
    ;

// pnchisq-it.c : --------------------------------------------------------------

void Pnchisq_it(double *x, double *f, double *theta,
		/* FIXME?? additionally vectorize in (x, prob) or (x,f,th) |-> prob */
		double *errmax, double *reltol, int *itrmax, int *verbose,
		int *i_0, int *n_terms, double *terms, double *prob)
    ;
SEXP Pnchisq_R(SEXP x_, SEXP f_, SEXP theta_,
	       SEXP lower_tail_, SEXP log_p_,
	       SEXP no_2nd_call_,
	       SEXP cutoff_ncp_, SEXP small_ncp_logspace_, SEXP it_simple_,
	       SEXP errmax_, SEXP reltol_, SEXP epsS_, SEXP itrmax_, SEXP verbose_)
    ;


// 310-pnbeta.c : --------------------------------------------------------------

void ncbeta(double *a, double *b, double *lambda, double *x, int *n,
	    int *use_AS226,
	    double *errmax, int *itrmax, int *ifault, double *res)
    ;

// ppois-direct.c : ------------------------------------------------------------

SEXP chk_LDouble(SEXP lambda_, SEXP verbose_, SEXP tol_)
    ;
SEXP ppoisD(SEXP X, SEXP lambda_, SEXP all_from_0_, SEXP verbose_)
    ;

// wienergerm_nchisq.c : -------------------------------------------------------

// TODO: export h() function ? (with longer name)?

double nonc_chi(double x, double ncp, double df, int lower_tail, int log_p,
		int variant);
// Called via .C():
void pchisqV(double *x, int *n, /* vectorized in x : x[1..n] : */
	     double *ncp, double *df,
	     logical *lower_tail, logical *log_p, int *variant)
    ;

// wienergerm_nchisq_F.f : -----------------------------------------------------
int F77_NAME(noncechi)(int *variant,
		       double *argument, double *noncentr, double *df, double *p,
		       int *ifault);


// algdiv.c: --------------------------------------------------------------------

double algdiv(double a, double b);
// .Call()ed :
SEXP R_algdiv(SEXP a_, SEXP b_)
    ;

// bd0.c: --------------------------------------------------------------------
double bd0(double x, double np, double delta, int maxit, int trace);
void  ebd0(double x, double M, double *yh, double *yl);

SEXP dpq_bd0(SEXP x, SEXP np, SEXP delta,
	     SEXP maxit, SEXP version, SEXP trace);


// logcf.c: --------------------------------------------------------------------
SEXP R_logcf(SEXP x_, SEXP i_, SEXP d_, SEXP eps_, SEXP trace_);
/*
 */


// lgammacor.c : -------------------------------------------------------------
double dpq_lgammacor(double x, int nalgm, double xbig);
SEXP     R_lgammacor(SEXP x_, SEXP nalgm_, SEXP xbig_);

// chebyshev.c : -------------------------------------------------------------

/* Chebyshev Polynomial */
int	chebyshev_init(const double[], int, double);
double	chebyshev_eval(double, const double[], const int);

SEXP R_chebyshev_eval(SEXP x_, SEXP a_, SEXP n_);
SEXP R_chebyshev_init(SEXP coef_, SEXP eta_);


// DPQ-misc.c: --------------------------------------------------------------------

// 1. Functions from R's  C API  Rmath.h  -- not (yet) existing as base R functions

SEXP R_log1pmx(SEXP x_);
/* double log1pmx (double X)
     Computes 'log(1 + X) - X' (_log 1 plus x minus x_), accurately even
     for small X, i.e., |x| << 1.
*/

SEXP R_log1pexp(SEXP x_);
/* double log1pexp (double X)
     Computes 'log(1 + exp(X))' (_log 1 plus exp_), accurately, notably
     for large X, e.g., x > 720.
*/

SEXP R_log1mexp(SEXP x_);
/* double log1mexp (double X)
     Computes 'log(1 - exp(-X))' (_log 1 minus exp_), accurately,
     carefully for two regions of X, optimally cutting off at log 2 (=
     0.693147..), using '((-x) > -M_LN2 ? log(-expm1(-x)) :
     log1p(-exp(-x)))'.
*/

SEXP R_lgamma1p(SEXP x_);
/* double lgamma1p (double X)
     Computes 'log(gamma(X + 1))' (_log(gamma(1 plus x))_), accurately
     even for small X, i.e., 0 < x < 0.5.
*/

SEXP R_frexp(SEXP x_);
// returns list(r = <double>, e = <integer>) where  x = r * 2^e , r in [0.5, 1) and integer e

SEXP R_ldexp(SEXP f_, SEXP E_);
// ldexp(f, E) := f * 2^E
