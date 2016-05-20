import glob
import os
import platform

from cffi import FFI

include_dirs = [os.path.join('..', 'src'),
                os.path.join('..', 'include')]

rmath_src = glob.glob(os.path.join('..', 'src', '*.c'))

# Take out dSFMT dependant files; Just use the basic rng
rmath_src = [f for f in rmath_src if ('librandom.c' not in f) and ('randmtzig.c' not in f)]

extra_compile_args = ['-DMATHLIB_STANDALONE']
if platform.system() == 'Windows':
    extra_compile_args.append('-std=c99')

ffi = FFI()
ffi.set_source('_rmath_ffi', '#include <Rmath.h>',
               include_dirs=include_dirs,
               sources=rmath_src,
               libraries=[],
               extra_compile_args=extra_compile_args)

# This is an incomplete list of the available functions in Rmath
# but these are sufficient for our example purposes and gives a sense of
# the types of functions we can get
ffi.cdef('''\
// Normal Distribution
double dnorm(double, double, double, int);
double pnorm(double, double, double, int, int);

// Uniform Distribution
double dunif(double, double, double, int);
double punif(double, double, double, int, int);

// Gamma Distribution
double dgamma(double, double, double, int);
double pgamma(double, double, double, int, int);
''')

if __name__ == '__main__':
    # Normally set verbose to `True`, but silence output
    # for reduced notebook noise
    ffi.compile(verbose=False)
