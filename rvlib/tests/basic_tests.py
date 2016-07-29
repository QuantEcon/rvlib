import rvlib as rl
import scipy.stats as st
import numpy as np

# Get random points to evaluate functions
np.random.seed(1234)
x = np.random.rand(10)

# Create normal distrtibution
N_rl = rl.Normal(0,1)
N_st = st.norm(0,1)

# Check normal cdfs/pdfs against each other
N_rl_cdf = N_rl.cdf(x)
N_st_cdf = N_st.cdf(x)
np.allclose(N_rl_cdf, N_st_cdf)

# Create chi2 distributions
chi2_rl = rl.Chisq(5)
chi2_st = st.chi2(5)

# Check chi2 cdfs/pdfs against each other
chi2_rl_cdf = chi2_rl.cdf(x)
chi2_st_cdf = chi2_st.cdf(x)
np.allclose(chi2_rl_cdf, chi2_st_cdf)