# Dirichlet Process Mixture Model Example
# Adapted from https://docs.pymc.io/notebooks/dp_mix.html

from __future__ import division

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

SEED = 5132290 # from random.org

np.random.seed(SEED)

# this is the original stick_breaking function provided in link above
# I think it doesn't guarantee the weights will sum to 1?
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining

# my modified stick_breaking function
def stick_breaking(v_km1):
    w_km1 = v_km1 * tt.concatenate([[1], tt.extra_ops.cumprod(1 - v_km1)[:-1]])
    
    w_last = tt.ones_like(v[0]) - tt.sum(w_km1, axis = 0, keepdims = True)
    w_last = tt.maximum(w_last, tt.zeros_like(w_last))
    w_full = tt.concatenate([w_km1, w_last], axis = 0)
    
    return w_full

old_faithful_df = pd.read_csv('old_faithful.csv')
old_faithful_df['std_waiting'] = (old_faithful_df.waiting - old_faithful_df.waiting.mean()) / old_faithful_df.waiting.std()

N = old_faithful_df.shape[0]
K = 30

with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    v = pm.Beta('v', 1., alpha, shape=K-1)
    w = pm.Deterministic('w', stick_breaking(v))
    
    # what is the lambda * tau parameterization doing for us?
    tau = pm.Gamma('tau', 1., 1., shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                           observed=old_faithful_df.std_waiting.values)
    
    # I've changed this to Metropolis with burnin of 1000 instead of NUTS
    # Metropolis is faster
    step = pm.Metropolis()
    trace = pm.sample(1000, step, tune=1000, random_seed=SEED)


# I haven't fully worked through what's going on with the axes here yet,
# but basically I think expand across mixture components, multiply by weights,
# and sum again?
x_plot = np.linspace(-3, 3, 200)
post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x_plot),
                                      trace['mu'][:, np.newaxis, :],
                                      1. / np.sqrt(trace['lambda'] * trace['tau'])[:, np.newaxis, :])
post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)

post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)



# make plot - minor modifications from original code
fig, ax = plt.subplots(figsize=(8, 6))

n_bins = 20
ax.hist(old_faithful_df.std_waiting.values, bins=n_bins, density=True,
        color='blue', lw=0, alpha=0.5);

ax.fill_between(x_plot, post_pdf_low, post_pdf_high,
                color='gray');
ax.plot(x_plot, post_pdfs[0],
        c='gray', label='Posterior sample densities');
ax.plot(x_plot, post_pdfs[::100].T, c='gray');
ax.plot(x_plot, post_pdfs.mean(axis=0),
        c='k', label='Posterior expected density');

ax.set_xlabel('Standardized waiting time between eruptions');

ax.set_yticklabels([]);
ax.set_ylabel('Density');

ax.legend(loc=2);

plt.draw()
plt.show()
