import matplotlib.pyplot as plt
import numpy as np
import itertools
from nengolib.stats import ball, sphere
from scipy.stats import ncx2, norm, chi
rng = np.random.default_rng(12345)

# SRD estimate with QMC sampling
def srd_qmc(n, nsamples, t):
    sum_chi = 0
    s = sphere.sample(nsamples, n)
    for i in range(0, nsamples):
        v = s[i]
        f = 1.0 - (t**2 * (1 - v[0]**2))
        if f > 0 and v[0] > 0:
            sum_chi += chi.cdf(t*v[0] + np.sqrt(f), n) - \
                chi.cdf(t*v[0] - np.sqrt(f), n)
    return sum_chi/nsamples

# Crude MC estimate
def crude_mc(n, nsamples, t):
    sum_mc = 0
    for _ in range(0, nsamples):
        v = rng.normal(0, 1, n)
        if np.linalg.norm(np.concatenate((v[0:1] - t, v[1:n]))) <= 1:
            sum_mc += 1.0
    return sum_mc/nsamples

# compute the various probability estimates
def compute_prob(n, trange, nsamples_srd, nsamples_mc):
    prob_exact = np.array([ncx2.cdf(1.0, n, t**2) for t in trange])
    prob_ldt_1 = np.array([norm.cdf(1.0 - t) for t in trange])
    prob_ldt_2 = np.array([norm.cdf(1.0 - t)*t**(0.5*(1 - n)) for t in trange])
    prob_srd_qmc = np.array([[srd_qmc(n, nsamples, t)
                            for t in trange] for nsamples in nsamples_srd])
    prob_crude_mc = np.array([[crude_mc(n, nsamples, t)
                            for t in trange] for nsamples in nsamples_mc])
    return prob_exact, prob_ldt_1, prob_ldt_2, prob_srd_qmc, prob_crude_mc

# make the plot
# n = dimension of the uncertainty
def compare_plot(n):
    trange = np.arange(1., 7.1, 0.1)
    nsamples_srd = [100] if n < 10 else [10000]
    nsamples_mc = [10000]
    prob_exact, prob_ldt_1, prob_ldt_2, prob_srd_qmc, prob_crude_mc = compute_prob(
        n, trange, nsamples_srd, nsamples_mc)
    fig, axs = plt.subplots(1, 6, figsize=(
        5*6, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.2, wspace=.2)
    axs = axs.ravel()
    marker = itertools.cycle(('d', 'x', 'v', 'o', '*'))
    for i in range(6):
        inds = range((i*10), (i*10) + 11)
        axs[i].plot(trange[inds], prob_exact[inds],
                    marker=next(marker), label="Exact")
        axs[i].plot(trange[inds], prob_ldt_1[inds],
                    marker=next(marker), label="LDT-1")
        axs[i].plot(trange[inds], prob_ldt_2[inds],
                    marker=next(marker), label="LDT-2")
        for j in range(len(nsamples_srd)):
            axs[i].plot(trange[inds], prob_srd_qmc[j][inds],
                        marker=next(marker), label="SRD-QMC-" + str(nsamples_srd[j]))
        for j in range(len(nsamples_mc)):
            axs[i].plot(trange[inds], prob_crude_mc[j][inds],
                        marker=next(marker), label="CrudeMC-" + str(nsamples_mc[j]))
        axs[i].set_yscale('log')
        axs[i].set_xlabel('t', fontsize=14)
        axs[i].tick_params(axis='both', which='major', labelsize=14)
        handles, labels = axs[i].get_legend_handles_labels()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.1)
    fig.legend(handles, labels, loc='upper center', ncol=5, prop={'size': 14})
    fig.savefig('fig-' + str(n) + '.png')


compare_plot(2)
compare_plot(10)
