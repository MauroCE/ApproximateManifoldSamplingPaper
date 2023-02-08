import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc
from copy import deepcopy
import numpy as np
from numpy import load, loadtxt
import os
from scipy.special import ndtr
import seaborn as sns


# Read Data
mainfolder = 'GK_Experiment'
subfolder = '1122_2233_3344_4455' #'1723_1923_6482_8921'
folder = os.path.join(mainfolder, subfolder)
# Save figure
savefolder = 'images'
# m = 50
# THUG00_CC_50 = load(os.path.join(folder, 'THUG00_CC_50.npy'))
# THUG00_AP_50 = load(os.path.join(folder, 'THUG00_AP_50.npy'))
# THUG09_CC_50 = load(os.path.join(folder, 'THUG09_CC_50.npy'))
# THUG09_AP_50 = load(os.path.join(folder, 'THUG09_AP_50.npy'))
# THUG99_CC_50 = load(os.path.join(folder, 'THUG99_CC_50.npy'))
# THUG99_AP_50 = load(os.path.join(folder, 'THUG99_AP_50.npy'))
# CRWM_CC_50 = load(os.path.join(folder, 'CRWM_CC_50.npy'))
# CRWM_AP_50 = load(os.path.join(folder, 'CRWM_AP_50.npy'))
# # m = 100
# THUG00_CC_100 = load(os.path.join(folder, 'THUG00_CC_100.npy'))
# THUG00_AP_100 = load(os.path.join(folder, 'THUG00_AP_100.npy'))
# THUG09_CC_100 = load(os.path.join(folder, 'THUG09_CC_100.npy'))
# THUG09_AP_100 = load(os.path.join(folder, 'THUG09_AP_100.npy'))
# THUG99_CC_100 = load(os.path.join(folder, 'THUG99_CC_100.npy'))
# THUG99_AP_100 = load(os.path.join(folder, 'THUG99_AP_100.npy'))
# CRWM_CC_100 = load(os.path.join(folder, 'CRWM_CC_100.npy'))
# CRWM_AP_100 = load(os.path.join(folder, 'CRWM_AP_100.npy'))
# # m = 200
# THUG00_CC_200 = load(os.path.join(folder, 'THUG00_CC_200.npy'))
# THUG00_AP_200 = load(os.path.join(folder, 'THUG00_AP_200.npy'))
# THUG09_CC_200 = load(os.path.join(folder, 'THUG09_CC_200.npy'))
# THUG09_AP_200 = load(os.path.join(folder, 'THUG09_AP_200.npy'))
# THUG99_CC_200 = load(os.path.join(folder, 'THUG99_CC_200.npy'))
# THUG99_AP_200 = load(os.path.join(folder, 'THUG99_AP_200.npy'))
# CRWM_CC_200 = load(os.path.join(folder, 'CRWM_CC_200.npy'))
# CRWM_AP_200 = load(os.path.join(folder, 'CRWM_AP_200.npy'))
# # Epsilons
# EPSILONS = load(os.path.join(folder, 'EPSILONS.npy'))


# def show_only_positive_ap(cc, ap, ix):
#     """USED FOR PLOTTING ONLY ESS WHERE WE HAD POSITIVE ACCEPTANCE PROBABILITY."""
#     cc_copy = cc.copy()
#     ap_copy = ap.copy()
#     flag = ap_copy[:, ix] < 1e-8
#     values = cc_copy[:, ix]
#     values[flag] = np.nan
#     return values

# def show_only_positive_ap_crwm(out_cc, out_ap, ϵs, ix):
#     """Same as above but for C-RWM."""
#     cc_copy = deepcopy(out_cc.copy())
#     ap_copy = deepcopy(out_ap.copy())
#     flag = ap_copy < 1e-8
#     cc_copy[flag] = np.nan
#     return np.repeat(cc_copy[ix], len(ϵs))


# # PLOT COMPUTATIONAL COST
# rc('font',**{'family':'STIXGeneral'})
# MARKERSIZE = 12
# LINEWIDTH  = 2
# MARKEREDGEWIDTH = 1.5
# CRWM_LINEWIDTH=2
# # Functions returning axes for different values of m.
# m50  = (0)
# m100 = (1)
# m200 = (2)
# fig, ax = plt.subplots(ncols=3, figsize=(18, 5), sharex=True, sharey=True)
# ### m = 50
# # α = 0.0
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG00_CC_50, THUG00_AP_50, 0), label='1T', marker='o', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG00_CC_50, THUG00_AP_50, 1), label='10T', marker='o', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG00_CC_50, THUG00_AP_50, 2), label='50T', marker='o', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # α = 0.9
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG09_CC_50, THUG09_AP_50, 0), label='1α', marker='*', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG09_CC_50, THUG09_AP_50, 1), label='10α', marker='*', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG09_CC_50, THUG09_AP_50, 2), label='50α', marker='*', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # α = 0.99
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG99_CC_50, THUG99_AP_50, 0), label='1α2', marker='^', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG99_CC_50, THUG99_AP_50, 1), label='10α2', marker='^', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap(THUG99_CC_50, THUG99_AP_50, 2), label='100α2', marker='^', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # CRWM
# ax[m50].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_50, CRWM_AP_50, EPSILONS, 0), label='1C', color='lightgray', lw=CRWM_LINEWIDTH, marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_50, CRWM_AP_50, EPSILONS, 1), label='10C', color='darkgrey', lw=CRWM_LINEWIDTH, marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH)
# ax[m50].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_50, CRWM_AP_50, EPSILONS, 2), label='50C', color='dimgrey', lw=CRWM_LINEWIDTH, marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH)
# ### m = 100
# # α = 0.0
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, 0), label='1T', marker='o', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, 1), label='10T', marker='o', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, 2), label='50T', marker='o', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # α = 0.9
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, 0), label='1α', marker='*', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, 1), label='10α', marker='*', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, 2), label='50α', marker='*', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # α = 0.99
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG99_CC_100, THUG99_AP_100, 0), label='1α', marker='^', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG99_CC_100, THUG99_AP_100, 1), label='10α', marker='^', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap(THUG99_CC_100, THUG99_AP_100, 2), label='50α', marker='^', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # C-RWM
# ax[m100].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_100, CRWM_AP_100, EPSILONS, 0), label='1C', marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH, color='lightgray', lw=CRWM_LINEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_100, CRWM_AP_100, EPSILONS, 1), label='10C', marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH, color='darkgrey', lw=CRWM_LINEWIDTH)
# ax[m100].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_100, CRWM_AP_100, EPSILONS, 2), label='50C', marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH, color='dimgrey', lw=CRWM_LINEWIDTH)
# ### m = 200
# # α = 0.0
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG00_CC_200, THUG00_AP_200, 0), label='1T', marker='o', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG00_CC_200, THUG00_AP_200, 1), label='10T', marker='o', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG00_CC_200, THUG00_AP_200, 2), label='50T', marker='o', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # α = 0.9
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG09_CC_200, THUG09_AP_200, 0), label='1T', marker='*', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG09_CC_200, THUG09_AP_200, 1), label='10T', marker='*', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG09_CC_200, THUG09_AP_200, 2), label='50T', marker='*', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # α = 0.99
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG99_CC_200, THUG99_AP_200, 0), label='1T', marker='^', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG99_CC_200, THUG99_AP_200, 1), label='10T', marker='^', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap(THUG99_CC_200, THUG99_AP_200, 2), label='50T', marker='^', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# # C-RWM
# ax[m200].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_200, CRWM_AP_200, EPSILONS, 0), label='1C', marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH, color='lightgray', lw=CRWM_LINEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_200, CRWM_AP_200, EPSILONS, 1), label='10C', marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH, color='darkgrey', lw=CRWM_LINEWIDTH)
# ax[m200].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_200, CRWM_AP_200, EPSILONS, 2), label='50C', marker='x', markersize=MARKERSIZE, markeredgecolor='k', markeredgewidth=MARKEREDGEWIDTH, color='dimgrey', lw=CRWM_LINEWIDTH)
# # Prettify
# for i in range(3):
#     # Set xticks
#     ax[i].set_xticks(EPSILONS)
#     ax[i].tick_params(axis='both', which='major', labelsize=15)
#     # Set labels
#     ax[i].set_xlabel(r"$\mathregular{\epsilon}$", fontsize=20)
#     # Log-log plot
#     ax[i].loglog()
# ax[0].set_ylabel("minESS/runtime", fontsize=20)

# # Add a proper legend
# triangle = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='^', linestyle='None', markersize=10, label='α=0.99')
# star     = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='*', linestyle='None', markersize=10, label='α=0.9')
# circle   = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='o', linestyle='None', markersize=10, label='α=0')
# cross    = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='x', linestyle='None', markersize=10, label='C-RWM')
# lightline  = mlines.Line2D([], [], color='lightgray', linestyle='-', label='B=1')
# mediumline = mlines.Line2D([], [], color='darkgrey', linestyle='-', label='B=10')
# darkline   = mlines.Line2D([], [], color='dimgrey', linestyle='-', label='B=50')
# ax[0].legend(handles=[circle, star, triangle, cross, lightline, mediumline, darkline], loc='lower right')
# plt.tight_layout()
# # plt.savefig(os.path.join(savefolder, '_'.join(['GK_CC', subfolder + '.png'])))
# plt.show()



# PLOT BEST EPSILON
# Read data
RWM_BE     = load(os.path.join(folder, 'RWM_BEST_EPSILON_SAMPLES.npy'))
THUG00_BE  = load(os.path.join(folder, 'THUG00_BEST_EPSILON_SAMPLES.npy'))
THUG999_BE = load(os.path.join(folder, 'THUG999_BEST_EPSILON_SAMPLES.npy'))
CRWM_BE    = load(os.path.join(folder, 'CRWM_BEST_EPSILON_SAMPLES.npy'))
PRANGLE    = loadtxt('gk_mcmc_samples.txt', skiprows=1, usecols=range(1, 5))[1:, :]

# Transform from N(0, 1) to U(0, 10) space.
G = lambda varθ: 10*ndtr(varθ)
sRWM_be_transformed_θ = G(RWM_BE[:, :4])
sTHUG00_be_transformed_θ = G(THUG00_BE[:, :4])
sTHUG999_be_transformed_θ = G(THUG999_BE[:, :4])
sCRWM_be_transformed_θ = G(CRWM_BE[:, :4])
# True parameter value
θ0 = np.array([3.0, 1.0, 2.0, 0.5])

BW = 0.3
LW = 3
FROM = 0
LABELS = ['a', 'b', 'g', 'k']
COLOR_THUG = "#1C110A"  # Dark Grey
COLOR_HUG  = "#E9B44C"  # Dark Yellow
COLOR_CRWM = "#50A2A7"  # Light Blue
COLOR_RWM  = "#9B2915"  # Light Red
COLOR_PRA  = "#E4D6A7"  # Light Yellow
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
for i in range(4):
    _ = sns.kdeplot(sRWM_be_transformed_θ[FROM:, i], bw_method=BW, ax=ax[i//2, i%2], lw=LW, label='RWM', color=COLOR_RWM)
    _ = sns.kdeplot(sTHUG00_be_transformed_θ[FROM:, i], bw_method=BW, ax=ax[i//2, i%2], lw=LW, label='THUG', color=COLOR_HUG)
    _ = sns.kdeplot(sTHUG999_be_transformed_θ[FROM:, i], bw_method=BW, ax=ax[i//2, i%2], lw=LW, label='αTHUG', color=COLOR_THUG)
    _ = sns.kdeplot(sCRWM_be_transformed_θ[FROM:, i], bw_method=BW, ax=ax[i//2, i%2], lw=LW, label='CRWM', color=COLOR_CRWM)
    _ = sns.kdeplot(PRANGLE[FROM:, i], bw_method=BW, ax=ax[i//2, i%2], lw=LW, label='MCMC', color=COLOR_PRA)
    min_val = min(sTHUG00_be_transformed_θ[FROM:, i].min(), sTHUG999_be_transformed_θ[FROM:, i].min(), sCRWM_be_transformed_θ[FROM:, i].min()) - 0.1
    max_val = max(sTHUG00_be_transformed_θ[FROM:, i].max(), sTHUG999_be_transformed_θ[FROM:, i].max(), sCRWM_be_transformed_θ[FROM:, i].max()) + 0.1
    ax[i//2, i%2].axvline(θ0[i], c='k', ls='--')
    ax[i//2, i%2].text(0.1 if (i%2 == 0) else 0.9, 0.9, "{}".format(LABELS[i]), transform=ax[i //2, i % 2].transAxes, fontsize=25, fontfamily='STIXGeneral')
    ax[i//2, i%2].set_ylabel("")
    ax[i//2, i%2].set_yticks([])
    ax[i//2, i%2].set_xticks([])

plt.tight_layout()
plt.legend(prop={'family':'STIXGeneral', 'size':18}, loc='lower right')
# plt.savefig("images/GK_best_epsilon.png")
plt.show()