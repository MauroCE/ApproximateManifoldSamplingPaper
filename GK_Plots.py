import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc
from copy import deepcopy
import numpy as np
from numpy import load

# Read Data
folder = 'GK_Experiment'
# m = 50
THUG00_CC_50 = load(folder + '/THUG00_CC_50.npy')
THUG00_AP_50 = load(folder + '/THUG00_AP_50.npy')
THUG09_CC_50 = load(folder + '/THUG09_CC_50.npy')
THUG09_AP_50 = load(folder + '/THUG09_AP_50.npy')
THUG99_CC_50 = load(folder + '/THUG99_CC_50.npy')
THUG99_AP_50 = load(folder + '/THUG99_AP_50.npy')
CRWM_CC_50 = load(folder + '/CRWM_CC_50.npy')
CRWM_AP_50 = load(folder + '/CRWM_AP_50.npy')
# m = 100
THUG00_CC_100 = load(folder + '/THUG00_CC_100.npy')
THUG00_AP_100 = load(folder + '/THUG00_AP_100.npy')
THUG09_CC_100 = load(folder + '/THUG09_CC_100.npy')
THUG09_AP_100 = load(folder + '/THUG09_AP_100.npy')
# THUG99_CC_100 = load(folder + '/THUG99_CC_100.npy')
# THUG99_AP_100 = load(folder + '/THUG99_AP_100.npy')
# CRWM_CC_100 = load(folder + '/CRWM_CC_100.npy')
# CRWM_AP_100 = load(folder + '/CRWM_AP_100.npy')
# Epsilons
EPSILONS = load(folder + '/EPSILONS.npy')


def show_only_positive_ap(cc, ap, ix):
    """USED FOR PLOTTING ONLY ESS WHERE WE HAD POSITIVE ACCEPTANCE PROBABILITY."""
    cc_copy = cc.copy()
    ap_copy = ap.copy()
    flag = ap_copy[:, ix] < 1e-8
    values = cc_copy[:, ix]
    values[flag] = np.nan
    return values

def show_only_positive_ap_crwm(out_cc, out_ap, ϵs, ix):
    """Same as above but for C-RWM."""
    cc_copy = deepcopy(out_cc.copy())
    ap_copy = deepcopy(out_ap.copy())
    flag = ap_copy < 1e-8
    cc_copy[flag] = np.nan
    return np.repeat(cc_copy[ix], len(ϵs))


# Settings for plot
rc('font',**{'family':'STIXGeneral'})
MARKERSIZE = 12
LINEWIDTH  = 2
MARKEREDGEWIDTH = 1.5
CRWM_LINEWIDTH=2
# Functions returning axes for different values of m.
m50  = (0)
m100 = (1)
m200 = (2)
fig, ax = plt.subplots(ncols=3, figsize=(18, 5), sharex=True, sharey=True)
### m = 50
# α = 0.0
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG00_CC_50, THUG00_AP_50, 0), label='1T', marker='o', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG00_CC_50, THUG00_AP_50, 1), label='10T', marker='o', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG00_CC_50, THUG00_AP_50, 2), label='50T', marker='o', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# α = 0.9
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG09_CC_50, THUG09_AP_50, 0), label='1α', marker='*', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG09_CC_50, THUG09_AP_50, 1), label='10α', marker='*', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG09_CC_50, THUG09_AP_50, 2), label='50α', marker='*', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# α = 0.99
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG99_CC_50, THUG99_AP_50, 0), label='1α2', marker='^', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG99_CC_50, THUG99_AP_50, 1), label='10α2', marker='^', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap(THUG99_CC_50, THUG99_AP_50, 2), label='100α2', marker='^', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# CRWM
ax[m50].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_50, CRWM_AP_50, EPSILONS, 0), label='1C', color='lightgray', lw=CRWM_LINEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_50, CRWM_AP_50, EPSILONS, 1), label='10C', color='darkgrey', lw=CRWM_LINEWIDTH)
ax[m50].plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_50, CRWM_AP_50, EPSILONS, 2), label='50C', color='dimgrey', lw=CRWM_LINEWIDTH)
### m = 100
# α = 0.0
ax[m100].plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, 0), label='1T', marker='o', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m100].plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, 1), label='10T', marker='o', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m100].plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, 2), label='50T', marker='o', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# α = 0.9
ax[m100].plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, 0), label='1α', marker='*', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m100].plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, 1), label='10α', marker='*', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax[m100].plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, 2), label='50α', marker='*', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# Prettify
for i in range(3):
    # Set xticks
    ax[i].set_xticks(EPSILONS)
    ax[i].tick_params(axis='both', which='major', labelsize=15)
    # Set labels
    ax[i].set_xlabel(r"$\mathregular{\epsilon}$", fontsize=20)
    # Log-log plot
    ax[i].loglog()
ax[0].set_ylabel("minESS/runtime", fontsize=20)

# Add a proper legend
triangle = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='^', linestyle='None', markersize=10, label='α=0.99')
star     = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='*', linestyle='None', markersize=10, label='α=0.9')
circle   = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='o', linestyle='None', markersize=10, label='α=0')
lightline  = mlines.Line2D([], [], color='lightgray', linestyle='-', label='B=1')
mediumline = mlines.Line2D([], [], color='darkgrey', linestyle='-', label='B=10')
darkline   = mlines.Line2D([], [], color='dimgrey', linestyle='-', label='B=50')
ax[0].legend(handles=[circle, star, triangle, lightline, mediumline, darkline], loc='lower right')
plt.tight_layout()
plt.show()