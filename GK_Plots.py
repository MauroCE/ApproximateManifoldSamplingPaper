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
THUG99_CC_100 = load(folder + '/THUG99_CC_100.npy')
THUG99_AP_100 = load(folder + '/THUG99_AP_100.npy')
CRWM_CC_100 = load(folder + '/CRWM_CC_100.npy')
CRWM_AP_100 = load(folder + '/CRWM_AP_100.npy')
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


rc('font',**{'family':'STIXGeneral'})
MARKERSIZE = 12
LINEWIDTH  = 2
MARKEREDGEWIDTH = 1.5
CRWM_LINEWIDTH=2
fig, ax = plt.subplots(figsize=(16, 8))
# α = 0.0
ax.plot(EPSILONS, show_only_positive_ap(THUG00_CC, THUG00_AP, 0), label='1T', marker='o', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap(THUG00_CC, THUG00_AP, 1), label='10T', marker='o', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap(THUG00_CC, THUG00_AP, 2), label='50T', marker='o', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# α = 0.9
ax.plot(EPSILONS, show_only_positive_ap(THUG09_CC, THUG09_AP, 0), label='1α', marker='*', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap(THUG09_CC, THUG09_AP, 1), label='10α', marker='*', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap(THUG09_CC, THUG09_AP, 2), label='50α', marker='*', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# α = 0.99
ax.plot(EPSILONS, show_only_positive_ap(THUG99_CC, THUG99_AP, 0), label='1α2', marker='^', color='lightgray', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap(THUG99_CC, THUG99_AP, 1), label='10α2', marker='^', color='darkgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap(THUG99_CC, THUG99_AP, 2), label='100α2', marker='^', color='dimgrey', markersize=MARKERSIZE, markeredgecolor='k', lw=LINEWIDTH, markeredgewidth=MARKEREDGEWIDTH)
# CRWM
ax.plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC, CRWM_AP, EPSILONS, 0), label='1C', color='lightgray', lw=CRWM_LINEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC, CRWM_AP, EPSILONS, 1), label='10C', color='darkgrey', lw=CRWM_LINEWIDTH)
ax.plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC, CRWM_AP, EPSILONS, 2), label='50C', color='dimgrey', lw=CRWM_LINEWIDTH)
# Set xticks
ax.set_xticks(EPSILONS)
# Set labels
ax.set_xlabel("ϵ", fontsize=20)
ax.set_ylabel("minESS/runtime", fontsize=20)
ax.loglog()
# Add a proper legend
triangle = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='^', linestyle='None', markersize=10, label='α=0.99')
star     = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='*', linestyle='None', markersize=10, label='α=0.9')
circle   = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='o', linestyle='None', markersize=10, label='α=0')
lightline  = mlines.Line2D([], [], color='lightgray', linestyle='-', label='B=1')
mediumline = mlines.Line2D([], [], color='darkgrey', linestyle='-', label='B=10')
darkline   = mlines.Line2D([], [], color='dimgrey', linestyle='-', label='B=50')
ax.legend(handles=[circle, star, triangle, lightline, mediumline, darkline], loc='lower right')
plt.show()