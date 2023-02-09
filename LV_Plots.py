from numpy import nan, load, repeat
from copy import deepcopy
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc


def show_only_positive_ap(cc, ap, ix):
    """USED FOR PLOTTING ONLY ESS WHERE WE HAD POSITIVE ACCEPTANCE PROBABILITY."""
    cc_copy = cc.copy()
    ap_copy = ap.copy()
    flag = ap_copy[:, ix] < 1e-8
    values = cc_copy[:, ix]
    values[flag] = nan
    return values

def show_only_positive_ap_crwm(out_cc, out_ap, ϵs, ix):
    """Same as above but for C-RWM."""
    cc_copy = deepcopy(out_cc.copy())
    ap_copy = deepcopy(out_ap.copy())
    flag = ap_copy < 1e-8
    cc_copy[flag] = nan
    return repeat(cc_copy[ix], len(ϵs))


if __name__ == "__main__":
    # Construct folder from which we load data
    mainfolder = "LV_Experiment"
    subfolder = '2222_3333_4444_5555'
    folder = os.path.join(mainfolder, subfolder)
    # Folder into which we save plots
    savefolder = 'images'
    # Load function 
    loadfile = lambda filename: load(os.path.join(folder, filename))
    # Load data
    EPSILONS      = loadfile('EPSILONS.npy')
    BS            = loadfile('BS.npy')
    THUG00_CC_100 = loadfile('THUG00_CC_100.npy')
    THUG00_AP_100 = loadfile('THUG00_AP_100.npy')
    THUG09_CC_100 = loadfile('THUG09_CC_100.npy')
    THUG09_AP_100 = loadfile('THUG09_AP_100.npy')
    THUG99_CC_100 = loadfile('THUG99_CC_100.npy')
    THUG99_AP_100 = loadfile('THUG99_AP_100.npy')
    CRWM_CC_100   = loadfile('CRWM_CC_100.npy')
    CRWM_AP_100   = loadfile('CRWM_AP_100.npy')

    # Plot settings
    greys = ['lightgray', 'darkgrey', 'dimgrey']
    MARKERSIZE = 12
    LINEWIDTH  = 2
    MARKEREDGEWIDTH = 1.5
    CRWM_LINEWIDTH=2
    rc('font',**{'family':'STIXGeneral'})

    # Plot computational costs
    fig, ax = plt.subplots(figsize=(15, 6))
    # alpha = 0.0
    for i in range(len(BS)):
        ax.plot(EPSILONS, show_only_positive_ap(THUG00_CC_100, THUG00_AP_100, i), color=greys[i], marker='o', markersize=MARKERSIZE, markeredgewidth=MARKEREDGEWIDTH, lw=LINEWIDTH, markeredgecolor='k')
        ax.plot(EPSILONS, show_only_positive_ap(THUG09_CC_100, THUG09_AP_100, i), color=greys[i], marker='*', markersize=MARKERSIZE, markeredgewidth=MARKEREDGEWIDTH, lw=LINEWIDTH, markeredgecolor='k')
        ax.plot(EPSILONS, show_only_positive_ap(THUG99_CC_100, THUG99_AP_100, i), color=greys[i], marker='^', markersize=MARKERSIZE, markeredgewidth=MARKEREDGEWIDTH, lw=LINEWIDTH, markeredgecolor='k')
        ax.plot(EPSILONS, show_only_positive_ap_crwm(CRWM_CC_100, CRWM_AP_100, EPSILONS, i), marker='x', color=greys[i], markersize=MARKERSIZE, markeredgewidth=MARKEREDGEWIDTH, lw=LINEWIDTH, markeredgecolor='k')
    # Legend
    triangle = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='^', linestyle='None', markersize=10, label='α=0.99')
    star     = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='*', linestyle='None', markersize=10, label='α=0.9')
    circle   = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='o', linestyle='None', markersize=10, label='α=0')
    cross    = mlines.Line2D([], [], markeredgecolor='k', color='white', marker='x', linestyle='None', markersize=10, label='C-RWM')
    lightline  = mlines.Line2D([], [], color='lightgray', linestyle='-', label='B={}'.format(BS[0]))
    mediumline = mlines.Line2D([], [], color='darkgrey', linestyle='-', label='B={}'.format(BS[1]))
    darkline   = mlines.Line2D([], [], color='dimgrey', linestyle='-', label='B={}'.format(BS[2]))
    ax.legend(handles=[circle, star, triangle, cross, lightline, mediumline, darkline], loc='lower right')
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(r"$\mathregular{\epsilon}$", fontsize=20)
    ax.set_ylabel("minESS/runtime", fontsize=20)
    ax.loglog()
    plt.savefig('images/lv_computational_cost_100.png')
    plt.show()