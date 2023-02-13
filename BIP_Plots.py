import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import load, nanmean
import os

if __name__ == "__main__":
    # Find correct folder from which we load the data. Will depend on the seeds for each chain
    # this is the folder containing the Acceptance Probability (AP) data for the grid plot
    mainfolder = "BIP_Experiment"
    seeds = [1122, 2233, 3344, 4455, 5566, 6677, 7788, 8899, 9900, 1100]
    subfolder = "_".join([str(seed) for seed in seeds])
    folder = os.path.join(mainfolder, subfolder)

    # This next folder is the one for the computational cost.
    seeds_cc = [1122, 2233, 3344, 4455, 5566, 6677, 7788, 8899, 9900, 1100, 1830, 1038]
    folder_cc = os.path.join(mainfolder, "_".join([str(seed) for seed in seeds_cc]))

    # Load data for AP grid plot
    THUG_AP  = load(os.path.join(folder, "THUG_AP.npy"))
    CRWM_AP  = load(os.path.join(folder, "CRWM_AP.npy"))
    HMC_AP   = load(os.path.join(folder, "HMC_AP.npy"))
    RMHMC_AP = load(os.path.join(folder, "RMHMC_AP.npy"))
    SIGMA_GRID = load(os.path.join(folder, 'SIGMA_GRID.npy'))
    DELTA_GRID = load(os.path.join(folder, "DELTA_GRID.npy"))

    # Load data for CC plot
    THUG_CC  = load(os.path.join(folder_cc, "THUG_CC.npy"))
    THUG99_CC  = load(os.path.join(folder_cc, "THUG99_CC.npy"))
    CRWM_CC  = load(os.path.join(folder_cc, "CRWM_CC.npy"))
    HMC_CC   = load(os.path.join(folder_cc, "HMC_CC.npy"))
    SIGMA_GRID_CC = load(os.path.join(folder_cc, "SIGMA_GRID_CC.npy"))

    DATA = [CRWM_AP, THUG_AP, RMHMC_AP, HMC_AP]
    TITLES = ['C-RWM', 'THUG', 'RM-HMC', 'HMC']
    
    # Plot AP grid
    fig = plt.figure(figsize=(16, 4))
    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                    nrows_ncols=(1,4),
                    axes_pad=0.15,
                    share_all=False,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="7%",
                    cbar_pad=0.15,
                    )

    # Add data to image grid
    for i, ax in enumerate(grid):
        im = ax.imshow(DATA[i].T, vmin=0, vmax=1, origin='lower')
        ax.set_xlabel(r'$\mathregular{\sigma}$', fontsize=20, fontname='STIXGeneral')
        ax.set_xticks(range(0, 12, 2))
        ax.set_xticklabels([r'$\mathregular{10^{-5}}$', r'$\mathregular{10^{-4}}$', r'$\mathregular{10^{-3}}$', r'$\mathregular{10^{-2}}$', r'$\mathregular{10^{-1}}$', r'$\mathregular{10^{0}}$'], fontname='STIXGeneral', fontsize=14)
        ax.set_title(TITLES[i], fontsize=15, fontname='STIXGeneral')

    grid[0].set_yticks(range(0, 12, 2))
    grid[0].set_yticklabels([r'$\mathregular{10^{-5}}$', r'$\mathregular{10^{-4}}$', r'$\mathregular{10^{-3}}$', r'$\mathregular{10^{-2}}$', r'$\mathregular{10^{-1}}$', r'$\mathregular{10^{0}}$'], fontname='STIXGeneral', fontsize=14)
    grid[0].set_ylabel(r'$\mathregular{\delta}$', fontsize=20, fontname='STIXGeneral')
        
    # Colorbar
    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    plt.savefig("images/bip_ap_grid.png", dpi=300)
    # plt.tight_layout()    # Works, but may still require rect paramater to keep colorbar labels visible
    plt.show()

    # Plot Computational Cost (CC)
    fig, ax = plt.subplots()
    # Plot minESS/time as \sigma varies, in log-log scale
    max_index = 6
    rc('font',**{'family':'STIXGeneral'})
    ax.plot(SIGMA_GRID_CC[:max_index], THUG_CC[:max_index], label='THUG', marker='o', linewidth=2.5, markersize=9.0, markeredgecolor='navy', color='dodgerblue', markeredgewidth=2.0)
    ax.plot(SIGMA_GRID_CC[:max_index], THUG99_CC[:max_index], label='THUG99', marker='o', linewidth=2.5, markersize=9.0, markeredgecolor='orange', color='navajowhite', markeredgewidth=2.0)
    ax.plot(SIGMA_GRID_CC[:max_index], HMC_CC[:max_index], label='HMC', marker='o', linewidth=2.5, markersize=9.0, markeredgecolor='brown', color='lightcoral', markeredgewidth=2.0)
    ax.plot(SIGMA_GRID_CC[:max_index], CRWM_CC[:max_index], label='CRWM', marker='o', linewidth=2.5, markersize=9.0, markeredgewidth=2.0, markeredgecolor='forestgreen', color='lawngreen')
    ax.set_xlabel(r'Noise Scale ' + r'$\mathregular{\sigma}$', fontsize=16)
    ax.set_ylabel(r'MinESS / runtime', fontsize=16)
    ax.set_xticks(ticks=SIGMA_GRID_CC[:max_index])
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=15)
    ax.loglog()
    #plt.savefig("images/bip_cc_plot.png", dpi=300)
    plt.show()


