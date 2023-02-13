import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import load 
import os

if __name__ == "__main__":
    # Find correct folder from which we load the data. Will depend on the seeds for each chain
    mainfolder = "BIP_Experiment"
    seeds = [1122, 2233, 3344, 4455, 5566, 6677, 7788, 8899, 9900, 1100]
    subfolder = "_".join([str(seed) for seed in seeds])
    folder = os.path.join(mainfolder, subfolder)

    # Load data
    THUG_AP  = load(os.path.join(folder, "THUG_AP.npy"))
    CRWM_AP  = load(os.path.join(folder, "CRWM_AP.npy"))
    HMC_AP   = load(os.path.join(folder, "HMC_AP.npy"))
    RMHMC_AP = load(os.path.join(folder, "RMHMC_AP.npy"))
    SIGMA_GRID = load(os.path.join(folder, 'SIGMA_GRID.npy'))
    DELTA_GRID = load(os.path.join(folder, "DELTA_GRID.npy"))

    DATA = [CRWM_AP, THUG_AP, RMHMC_AP, HMC_AP]
    TITLES = ['C-RWM', 'THUG', 'RM-HMC', 'HMC']
    # Set up figure and image grid
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