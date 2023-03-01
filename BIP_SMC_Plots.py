from numpy import load, array
import matplotlib.pyplot as plt
from matplotlib import rc
import os


folder = 'BIP_Experiment/SMC'
load_data = lambda file_name: load(os.path.join(folder, file_name), allow_pickle=True)[()]
# delta = 0.01
RWM_001 = load_data('SMC_DICT_RWM_001.npy')
THUG_001 = load_data('SMC_DICT_THUG_001.npy')
αTHUG_001 = load_data('SMC_DICT_ATHUG_001.npy')
# delta = 0.1
RWM_01 = load_data('SMC_DICT_RWM_01.npy')
THUG_01 = load_data('SMC_DICT_THUG_01.npy')
αTHUG_01 = load_data('SMC_DICT_ATHUG_01.npy')
# delta = 0.5
RWM_05 = load_data('SMC_DICT_RWM_05.npy')
THUG_05 = load_data('SMC_DICT_THUG_05.npy')
αTHUG_05 = load_data('SMC_DICT_ATHUG_05.npy')
# delta = 1.0
RWM_1 = load_data('SMC_DICT_RWM_1.npy')
THUG_1 = load_data('SMC_DICT_THUG_1.npy')
αTHUG_1 = load_data('SMC_DICT_ATHUG_1.npy')




if __name__ == "__main__":
    # Parameters
    LW = 3
    TICK_LABEL_SIZE=12
    LABEL_SIZE = 15
    TITLE_SIZE=20
    # Plot meta parameters
    n_rows = 4
    n_cols = 4
    # Storage
    D = array([
        [αTHUG_001, THUG_001, RWM_001, 0.01], 
        [αTHUG_01, THUG_01, RWM_01, 0.1],
        [αTHUG_05,  THUG_05,  RWM_05, 0.5],
        [αTHUG_1,  THUG_1,  RWM_1, 1.0]])
    K = ['UNIQUE_PARTICLES', 'ALPHAS', 'ESS', 'AP']
    TITLES = [r'$\mathregular{\delta=0.01}$', r'$\mathregular{\delta=0.1}$', r'$\mathregular{\delta=0.5}$', r'$\mathregular{\delta=1.0}$']
    COLOR_THUG = '#354F60'
    COLOR_HUG  = '#BC0E4C'
    COLOR_RWM  = '#FFC501'
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey='row', figsize=(16, 16))

    # Row 1 - Unique Particles
    for j in range(n_cols):
        # αTHUG
        ax[0, j].plot(D[j, 0]['EPSILON'], D[j, 0]['UNIQUE_PARTICLES'], label=r'$\alpha$THUG', lw=LW, color=COLOR_THUG)
        # THUG (α fixed)
        ax[0, j].plot(D[j, 1]['EPSILON'], D[j, 1]['UNIQUE_PARTICLES'], label=r'THUG', lw=LW, color=COLOR_HUG)
        # RWM
        ax[0, j].plot(D[j, 2]['EPSILON'], D[j, 2]['UNIQUE_PARTICLES'], label='RWM', lw=LW, color=COLOR_RWM)
        # Titles
        ax[0, j].set_title(TITLES[j], fontsize=TITLE_SIZE)
    ax[0, 0].set_ylabel("Unique Particles", fontsize=LABEL_SIZE)


    # Row 2 - Alphas
    for j in range(n_cols):
        # αTHUG
        ax[1, j].plot(D[j, 0]['EPSILON'], D[j, 0]['ALPHAS'], label=r'$\alpha$THUG', lw=LW, color=COLOR_THUG)
    ax[1, 0].set_ylabel(r"$\mathregular{\alpha}$", fontsize=LABEL_SIZE)

    # Row 3 - ESS
    for j in range(n_cols):
        # αTHUG
        ax[2, j].plot(D[j, 0]['EPSILON'], D[j, 0]['ESS'], label=r'$\alpha$THUG', lw=LW, color=COLOR_THUG)
        # THUG (α fixed)
        ax[2, j].plot(D[j, 1]['EPSILON'], D[j, 1]['ESS'], label=r'THUG', lw=LW, color=COLOR_HUG)
        # RWM
        ax[2, j].plot(D[j, 2]['EPSILON'], D[j, 2]['ESS'], label='RWM', lw=LW, color=COLOR_RWM)
    ax[2, 0].set_ylabel('ESS', fontsize=LABEL_SIZE)

    # Row 4 - Acceptance Probability
    for j in range(n_cols):
        # αTHUG
        ax[3, j].plot(D[j, 0]['EPSILON'], D[j, 0]['AP'], label=r'$\alpha$THUG', lw=LW, color=COLOR_THUG)
        # THUG (α fixed)
        ax[3, j].plot(D[j, 1]['EPSILON'], D[j, 1]['AP'], label=r'THUG', lw=LW, color=COLOR_HUG)
        # RWM
        ax[3, j].plot(D[j, 2]['EPSILON'], D[j, 2]['AP'], label='RWM', lw=LW, color=COLOR_RWM)
    ax[3, 0].set_ylabel('Acceptance Probability', fontsize=LABEL_SIZE)

    # Prettify
    for i in range(n_rows):
        for j in range(n_cols):
            if i == n_rows-1:
                ax[i, j].set_xlabel(r"$\mathregular{\epsilon}$", fontsize=20)
            ax[i, j].set_xscale('log')
            if i not in [0, 1, 2]:
                ax[i, j].set_yscale('log')
            ax[i, j].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    plt.legend(fontsize=12, loc='lower right')
    plt.tight_layout()
    plt.savefig('images/smc_thug_fixedstepsize.png')
    plt.show()