import matplotlib.pyplot as plt
from matplotlib import rc
from numpy import load, pad, array, float64, nan
import os

folder = 'BIP_Experiment/SMC_RWM_THEN_THUG'
load_data = lambda file_name: load(os.path.join(folder, file_name), allow_pickle=True)[()]
αTHUG  = load_data('SMC_DICT_RWM_THEN_ATHUG.npy')
THUG = load_data('SMC_DICT_RWM_THEN_THUG.npy')

rc('font',**{'family':'STIXGeneral'})
COLOR_THUG = '#354F60'
COLOR_HUG  = '#BC0E4C'
COLOR_RWM  = '#FFC501'

padit = lambda out, lb: pad(array(out[lb], dtype=float64)[:out['SWITCH_TO_THUG']], (0, len(out[lb]) - len(out[lb][:out['SWITCH_TO_THUG']])), 'constant', constant_values=nan)


if __name__ == "__main__":

    padded_alphas = pad(αTHUG['ALPHAS'], (len(αTHUG['EPSILON']) - len(αTHUG['ALPHAS']), 0), 'constant', constant_values=nan)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4), sharex=True)
    lw = 2.5
    TICK_LABEL_SIZE=12
    TITLE_SIZE = 20

# Unique Particles
ax[0].plot(αTHUG['EPSILON'], αTHUG['UNIQUE_PARTICLES'], label=r'THUG-$\alpha$', lw=lw, color=COLOR_THUG)
ax[0].plot(THUG['EPSILON'], THUG['UNIQUE_PARTICLES'], label='THUG', lw=lw, color=COLOR_HUG)
# Before switching
ax[0].plot(padit(αTHUG, 'EPSILON'), padit(αTHUG, 'UNIQUE_PARTICLES'), label=r'RWM', lw=lw, color=COLOR_RWM)
ax[0].plot(padit(THUG, 'EPSILON'), padit(THUG, 'UNIQUE_PARTICLES'), lw=lw, color=COLOR_RWM)
ax[0].set_xlabel("ϵ", fontsize=20)
#ax[0].set_ylabel("Unique Particles")
ax[0].set_xscale('log')
ax[0].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
ax[0].set_title("Unique Particles", fontsize=TITLE_SIZE)

# ALPHAS
ax[1].plot(αTHUG['EPSILON'], padded_alphas, label=r'THUG-$\alpha$', lw=lw, color=COLOR_THUG)
ax[1].set_xlabel("ϵ", fontsize=20)
#ax[1].set_ylabel("Step Sizes")
ax[1].set_xscale('log')
# ax[1].set_yscale('log')
ax[1].tick_params(axis='both', which='both', labelsize=TICK_LABEL_SIZE)
ax[1].set_title(r"$\mathregular{\alpha}$", fontsize=TITLE_SIZE)

# ESS
ax[2].plot(αTHUG['EPSILON'], αTHUG['ESS'], label=r'THUG-$\alpha$', lw=lw, color=COLOR_THUG)
ax[2].plot(THUG['EPSILON'], THUG['ESS'], label='THUG', lw=lw, color=COLOR_HUG)
# swtich
ax[2].plot(padit(αTHUG, 'EPSILON'), padit(αTHUG, 'ESS'), label=r'RWM', lw=lw, color=COLOR_RWM)
ax[2].plot(padit(THUG, 'EPSILON'), padit(THUG, 'ESS'), lw=lw, color=COLOR_RWM)
ax[2].set_xlabel("ϵ", fontsize=20)
#ax[2].set_ylabel("ESS")
ax[2].set_xscale('log')
# ax[2].set_yscale('log')
ax[2].tick_params(axis='both', which='both', labelsize=TICK_LABEL_SIZE)
ax[2].set_title("ESS", fontsize=TITLE_SIZE)

# Acceptance Probability
ax[3].plot(αTHUG['EPSILON'], αTHUG['AP'], label=r'$\alpha$THUG', lw=lw, color=COLOR_THUG)
ax[3].plot(THUG['EPSILON'], THUG['AP'], label='THUG', lw=lw, color=COLOR_HUG)
# SWITCH
ax[3].plot(padit(αTHUG, 'EPSILON'), padit(αTHUG, 'AP'), label=r'RWM', lw=lw, color=COLOR_RWM)
ax[3].plot(padit(THUG, 'EPSILON'), padit(THUG, 'AP'), lw=lw, color=COLOR_RWM)
ax[3].set_xlabel("ϵ", fontsize=20)
#ax[3].set_ylabel("Acceptance probability")
ax[3].set_xscale('log')
ax[3].set_yscale('log')
ax[3].tick_params(axis='both', which='both', labelsize=TICK_LABEL_SIZE)
ax[3].set_title("Acceptance Probability", fontsize=TITLE_SIZE)

for i in range(4):
    ax[i].axvline(αTHUG['EPSILON'][αTHUG['SWITCH_TO_THUG']], color='dimgrey', ls='--', zorder=0)
    ax[i].set_xscale('log')


plt.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig("images/smc_thug_rwm_then_thug.png", dpi=300)
plt.show()