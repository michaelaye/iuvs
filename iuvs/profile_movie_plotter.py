import os
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from . import io, scaling
from .multitools import IntProgress, display

matplotlib.use('Agg')


def apply_and_compare(l1b, Scaler=scaling.PolyScaler1):
    plt.ioff()
    plt.rcParams['image.cmap'] = 'cubehelix'
    plt.rcParams['image.interpolation'] = None

    # set up ani writer
    Writer = animation.writers['ffmpeg']
    metadata = dict(title='IUVS dark subtracted profiles',
                    artist='K.-Michael Aye')
    writer = Writer(fps=3, metadata=metadata, bitrate=1800)

    # determine visual max and min for image plots
    min_, max_ = np.percentile(l1b.dds_dn_s, (1, 99))
    # now calculate min/max for profiles but with rows as index
#     profmin, profmax = np.percentile(l1b.dds_dn_s, (0.5,99.5), axis=(0,2))
    profmin = {5: -1, 31: -1, 55: -1}
    profmax = {5: 3, 31: 3, 55: 3}
    for ndark in range(l1b.n_darks):
        prog = IntProgress(min=0, max=l1b.n_integrations)
        display(prog)
        fulldark = l1b.get_integration('dark_dn_s', ndark)
        for nraw in range(l1b.n_integrations):
            fullraw = l1b.get_integration('raw_dn_s', nraw)
            spa_slice, spe_slice = l1b.find_scaling_window(fullraw)
            raw_subframe = fullraw[spa_slice, spe_slice]
            dark_subframe = fulldark[spa_slice, spe_slice]
            scaler = Scaler(dark_subframe, raw_subframe)
            scaler.do_fit()
            newdark = scaler.apply_fit(fulldark)
            subtracted = fullraw - newdark

            # plotting
            fig, ax = plt.subplots(nrows=3)
            rawmin, rawmax = np.percentile(fullraw, (1, 99))
            ax[0].imshow(fullraw, vmin=rawmin, vmax=rawmax)
            ax[0].set_title('Raw, {}_{} (NOT same colormap!)'.format(ndark, nraw))
            current = l1b.get_integration('dds_dn_s', nraw)
            ax[1].imshow(current, vmin=min_, vmax=max_)
            ax[1].set_title('Current dark subtraction')
            im = ax[2].imshow(subtracted, vmin=min_, vmax=max_)
            ax[2].set_title("{} scaled dark subtracted".format(Scaler))
            ax[2].set_xlabel('Spectral pixel number')
            fig.tight_layout()
            fig.subplots_adjust(top=0.9, bottom=0.1)
            cb = plt.colorbar(im, ax=ax.ravel().tolist())
            cb.set_label('  DN/s', fontsize=13, rotation=0)
            fig.savefig(os.path.join(str(io.plotfolder),
                                     'compare_{}_{}.png'.format(ndark, str(nraw).zfill(2))),
                        dpi=120)
            plt.close(fig)
            with sns.axes_style('whitegrid'):
                fig, ax = plt.subplots(nrows=4, sharex=True)
                with writer.saving(fig,
                                   '/Users/klay6683/plots/profiles_dark{}.mp4'
                                   .format(ndark), 100):
                    for row in [5, 31, 55]:
                        ax[0].plot(fullraw[row], lw=1, label='Raw, row{}'.format(row))
                    ax[0].set_ylim(0, 4)
                    ax[0].legend()
                    ax[0].set_title('Raw profiles', fontsize=10)
                    for row, myaxis in zip([5, 31, 55], ax[1:]):
                        myaxis.plot(current[row], 'r-', label='Current, row{}'.format(row),
                                    lw=1, alpha=0.7)
                        myaxis.plot(subtracted[row], 'g-', label='ScaledDark, row{}'.format(row),
                                    lw=1, alpha=0.7)
                        myaxis.set_ylim(profmin[row], profmax[row])
                        myaxis.legend()
                        myaxis.set_ylabel('DN / s')
                        myaxis.set_title('Row {}'.format(row), fontsize=10)
                    ax[3].set_xlabel('Spectral pixel number')
                    fig.suptitle('Profile comparison')
                    fig.savefig(os.path.join(str(io.plotfolder),
                                             'mean_profs_compare_{}_{}.png'
                                             .format(ndark, str(nraw).zfill(2))),
                                dpi=120)
                    writer.grab_frame()
                plt.close(fig)
            prog.value = nraw + 1

if __name__ == '__main__':
    l1b = io.L1BReader(sys.argv[1])
    sys.exit(apply_and_compare(l1b))
