import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker, pyplot as plt
from matplotlib import markers
import matplotlib.colors

from .util import stderr

def plot_features(
        obs, # shape (T, K)
        states, # list of L arrays of shape (T, K)
        preds, # list of L arrays of shape (T, K)
        errs, # list of L arrays of shape (T, K)
        gates, # list of L arrays of shape (T,)
        phn_segs=None, # shape (T,)
        wrd_segs=None, # shape (T,)
        pred_segs=None, # list of L arrays of shape (T,)
        outdir='./plots',
        prefix='pcrnn',
        sr = 16000,
        hop_length = 160,
):
    assert len(states) == len(preds) == len(errs) == len(gates), 'states, preds, errs, and gates must all have the same first dimension (number of layers). Saw %s, %s, %s, %s, %s.' % (len(states), len(preds), len(errs), len(gates), len(rho))

    targs = [obs] + states[:-1] # Targets are the activities of the layer below

    for l in range(len(states)):
        fig = plt.figure()
        axes = []
        for i in range(5):
            axes.append(fig.add_subplot(5, 1, i + 1))

        for m, ax, lab in zip((targs[l], states[l], preds[l], errs[l]), axes, ('Targets', 'States', 'Predictions', 'Residuals')):
            librosa.display.specshow(
                m.T,
                sr=sr,
                hop_length=hop_length,
                fmax=8000,
                x_axis='time',
                ax=ax,
                cmap='coolwarm',
                norm=matplotlib.colors.TwoSlopeNorm(vcenter=0.)
            )

            n_gold_seg_levels = int(phn_segs is not None) + int(wrd_segs is not None)
            if n_gold_seg_levels:
                if phn_segs is None:
                    wrd_ymin = 0
                    wrd_ymax = m.shape[-1]
                elif phn_segs is None:
                    phn_ymin = 0
                    phn_ymax = m.shape[-1]
                else:
                    phn_ymin = 0
                    phn_ymax = float(m.shape[-1]) // 2
                    wrd_ymin = float(m.shape[-1]) // 2
                    wrd_ymax = m.shape[-1]

            if phn_segs is not None and lab == 'Targets':
                timestamps = np.where(phn_segs)[0] / 100
                ax.vlines(timestamps, phn_ymin, phn_ymax, color='k', linewidth=1, alpha=0.2)
            if wrd_segs is not None and lab == 'Targets':
                timestamps = np.where(wrd_segs)[0] / 100
                ax.vlines(timestamps, wrd_ymin, wrd_ymax, color='k', linewidth=1, alpha=0.2)
            if pred_segs is not None and lab == 'Predictions':
                timestamps = np.where(pred_segs[l])[0] / 100
                ax.vlines(timestamps, 0, m.shape[-1], color='k', linewidth=1, alpha=0.2)
            ax.set_title(lab)

        # Segmentation signal

        ax = axes[-1]
        x = np.arange(len(gates[l])) / 100
        ax.plot(x, gates[l])
        ax.set_xlim(0., len(obs) / 100)
        ax.set_ylim(0., 1.)
        ax.set_title('Gate')

        fig.set_size_inches(12, 12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        outdir = os.path.normpath(outdir)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        try:
            fig.savefig(os.path.join(outdir, prefix + '_l%d_featureplot.png' % (l + 1)))
        except Exception:
            stderr('IO error when saving plot. Skipping plotting...\n')

        plt.close(fig)
