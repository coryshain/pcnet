import pickle
import tensorflow as tf
import argparse

from ..kwargs import PCNET_INITIALIZATION_KWARGS
from ..config import Config
from ..model import PCRNNModel
from ..data import *
from ..plot import plot_features
from ..util import stderr, f_measure

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("""
    Trains a model.
    """)
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-c', '--cpu_only', action='store_true', help='Use CPU only, even if a GPU is available.')
    argparser.add_argument('-f', '--force_restart', action='store_true', help='Restart training, even if a checkpoint is available.')
    args = argparser.parse_args()

    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    force_restart = args.force_restart

    p = Config(args.config_path)
    train_data_dirs = p.train_data_dir.split(';')
    test_data_dirs = p.test_data_dir.split(';')
    data_filename = p.data_filename
    B = p['minibatch_size']
    F = p['n_features']
    U = p['n_units']
    L = p['n_layers']
    T = p['chunk_length']
    I = p['n_epochs']
    S = p['save_freq']
    P = p['plot_freq']
    E = p['eval_freq']
    outdir = p.outdir

    # Load saved data or convert WAV to cochleagram
    train_data = load_data(train_data_dirs, data_filename, F, T)
    X_train = train_data['X']

    # Process test data
    if test_data_dirs is None:
        test_data = train_data
    else:
        test_data = load_data(test_data_dirs, data_filename, F, T)
    cochleagram_files_test = test_data['cochleagram_files']
    file_keys_test = test_data['file_keys']
    X_test = test_data['X']
    phn_test = test_data['phn']
    phn_segs_test = test_data['phn_segs']
    wrd_test = test_data['wrd']
    wrd_segs_test = test_data['wrd_segs']
    vad_test = test_data['vad']

    # Initialize model
    kwargs = {}
    for kwarg in PCNET_INITIALIZATION_KWARGS:
        kwargs[kwarg.key] = p[kwarg.key]

    m = PCRNNModel(**kwargs)
    # m.build((None, T, F))
    # print(m.summary())
    checkpoint = tf.train.Checkpoint(m)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, m.outdir, 1)
    if not force_restart:
        stderr('Loading...\n')
        checkpoint_manager.restore_or_initialize()

    label_splits = []
    label_ix = 0
    pred_splits = []
    pred_ix = 0
    err_splits = []
    err_ix = 0
    for j in range(L - 1):
        if j == 0:
            pred_splits.append(pred_ix + F)
            pred_ix += F
            err_splits.append(err_ix + F)
            err_ix += F
        else:
            pred_splits.append(pred_ix + U)
            pred_ix += U
            err_splits.append(err_ix + U)
            err_ix += U
        label_splits.append(label_ix + U)
        label_ix += U

    # Train
    n_minibatch = int(math.ceil(len(X_train) / B))

    usingGPU = tf.test.is_gpu_available()
    stderr('Using GPU: %s\n' % usingGPU)

    while (m.optimizer.iterations.numpy() // n_minibatch) < I:
        i = m.optimizer.iterations.numpy() // n_minibatch
        print('Epoch %d' % (i + 1))

        if i == 0:
            with open(os.path.join(os.path.normpath(outdir), 'eval.csv'), 'a') as f:
                row = ['epoch']
                for l in range(L):
                    row += ['L%s_phn_B_P' % (l + 1), 'L%s_phn_B_R' % (l + 1), 'L%s_phn_B_F' % (l + 1)]
                    row += ['L%s_phn_W_P' % (l + 1), 'L%s_phn_W_R' % (l + 1), 'L%s_phn_W_F' % (l + 1)]
                    row += ['L%s_wrd_B_P' % (l + 1), 'L%s_wrd_B_R' % (l + 1), 'L%s_wrd_B_F' % (l + 1)]
                    row += ['L%s_wrd_W_P' % (l + 1), 'L%s_wrd_W_R' % (l + 1), 'L%s_wrd_W_F' % (l + 1)]
                f.write(','.join(row) + '\n')

        # Update
        m.fit(X_train, batch_size=B, epochs=1)

        if (i + 1) % S == 0:
            # Save
            stderr('Saving...\n')
            checkpoint_manager.save()

        if (i + 1) % P == 0:
            # Plot
            stderr('Plotting...\n')
            p = np.random.permutation(np.arange(len(X_test)))
            ix = p[:10]
            _X = X_test[ix]
            _phn_segs = phn_segs_test[ix]
            _wrd_segs = wrd_segs_test[ix]
            labels, predictions, errors, gates = m(
                _X,
                return_states=True,
                return_predictions=True,
                return_errors=True,
                return_gates=True
            )
            labels = np.concatenate([x.numpy() for x in labels], axis=-1)
            predictions = np.concatenate([x.numpy() for x in predictions], axis=-1)
            errors = np.concatenate([x.numpy() for x in errors], axis=-1)
            gates = np.stack([process_gates(x.numpy()) for x in gates], axis=1)
            pred_segs = segment_at_peaks(gates)

            plot_data = zip(
                _X,
                labels,
                predictions,
                errors,
                gates,
                _phn_segs,
                _wrd_segs,
                pred_segs
            )

            for j, arrs in enumerate(plot_data):
                _x, \
                _labels, \
                _preds, \
                _err, \
                _gates, \
                _p, \
                _w, \
                _pred_segs = arrs

                _labels = np.split(_labels, label_splits, axis=-1)
                _preds = np.split(_preds, pred_splits, axis=-1)
                _err = np.split(_err, err_splits, axis=-1)

                prefix = 'pcrnn_%d' % j
                plot_features(
                    _x,
                    _labels,
                    _preds,
                    _err,
                    _gates,
                    phn_segs=_p,
                    wrd_segs=_w,
                    pred_segs=_pred_segs,
                    outdir=os.path.join(os.path.normpath(outdir), 'plots'),
                    prefix=prefix
                )

        if (i + 1) % E == 0:
            # Evaluate
            stderr('Evaluating...\n')
            segment_tables = [dict() for _ in range(L)]

            max_len = max([len(cochleagram_files_test[x]) for x in file_keys_test])
            _X = np.stack(
                [np.pad(cochleagram_files_test[x], ((0, max_len - len(cochleagram_files_test[x])), (0, 0))) for x in file_keys_test],
                axis=0
            )

            labels, predictions, errors, gates = m(
                _X,
                return_states=True,
                return_predictions=True,
                return_errors=True,
                return_gates=True
            )
            gates = np.stack([process_gates(x.numpy()) for x in gates], axis=1)
            pred_segs = segment_at_peaks(gates)
            # pred_segs = np.zeros((len(test_file_keys), L, max_len))

            textgrids = {}
            for l in range(L):
                for j, k in enumerate(file_keys_test):
                    segment_tables[l][k] = get_segment_table(pred_segs[j, l, :len(cochleagram_files_test[k])], vad_df=vad_test[k])
                    if not k in textgrids:
                        textgrids[k] = [
                        ('phn', phn_test[k]),
                        ('wrd', wrd_test[k])
                    ]
                    textgrids[k].append(('L%d' % (l + 1), segment_tables[l][k]))
            stderr('\n')

            for fileID in textgrids:
                segment_table_to_textgrid(
                    textgrids[fileID],
                    fileID,
                    outdir=os.path.join(os.path.normpath(outdir), 'textgrids')
                )

            with open (os.path.join(os.path.normpath(outdir), 'eval.csv'), 'a') as f:
                row = ['%d' % (i + 1)]
                for l in range(L):
                    stderr('Layer %d:\n' % (l + 1))
                    # Phn eval
                    s = score_segmentation(phn_test, segment_tables[l], tol=0.02)[0]
                    B_P, B_R, B_F = f_measure(s['b_tp'], s['b_fp'], s['b_fn'])
                    W_P, W_R, W_F = f_measure(s['w_tp'], s['w_fp'], s['w_fn'])
                    stderr('  Phonemes:\n')
                    stderr('    BP: %.3f | BR: %.3f | BF: %.3f\n' % (B_P, B_R, B_F))
                    stderr('    WP: %.3f | WR: %.3f | WF: %.3f\n' % (W_P, W_R, W_F))
                    row += ['%.3f' % B_P, '%.3f' % B_R, '%.3f' % B_F]
                    row += ['%.3f' % W_P, '%.3f' % W_R, '%.3f' % W_F]

                    # Wrd eval
                    s = score_segmentation(wrd_test, segment_tables[l], tol=0.03)[0]
                    B_P, B_R, B_F = f_measure(s['b_tp'], s['b_fp'], s['b_fn'])
                    W_P, W_R, W_F = f_measure(s['w_tp'], s['w_fp'], s['w_fn'])
                    stderr('  Words:\n')
                    stderr('    BP: %.3f | BR: %.3f | BF: %.3f\n' % (B_P, B_R, B_F))
                    stderr('    WP: %.3f | WR: %.3f | WF: %.3f\n' % (W_P, W_R, W_F))
                    stderr('\n')
                    row += ['%.3f' % B_P, '%.3f' % B_R, '%.3f' % B_F]
                    row += ['%.3f' % W_P, '%.3f' % W_R, '%.3f' % W_F]

                f.write(','.join(row) + '\n')
