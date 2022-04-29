import os
import math
import numpy as np
import pandas as pd
import pickle

from .cochleagram import wav_to_cochleagram
from .util import stderr


def load_data(train_data_dirs, data_filename, n_features, chunk_length):
    cochleagram_files = {}
    file_keys = []
    data_dict = {}

    for data_dir in train_data_dirs:
        _cochleagram_files = []
        data_path = os.path.join(os.path.normpath(data_dir), data_filename)
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                __cochleagram_files = pickle.load(f)
        else:
            cochleagram_file_paths = sorted([x for x in os.listdir(os.path.normpath(data_dir)) if x.endswith('.wav')])
            __cochleagram_files = []
            for p, path in enumerate(cochleagram_file_paths):
                stderr('\rProcessing file %d/%d: %s                      ' % (p+1, len(cochleagram_file_paths), path))
                c = wav_to_cochleagram(os.path.join(os.path.normpath(data_dir), path), n_coef=n_features, order=0)
                __cochleagram_files.append((os.path.basename(path)[:-4], c[0].T))

            with open(os.path.join(os.path.normpath(data_dir), data_filename), 'wb') as f:
                pickle.dump(_cochleagram_files, f)
            stderr('\n')
        _cochleagram_files += __cochleagram_files

        _file_keys = [x[0] for x in _cochleagram_files]
        _cochleagram_files = dict(_cochleagram_files)
        cochleagram_files.update(_cochleagram_files)
        file_keys += _file_keys

        # Standardize data
        for k in _cochleagram_files:
            v = _cochleagram_files[k]
            v = v / (v.std(axis=-1, keepdims=True) + 1e-3)
            # v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-5)
            _cochleagram_files[k] = v

        phn = get_segs_from_dir(data_dir, segtype='phn')
        wrd = get_segs_from_dir(data_dir, segtype='wrd')
        vad = get_segs_from_dir(data_dir, segtype='vad')

        # Chop sequences into T-sized chunks
        _data_dict = {}
        for key in _file_keys:
            _data_dict[key] = {}
            c = _cochleagram_files[key]
            s_phn = segs_to_indicator(phn[key], len(c))
            s_wrd = segs_to_indicator(wrd[key], len(c))
            s_vad = segs_to_indicator(vad[key], len(c))
            paddings, pred_splits = get_batch_padding_and_split_indices(c, chunk_length)
            _data_dict[key]['X'] = c
            _data_dict[key]['X_batched'] = np.stack(np.split(np.pad(c, paddings), pred_splits), 0)
            _data_dict[key]['phn'] = s_phn
            _data_dict[key]['phn_batched'] = np.stack(np.split(np.pad(s_phn, paddings[:-1]), pred_splits), 0)
            _data_dict[key]['wrd'] = s_wrd
            _data_dict[key]['wrd_batched'] = np.stack(np.split(np.pad(s_wrd, paddings[:-1]), pred_splits), 0)
            _data_dict[key]['vad'] = s_vad
            _data_dict[key]['vad_batched'] = np.stack(np.split(np.pad(s_vad, paddings[:-1]), pred_splits), 0)
            _data_dict[key]['vad_batched'] = np.stack(np.split(np.pad(s_vad, paddings[:-1]), pred_splits), 0)

        data_dict.update(_data_dict)

    X = np.concatenate([data_dict[key]['X_batched'] for key in file_keys], axis=0)
    phn_segs = np.concatenate([data_dict[key]['phn_batched'] for key in file_keys], axis=0)
    wrd_segs = np.concatenate([data_dict[key]['wrd_batched'] for key in file_keys], axis=0)

    return {
        'cochleagram_files': cochleagram_files,
        'file_keys': file_keys,
        'data_dict': data_dict,
        'X': X,
        'phn_segs': phn_segs,
        'wrd_segs': wrd_segs,
        'phn': phn,
        'wrd': wrd,
        'vad': vad,
    }


def get_segs_from_dir(dir_path, segtype='phn'):
    file_paths = [dir_path + '/' + x for x in os.listdir(dir_path) if x.endswith(segtype)]
    out = {}
    for file_path in sorted(file_paths):
        ID = os.path.basename(file_path)[:-4]
        df = pd.read_csv(file_path, sep=' ')
        df.speaker = df.speaker.astype('str')
        df.sort_values('end', inplace=True)
        df.fileID = ID
        out[ID] = df

    return out


def get_batch_padding_and_split_indices(x, chunk_len, pad_type='random'):
    n_x = len(x)
    n_xtra = n_x % chunk_len
    if n_xtra:
        n_pad = chunk_len - n_xtra
    else:
        n_pad = 0
    if pad_type == 'pre':
        padding = (n_pad, 0)
    elif pad_type == 'post':
        padding = (0, n_pad)
    elif pad_type == 'random':
        i = np.random.randint(n_pad)
        padding = (i, n_pad - i)
    else:
        raise ValueError('Unrecognized pad_type %s' % pad_type)
    paddings = [padding]
    for _ in x.shape[1:]:
        paddings.append((0, 0))

    splits = np.arange(chunk_len, n_x + n_pad, chunk_len)

    return paddings, splits


def process_gates(x):
    # Gates index segment starts. Shift to index segment ends. Time is last dim.
    x = x[..., :-1]
    # paddings = []
    # for _ in x.shape[:-2]:
    #     paddings.append((0, 0))
    # paddings.append((0, 1))
    # x = np.pad(x, paddings)

    return x


def segment_at_peaks(x, threshold=0.):
    tm1 = x[..., :-2]
    t = x[..., 1:-1]
    tp1 = x[..., 2:]

    segs = np.logical_and(t >= tm1, t > tp1)
    if threshold:
        segs = np.logical_and(
            segs,
            t >= threshold
        )
    segs = segs.astype('float')

    paddings = []
    for _ in x.shape[:-1]:
        paddings.append((0,0))
    paddings.append((1,1))
    segs = np.pad(segs, paddings)

    return segs


def segs_to_indicator(segs, n, steps_per_second=100):
    out = np.zeros(n)
    ix = np.maximum(np.round(segs.end * steps_per_second).astype('int') - 1, 0)

    out[ix] = 1.

    return out


def get_segment_table(seg_indicators, vad_df=None, steps_per_second=100):
    """
    Convert a one-hot segment indicator vector into a pandas segment table, one segment per row.
    Supports masking with voice activity detection (VAD) intervals.

    :param seg_indicators: list or 1D numpy vector of segment endpoint indicators (0, 1) by timestep
    :param vad_df: pandas dataframe containing VAD intervals
    :param steps_per_second: Number of frames (timesteps) per second in the indicator array
    :return: segment table as pandas DataFrame
    """

    if vad_df is None:
        chunks = [(0, seg_indicators)]
    else:
        chunks = []
        starts = vad_df.start.values
        ends = vad_df.end.values
        for s, e in zip(starts, ends):
            s = int(np.round(s * steps_per_second))
            e = int(np.round(e * steps_per_second))
            chunk = seg_indicators[s:e]
            chunk[0] = 1
            chunk = (s, chunk)
            chunks.append(chunk)

    starts = []
    ends = []
    for chunk in chunks:
        s_c, _seg_indicators = chunk
        _starts = np.where(_seg_indicators)[0] + s_c
        if len(_starts):
            _ends = np.concatenate([_starts[1:], [len(_seg_indicators) + s_c]], axis=0)
            starts.append(_starts)
            ends.append(_ends)

    starts = np.concatenate(starts, axis=0)
    starts = starts / steps_per_second
    ends = np.concatenate(ends, axis=0)
    ends = ends / steps_per_second

    out = pd.DataFrame({'start': starts, 'end': ends})
    if vad_df is None:
        out['fileID'] = 1
        out['speaker'] = 1
    else:
        out['fileID'] = vad_df.fileID.unique()[0]
        out['speaker'] = vad_df.speaker.unique()[0]

    return out

def segment_table_to_textgrid(segments, fileID, outdir=None, suffix=''):
    """
    Dump one or more segmentation tables for a given acoustic file to a Praat-readable TextGrid file.

    :param segments: iterable of ``(str, pandas DataFrame)``; List of tables to write with their associated string labels, supplied as (label, table). Each table in the list will appear as an annotation level in the output.
    :param fileID: ``str``; string ID of audio file.
    :param outdir: ``str``; path to output directory
    :param suffix: ``str``; any additional info to be suffixed to the output file basename
    :return: ``None``
    """

    if outdir is None:
        outdir = './textgrids'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not isinstance(segments, list):
        segments = [segments]

    path = [outdir, '/', fileID, '_segmentations']
    if suffix:
        path.append('_' + suffix)
    path.append('.TextGrid')
    path = ''.join(path)
    with open(path, 'w') as f:
        for i, (segment_type, seg) in enumerate(segments):
            if i == 0:
                f.write('File type = "ooTextFile"\n')
                f.write('Object class = "TextGrid"\n')
                f.write('\n')
                f.write('xmin = %.3f\n' % seg.start.iloc[0])
                f.write('xmax = %.3f\n' % seg.end.iloc[-1])
                f.write('tiers? <exists>\n')
                f.write('size = %d\n' %len(segments))
                f.write('item []:\n')

            f.write('    item [%d]:\n' %i)
            f.write('        class = "IntervalTier"\n')
            f.write('        class = "segmentations %s"\n' %segment_type)
            f.write('        xmin = %.3f\n' % seg.start.iloc[0])
            f.write('        xmax = %.3f\n' % seg.end.iloc[-1])
            f.write('        intervals: size = %d\n' % len(seg))

            row_str = '        intervals [%d]:\n' + \
                '            xmin = %.3f\n' + \
                '            xmax = %.3f\n' + \
                '            text = "%s"\n\n'

            if 'label' in seg.columns:
                for j, r in seg[['start', 'end', 'label']].iterrows():
                    f.write(row_str % (j, r.start, r.end, r.label))
            else:
                for j, r in seg[['start', 'end']].iterrows():
                    f.write(row_str % (j, r.start, r.end, ''))


def score_segmentation_inner(true, pred, tol=0.02):
    """
    Score acoustic segmentations.

    :param true: pandas DataFrame, gold segment table, one segment per row
    :param pred: pandas DataFrame, predicted segment table, one segment per row
    :param tol: float, boundary alignment tolerance (larger is more forgiving)
    :return: 2-tuple, global and by-file dicts true positives, false positives, and false negatives at boundary and word levels
    """

    _true = np.array(true[['start', 'end']])
    _pred = np.array(pred[['start', 'end']])

    i = 0
    j = 0

    b_tp = 0
    b_fp = 0
    b_fn = 0

    w_tp = 0
    w_fp = 0
    w_fn = 0

    e_true_prev = _true[0, 0]
    e_pred_prev = _pred[0, 0]

    while i < len(_true) or j < len(_pred):
        if i >= len(_true):
            # All gold segments have been read.
            # Scan to the end of the predicted segments and tally up precision penalties.

            s_pred, e_pred = _pred[j]

            jump_pred = s_pred - e_pred_prev > 1e-5
            if jump_pred:
                e_pred = s_pred

            j += not jump_pred
            b_fp += 1
            w_fp += not jump_pred

            e_pred_prev = e_pred

        elif j >= len(_pred):
            # All predicted segments have been read.
            # Scan to the end of the true segments and tally up recall penalties.

            s_true, e_true = _true[i]

            jump_true = s_true - e_true_prev > 1e-5
            if jump_true:
                e_true = s_true

            i += not jump_true
            b_fn += 1
            w_fn += not jump_true

            e_true_prev = e_true

        else:
            # Neither true nor pred have finished
            s_true, e_true = _true[i]
            s_pred, e_pred = _pred[j]

            # If there is a "jump" in the true segs, create a pseudo segment spanning the jump
            jump_true = s_true - e_true_prev > 1e-5
            if jump_true:
                e_true = s_true
                s_true = e_true_prev

            # If there is a "jump" in the predicted segs, create a pseudo segment spanning the jump
            jump_pred = s_pred - e_pred_prev > 1e-5
            if jump_pred:
                e_pred = s_pred
                s_pred = e_pred_prev

            # Compute whether starts and ends of true and predicted segments align within the tolerance
            s_hit = False
            e_hit = False

            s_diff = math.fabs(s_true-s_pred)
            e_diff = math.fabs(e_true-e_pred)

            if s_diff <= tol:
                s_hit = True
            if e_diff <= tol:
                e_hit = True

            if s_hit:
                # Starts align, move on to the next segment in both feeds.
                # If we are in a pseudo-segment, don't move the pointer,
                # just update ``e_true_prev`` and ``e_true_prev``.

                i += not jump_true
                j += not jump_pred

                b_tp += 1

                # Only update word score tallies for non-pseudo-segments
                w_tp += e_hit and not (jump_true or jump_pred)
                w_fp += not e_hit and not jump_pred
                w_fn += not e_hit and not jump_true

                e_true_prev = e_true
                e_pred_prev = e_pred

            elif s_true < s_pred:
                # Starts did not align and the true segment starts before the predicted one.
                # Move on to the next segment in the true feed and tally recall penalties.
                # If we are in a pseudo-segment, don't move the pointer,
                # just update ``e_true_prev``.
                i += not jump_true

                b_fn += 1
                # Only update word score tallies for non-pseudo-segments
                w_fn += not jump_true

                e_true_prev = e_true

            else:
                # Starts did not align and the predicted segment starts before the true one.
                # Move on to the next segment in the predicted feed and tally precision penalties.
                # If we are in a pseudo-segment, don't move the pointer,
                # just update ``e_pred_prev``.
                j += not jump_pred

                b_fp += 1
                # Only update word score tallies for non-pseudo-segments
                w_fp += not jump_pred

                e_pred_prev = e_pred

    # Score final boundary
    hit = math.fabs(e_true_prev - e_pred_prev) <= tol
    b_tp += hit
    b_fp += not hit
    b_fn += not hit

    out = {
        'b_tp': b_tp,
        'b_fp': b_fp,
        'b_fn': b_fn,
        'w_tp': w_tp,
        'w_fp': w_fp,
        'w_fn': w_fn
    }

    return out


def score_segmentation(true, pred, tol=0.02):
    """
    Score acoustic segmentations.

    :param true: dict of pandas DataFrame, gold segment tables, one entry per file, one segment per row
    :param pred: dict of pandas DataFrame, gold segment tables, one entry per file, one segment per row
    :param tol: float, boundary alignment tolerance (larger is more forgiving)
    :return: 2-tuple, global and by-file dicts true positives, false positives, and false negatives at boundary and word levels
    """

    score_dict = {}
    for f in true:
        score_dict[f] = score_segmentation_inner(true[f], pred[f], tol=tol)

    global_score_dict = {
        'b_tp': sum([score_dict[f]['b_tp'] for f in true]),
        'b_fp': sum([score_dict[f]['b_fp'] for f in true]),
        'b_fn': sum([score_dict[f]['b_fn'] for f in true]),
        'w_tp': sum([score_dict[f]['w_tp'] for f in true]),
        'w_fp': sum([score_dict[f]['w_fp'] for f in true]),
        'w_fn': sum([score_dict[f]['w_fn'] for f in true]),
    }

    return global_score_dict, score_dict