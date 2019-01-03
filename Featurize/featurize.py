import librosa
import copy
import numpy as np


def mel_spec(x, sr, n, fmax, fmin):
    # n_fft = 50ms window
    # hop_length = strided by 10ms
    mel_s = librosa.feature.melspectrogram(y=x, sr=sr, fmin=fmin,
                                           fmax=fmax, n_mels=n, power=1,
                                           n_fft=800, hop_length=160)
    assert np.min(mel_s) >= 0.0
    return mel_s


def standardize(xs, eps=1e-14):
    # get feature mean and standard deviation
    stacked = np.hstack(xs)
    feats_mean = np.mean(stacked, axis=1)
    feats_std = np.std(stacked, axis=1)
    # standardize each sample
    func = lambda x: (x - feats_mean) / feats_std + eps
    xs_stand = [np.apply_along_axis(func, axis=0, arr=x) for x in xs]
    return xs_stand


def add_channel_interaction(spec):
    new_spec = np.zeros(spec.shape)

    for row_idx in range(spec.shape[0]):

        upper_lower_idxs = []
        if row_idx == 0:
            upper_lower_idxs.append(row_idx + 1)
        elif row_idx == spec.shape[0] - 1:
            upper_lower_idxs.append(row_idx - 1)
        else:
            upper_lower_idxs.append(row_idx + 1)
            upper_lower_idxs.append(row_idx - 1)

        row = copy.deepcopy(spec[row_idx])
        for idx in upper_lower_idxs:
            row += spec[idx]
        new_spec[row_idx] = row

    return new_spec


def enlarge_melspec(spec, enlarge_factor):

    new_spec = np.zeros((enlarge_factor * spec.shape[0], spec.shape[1]))

    start = 0
    for row_idx in range(spec.shape[0]):
        for i in range(start, start + enlarge_factor):
            new_spec[i] = spec[row_idx]
        start += enlarge_factor

    return new_spec


def pad_or_cut(vectors, nmels, cut_to_mean_length, debug=True):

    assert vectors[0].shape[0] == nmels

    frame_lens = [x.shape[1] for x in vectors]

    if debug:
        print('INPUT LENGTH INFO')
        print('mean nr frames %s (sd: %s)'
              '-- min: %s -- max: %s' %
              (np.mean(frame_lens), np.std(frame_lens),
               min(frame_lens), max(frame_lens)))

    if cut_to_mean_length:
        max_frames = int(np.mean(frame_lens))
    else:
        max_frames = max(frame_lens)
    padded = np.zeros((len(vectors), nmels, max_frames))
    for idx, x in enumerate(vectors):
        # cut off frames if necessary
        if x.shape[1] > max_frames:
            x = x[:, :max_frames]
        padded[idx, :, :x.shape[1]] = x
    return padded


def featurize(wav_data, n_hires, n_medres, sr, cut_to_mean_len):

    wavs = []
    ys = []
    ids = []
    for k in ['train', 'valid', 'test']:
        for label, wav, id_ in wav_data[k]:
            wavs.append(wav)
            ys.append(label)
            ids.append(id_)

    unique_y = sorted(list(set(ys)))
    y_to_idx = dict(zip(unique_y, range(len(unique_y))))
    ys = [y_to_idx[y] for y in ys]

    xs_medres = [mel_spec(x, sr, n_medres, fmax=7000, fmin=200) for x in wavs]
    xs_hires = [mel_spec(x, sr, n_hires, fmax=7000, fmin=200) for x in wavs]
    xs_lores = [add_channel_interaction(x) for x in xs_medres]

    enlarge_factor = int(n_hires / n_medres)
    if not n_hires % n_medres == 0:
        raise Exception('n_hires must be evenly divisible by n_medres')

    xs_lores = [enlarge_melspec(x, enlarge_factor) for x in xs_lores]
    xs_medres = [enlarge_melspec(x, enlarge_factor) for x in xs_medres]

    n_train = len(wav_data['train'])
    n_valid = len(wav_data['valid'])

    ret = dict()

    for key, xs in [('hires', xs_hires), ('medres', xs_medres),
                    ('lores', xs_lores)]:

        xs = standardize(xs)
        xs = pad_or_cut(xs, n_hires, cut_to_mean_len)

        ret[key] = dict()

        ret[key]['x_train'] = xs[:n_train]
        ret[key]['x_valid'] = xs[n_train:n_train + n_valid]
        ret[key]['x_test'] = xs[n_train + n_valid:]

        ret[key]['y_train'] = ys[:n_train]
        ret[key]['y_valid'] = ys[n_train:n_train + n_valid]
        ret[key]['y_test'] = ys[n_train + n_valid:]

        ret[key]['train_ids'] = ids[:n_train]
        ret[key]['valid_ids'] = ids[n_train:n_train + n_valid]
        ret[key]['test_ids'] = ids[n_train + n_valid:]

    return ret, y_to_idx
