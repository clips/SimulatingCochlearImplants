import os
import matplotlib
matplotlib.use('Agg')
import librosa.display
from matplotlib import pyplot
pyplot.rcParams["font.family"] = "Times New Roman"
from Featurize.utils import load_featurized


def plot_spectrogram(x, plot_path):
    fig = pyplot.figure()
    ax = pyplot.Axes(fig, [0., 0., 1., 1.])
    pyplot.rcParams["axes.edgecolor"] = "0.15"
    pyplot.rcParams["axes.linewidth"] = 1.25
    pyplot.rcParams["axes.grid"] = True
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(x, y_axis='mel', x_axis='time', fmin=200,
                             fmax=7000, sr=16000, cmap='gray_r')
    pyplot.savefig(plot_path, bbox_inches=0)
    pyplot.close()


def plot_spectrograms(highres_path, medres_path, lowres_path, plot_dir):

    high_xs, ys, label_to_idx, x_ids = load_featurized(highres_path)
    med_xs, ys, label_to_idx, x_ids = load_featurized(medres_path)
    low_xs, ys, label_to_idx, x_ids = load_featurized(lowres_path)

    for res, xs in [('highres', high_xs),
                    ('medres', med_xs),
                    ('lowres', low_xs)]:
        for i, x, in enumerate(xs[:100]):
            plot_path = os.path.join(plot_dir, '%s_%s' % (i, res))
            plot_spectrogram(x, plot_path)


if __name__ == '__main__':

    for dataset in ('gender', 'words'):

        file_dir = os.path.abspath(os.path.dirname(__file__))
        data_dir = os.path.join(file_dir, 'Featurize', 'featurized')

        highres = os.path.join(data_dir, dataset, 'hires_test')
        medres = os.path.join(data_dir, dataset, 'medres_test')
        lowres = os.path.join(data_dir, dataset, 'lores_test')

        plot_dir = os.path.join(file_dir, 'results', 'plots',
                                '%s_spectrograms' % dataset)

        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

        plot_spectrograms(highres, medres, lowres, plot_dir)
