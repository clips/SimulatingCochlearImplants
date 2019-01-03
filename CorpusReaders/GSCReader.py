import os
import re
import hashlib
import librosa
from scipy.io.wavfile import read as read_wav


class GSCReader(object):
    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.core_words = {"yes", "no", "up", "down", "left", "right", "on",
                           "off", "stop", "go", "zero", "one", "two", "three",
                           "four", "five", "six", "seven", "eight", "nine"}
        self.aux_words = {"bed", "bird", "cat", "dog", "happy", "house",
                          "marvin", "sheila", "tree", "wow"}
        self.all_words = set(list(self.core_words) + list(self.aux_words))

    def _which_set(self, filepath, validation_percentage, testing_percentage):
        """
        From README.md

        Determines which data partition the file should belong to.

        We want to keep files in the same training, validation, or testing sets
        even if new ones are added over time. This makes it less likely that
        testing samples will accidentally be reused in training when long runs
        are restarted for example. To keep this stability, a hash of the
        filename is taken and used to determine which set it should belong to.
        This determination only depends on the name and the set proportions,
        so it won't change as other files are added.

        It's also useful to associate particular files as related (for example
        words spoken by the same person), so anything after '_nohash_' in a
        filename is ignored for set determination. This ensures that
        'bobby_nohash_0.wav' and 'bobby_nohash_1.wav' are always in the same
        set, for example.

        Args:
          filepath: File path of the data sample.
          validation_percentage: How much of the data set to use for
          validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          String, one of 'training', 'validation', or 'testing'.
        """
        base_name = os.path.basename(filepath)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a
        #  way of grouping wavs that are close variations of each other.
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        # This looks a bit magical, but we need to decide whether this file
        # should go into the training, testing, or validation sets,
        # and we want to keep existing files in the same set even if more
        # files are subsequently added.
        # To do that, we need a stable way of deciding based on just the
        # file name itself, so we do a hash of that and then use that to
        # generate a probability value that we use to assign it.
        hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
        MAX_NUM_WAVS_PER_CLASS = 2 ** 27
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_WAVS_PER_CLASS + 1)) *
                           (100.0 / MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
            result = 'valid'
        elif percentage_hash < (testing_percentage + validation_percentage):
            result = 'test'
        else:
            result = 'train'
        return result

    def _filter_vocoded(self, filenames):
        filenames = [fn for fn in filenames if
                     not fn.endswith('_vocoded.wav')]
        return filenames

    def _is_relevant_word(self, word, aux_words):
        if aux_words:
            return word in self.all_words
        else:
            return word in self.core_words

    def get_train_valid_test(self, valid_percentage, test_percentage,
                             aux_words=False):

        ret = {'train': [], 'test': [], 'valid': []}
        for root, dirs, filenames in os.walk(self.corpus_dir):
            for d in dirs:
                if not self._is_relevant_word(d, aux_words):
                    continue
                worddir = os.path.join(self.corpus_dir, d)
                for root, dirs, filenames in os.walk(worddir):
                    for f in filenames:
                        filepath = os.path.join(worddir, f)
                        fileid = os.path.join(d, f)
                        partition = self._which_set(filepath, valid_percentage,
                                                    test_percentage)
                        try:
                            samples, sr = librosa.load(filepath, sr=None)
                            ret[partition].append((d, samples, fileid))
                        except FileNotFoundError:
                            print('File not found: %s' % filepath)

        print('nr training examples: %s' % len(ret['train']))
        print('nr validation examples: %s' % len(ret['valid']))
        print('nr test examples: %s' % len(ret['test']))
        return ret
