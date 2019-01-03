import os
import random
import librosa


class TIMITReader(object):
    def __init__(self, timit_dir):
        self.corpus_dir = timit_dir

    def get_timit_wav_file(self, uttid):
        fn = '%s_convert.wav' % uttid
        path = os.path.join(self.corpus_dir, fn)
        samples, sr = librosa.load(path, sr=None)
        return samples, sr

    def _get_speaker_gender(self, uttid):
        # return the gender of the speaker who produced the utterance
        train_test, dialect_region, speakerid, sentid = uttid.split('/')
        if speakerid.startswith('f'):
            return 0
        elif speakerid.startswith('m'):
            return 1
        else:
            raise Exception('SpeakerID not marked for gender: %s' % speakerid)

    def _map_sentids_to_uttids(self):
        # map sentence IDs to their file paths
        # (several paths per sentence ID, each corresponding to an utterance)
        sentid_to_uttids = dict()
        with open(os.path.join(self.corpus_dir, 'allsenlist.txt'), 'r') as f:
            for line in f:
                uttid = line.split()[0]
                sentid = int(uttid.split('/')[-1][2:])
                if sentid not in sentid_to_uttids:
                    sentid_to_uttids[sentid] = []
                sentid_to_uttids[sentid].append(uttid)
        return sentid_to_uttids

    def _get_train_valid_test_uttids(self, validation_percentage,
                                    testing_percentage):
        """
        Divide utterances into training, validation, and test set.
        """
        ret = {'train': [], 'valid': [], 'test': []}

        sentids_to_uttids = self._map_sentids_to_uttids()
        uttids = []
        for ids in sentids_to_uttids.values():
            uttids += ids

        # so we get the same split every time
        random.seed(123)
        uttids = sorted(uttids)
        random.shuffle(uttids)

        valid_n = int(validation_percentage * len(uttids))
        test_n = int(testing_percentage * len(uttids))
        ret['valid'] = uttids[:valid_n]
        ret['test'] = uttids[valid_n:valid_n + test_n]
        ret['train'] = uttids[valid_n + test_n:]

        print('nr training examples: %s' % len(ret['train']))
        print('nr validation examples: %s' % len(ret['valid']))
        print('nr test examples: %s' % len(ret['test']))
        return ret

    def get_gender_utterances(self, validation_percentage, testing_percentage):
        """
        Divide utterances into train and test set, and assign gender class to
        each utterance (female=0, male=1).
        """
        data = {'train': [], 'test': [], 'valid': []}
        ids = self._get_train_valid_test_uttids(validation_percentage,
                                               testing_percentage)
        for partition, uttids in ids.items():
            for uttid in uttids:
                wav, rate = self.get_timit_wav_file(uttid)
                gender = self._get_speaker_gender(uttid)
                data[partition].append((gender, wav, uttid))
        return data
