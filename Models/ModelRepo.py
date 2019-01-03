import os
import sqlite3


class ModelRepo(object):
    """
    Contains various methods for writing model statistics from training
    to an SQLITE data base. Each model for which you store results
    is assigned a new model ID and corresponding row in table 'ModelsT'.
    Loss and accuracy for each epoch are stored in table 'EpochsT'.
    """
    def __init__(self):
        self.conn = None    # data base connection
        self.cursor = None  # data base cursor
        self.model_id = None
        self.modelstable = 'ModelsT'
        self.epochstable = 'EpochsT'

    def set_new_model(self, params, params_dtypes):
        """
        :param params: a dictionary whose keys are strings specifying
        column names in ModelsT, and keys are field values.
        :param params_dtypes: a dictionary whose keys correspond to the keys
        in 'state'; values give the SQL data type of the values belonging to
        the same key in 'state'.
        """
        # assumes that the connection has been properly closed before.
        # user has to make sure this happened.
        db_path = os.path.join(params['data_base_path'])
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        if not self._table_exists(self.modelstable):
            self._create_models_table(params_dtypes)
        if not self._table_exists(self.epochstable):
            self._create_epochs_table()
        self.model_id = self._fetch_model_id()
        self._add_row_models_table(params, params_dtypes)

    def store_metrics_for_epochs(self, epochs, keras_model_history):
        """
        Store performance metrics across epochs.

        :param epochs: list of epochs
        :param keras_model_history: dictionary with keys 'accuracy_train',
        'accuracy_valid', 'loss_train', 'loss_valid'; values are lists
        with loss / acccuracy over epochs.
        """
        for i in epochs:
            self.store_metrics_at_epoch(i, keras_model_history)

    def store_metrics_at_epoch(self, epoch, keras_model_history):
        """
        Store evaluation metrics in EpochsT. You can call this
        e.g. after each epoch, if you are interested in storing
        evaluation metrics every epoch.

        :param epoch: epoch number
        :param keras_model_history: dictionary with keys 'accuracy_train',
        'accuracy_valid', 'loss_train', 'loss_valid'; values are lists
        with loss / acccuracy over epochs.
        """
        # get loss and accuracy for epoch
        metrics = {'acc': None, 'loss': None,
                   'val_acc': None, 'val_loss': None}
        for k in metrics.keys():
            # keras models always return training loss ('loss'),
            # but they may not return validation loss ('val_loss'),
            # accuracy ('acc') or validation accuracy ('val_acc')
            if k in keras_model_history:
                metrics[k] = keras_model_history[k][epoch]

        # row to be written to EpochsT
        row = [self.model_id,
               epoch,
               metrics['loss'],
               metrics['acc'],
               metrics['val_loss'],
               metrics['val_acc']]

        placeholders = ', '.join('?' * len(row))
        query = "INSERT INTO {tn} VALUES ({vals})".format(tn=self.epochstable,
                                                          vals=placeholders)
        self.cursor.execute(query, row)
        self.conn.commit()

    def _create_models_table(self, state_dtypes):
        # create ModelsT with 'model_id' column
        # make 'model_id' the primary key
        n0 = 'model_id'
        t0 = 'INTEGER'
        self.cursor.execute('CREATE TABLE {tn}({nf0} {ft0} '
                            'PRIMARY KEY)'.format(tn=self.modelstable,
                                                  nf0=n0, ft0=t0))
        # add additional columns to table
        # columns names are given by keys of 'state_dtypes',
        # with values specifying the column type
        for n, t in state_dtypes.items():
            self.cursor.execute('ALTER TABLE {tn} ADD COLUMN {fn} {ft}'
                                .format(tn=self.modelstable, fn=n, ft=t))
        self.conn.commit()

    def _create_epochs_table(self):
        # create EpochsT with with column 'model_id'
        # 'EpochsT.model_id' is a foreign key mapping from 'ModelsT.model_id'
        self.cursor.execute('CREATE TABLE {tn1}({fn} INTEGER,'
                            'FOREIGN KEY({fn}) REFERENCES {tn2}({fn}))'.
                            format(tn1=self.epochstable,
                                   tn2=self.modelstable,
                                   fn='model_id'))
        # place index on EpochsT.model_id to faster locate results
        # (this is useful if you store results for many models)
        self.cursor.execute("CREATE INDEX 'FK' ON {tn}({fn})".
                            format(tn=self.epochstable, fn='model_id'))

        # add some additional hard-coded columns to EpochsT
        fields = [('epoch', 'INTEGER'),
                  ('loss_train', 'REAL'),
                  ('accuracy_train', 'REAL'),
                  ('loss_valid', 'REAL'),
                  ('accuracy_valid', 'REAL')]
        for n, t in fields:
            self.cursor.execute('ALTER TABLE {tn} ADD COLUMN {fn} {ft}'.
                                format(tn=self.epochstable, fn=n, ft=t))
        self.conn.commit()

    def _add_row_models_table(self, state, state_dtypes):
        """ Add a row to ModelsT. """
        row = [self.model_id] + [state[k] for k in state_dtypes.keys()]
        placeholders = ', '.join('?' * len(row))
        query = 'INSERT INTO {tn} VALUES ({vals})'.\
            format(tn=self.modelstable, vals=placeholders)
        self.cursor.execute(query, row)
        self.conn.commit()

    def _fetch_model_id(self):
        """
        Construct a new model id
        (through incrementing the largest available id by 1).
        """
        ids = self._get_all_values(column_n='model_id',
                                   table_n=self.modelstable)
        if len(ids) == 0:
            new_id = 0
        else:
            ids = [i[0] for i in ids]
            new_id = max(ids) + 1
        return new_id

    def _table_exists(self, table_n):
        """
        Check if table named 'table_n' exists in data base.
        """
        tables = self.cursor.execute("SELECT name FROM sqlite_master "
                                     "WHERE type='table'")
        ret = False
        for t in tables:
            if t[0] == table_n:
                ret = True
        return ret

    def _get_all_values(self, column_n, table_n):
        """
        Return all unique values for column with name 'column_n'
        in table with name 'table_n'.
        """
        ids = self.cursor.execute('SELECT {fn} FROM {tn}'.
                                  format(tn=table_n, fn=column_n))
        ids = set([i for i in ids])
        return ids
