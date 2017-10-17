import tensorflow as tf

from global_module.settings_module import set_params, set_dir


class SMN():
    def __init__(self, params, dir_obj):
        self.params = params
        self.dir_obj = dir_obj
        self.init_pipeline()

    def init_pipeline(self):
        self.create_placeholders()
        self.extract_word_embedding()
        self.initial_hidden_state()
        self.compute_matching_matrix()

    def initial_hidden_state(self):
        self.extract_ctx_hidden_embedding('layer1')
        self.extract_resp_hidden_embedding('layer1')

    def create_placeholders(self):
        self.ctx = tf.placeholder(dtype=tf.int32,
                                  shape=[None,
                                         self.params.NUM_CONTEXT,
                                         self.params.MAX_CTX_UTT_LENGTH],
                                  name='ctx_placeholder')

        self.ctx_len_placeholders = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, self.params.NUM_CONTEXT],
                                                   name='ctx_len_placeholder')

        self.num_ctx_placeholders = tf.placeholder(dtype=tf.int32,
                                                   shape=[None],
                                                   name='num_ctx_placeholder')

        self.resp = tf.placeholder(dtype=tf.int32,
                                   shape=[None, self.params.MAX_RESP_UTT_LENGTH],
                                   name='res_placeholder')

        self.resp_len_placeholders = tf.placeholder(dtype=tf.int32,
                                                    shape=[None],
                                                    name='resp_len_placeholder')

        self.label = tf.placeholder(dtype=tf.float32,
                                    shape=[None],
                                    name='response_label')

    def extract_word_embedding(self):
        self.word_emb_matrix = tf.get_variable("word_embedding_matrix",
                                               shape=[self.params.vocab_size, self.params.EMB_DIM],
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                               trainable=self.params.is_word_trainable)

        self.ctx_word_emb = tf.nn.embedding_lookup(params=self.word_emb_matrix,
                                                   ids=self.ctx,
                                                   name='ctx_word_emb',
                                                   validate_indices=True)

        self.resp_word_emb = tf.nn.embedding_lookup(params=self.word_emb_matrix,
                                                    ids=self.resp,
                                                    name='resp_word_emb',
                                                    validate_indices=True)

        print 'Extracted word embedding'

    def create_rnn_cell(self, scope, option='lstm'):
        if option == 'lstm':
            with tf.variable_scope(scope):
                rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.params.RNN_HIDDEN_DIM, forget_bias=1.0)
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.params.keep_prob)
                return rnn_cell

        elif option == 'gru':
            with tf.variable_scope(scope):
                rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.params.RNN_HIDDEN_DIM)
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.params.keep_prob)
                return rnn_cell

    def extract_ctx_hidden_embedding(self, name):
        with tf.variable_scope('rnn_ctx_layer') as scope:
            self.rnn_ctx_cell = self.create_rnn_cell(name, self.params.rnn)
            reshaped_input = tf.reshape(self.ctx_word_emb, shape=[-1, self.params.MAX_CTX_UTT_LENGTH, self.params.EMB_DIM])
            reshaped_length = tf.reshape(self.ctx_len_placeholders, shape=[-1])
            rnn_output, rnn_state = tf.nn.dynamic_rnn(self.rnn_ctx_cell,
                                                      reshaped_input,
                                                      reshaped_length,
                                                      dtype=tf.float32)

            self.rnn_ctx_output = tf.reshape(rnn_output,
                                             shape=[-1, self.params.NUM_CONTEXT, self.params.MAX_CTX_UTT_LENGTH, self.params.RNN_HIDDEN_DIM],
                                             name='layer1_output')
            self.rnn_ctx_state = tf.reshape(rnn_state,
                                            shape=[-1, self.params.NUM_CONTEXT, self.params.RNN_HIDDEN_DIM],
                                            name='layer1_state')

            if self.params.rnn == 'lstm':
                self.rnn_ctx_state = self.rnn_ctx_state[0]

            print 'Extracted rnn hidden states.'

    def extract_resp_hidden_embedding(self, name):
        with tf.variable_scope('rnn_resp_layer') as scope:
            if self.params.USE_SAME_CELL:
                self.rnn_resp_cell = self.rnn_ctx_cell
            else:
                self.rnn_resp_cell = self.create_rnn_cell(name, self.params.rnn)

            self.rnn_resp_output, self.rnn_resp_state = tf.nn.dynamic_rnn(self.rnn_resp_cell,
                                                                          self.resp_word_emb,
                                                                          self.resp_len_placeholders,
                                                                          dtype=tf.float32)

            if self.params.rnn == 'lstm':
                self.rnn_resp_state = self.rnn_resp_state[0]

            print 'Extracted rnn hidden states.'

    def compute_matching_matrix(self):
        ctx_word_emb_split = tf.split(self.ctx_word_emb, num_or_size_splits=self.params.NUM_CONTEXT, axis=1)
        ctx_hidden_emb_split = tf.split(self.rnn_ctx_output, self.params.NUM_CONTEXT, axis=1)

        word_matching_matrix = []

        for each_ctx in ctx_word_emb_split:
            word_matching_matrix.append(tf.matmul(tf.squeeze(each_ctx, axis=1), self.resp_word_emb, transpose_b=True))

        hidden_matching_matrix = []

        self.linear_transform = tf.get_variable(name='linear_transform',
                                                shape=[self.params.RNN_HIDDEN_DIM, self.params.RNN_HIDDEN_DIM],
                                                dtype=tf.float32)

        for each_ctx in ctx_hidden_emb_split:
            each_ctx_reshaped = tf.reshape(tf.squeeze(each_ctx, axis=1), [-1, self.params.RNN_HIDDEN_DIM])
            mul1 = tf.matmul(each_ctx_reshaped, self.linear_transform)
            mul1_reshaped = tf.reshape(mul1, [-1, self.params.MAX_CTX_UTT_LENGTH, self.params.RNN_HIDDEN_DIM])
            mul2 = tf.matmul(mul1_reshaped, self.rnn_resp_output, transpose_b=True)
            hidden_matching_matrix.append(mul2)

        print 'Matching matrix computation done.'


def main():
    SMN(set_params.ParamsClass(), set_dir.Directory('TR'))


if __name__ == '__main__':
    main()
