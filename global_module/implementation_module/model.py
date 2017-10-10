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
        self.extract_hidden_embedding('layer1')

    def create_placeholders(self):
        self.num_context_with_resp = self.params.NUM_CONTEXT + 1
        self.ctx_with_resp = tf.placeholder(dtype=tf.int32,
                                            shape=[None,
                                                   self.num_context_with_resp,
                                                   self.params.MAX_UTT_SEQ_LENGTH],
                                            name='ctx_resp_placeholder')

        self.seq_len_placeholders = tf.placeholder(dtype=tf.int32,
                                                   shape=[None, self.num_context_with_resp],
                                                   name='seq_len_placeholder')

        self.ctx_len_placeholders = tf.placeholder(dtype=tf.int32,
                                                   shape=[None],
                                                   name='ctx_len_placeholder')

        self.label = tf.placeholder(dtype=tf.float32,
                                    shape=[None],
                                    name='response_label')

    def extract_word_embedding(self):
        self.word_emb_matrix = tf.get_variable("word_embedding_matrix",
                                               shape=[self.params.vocab_size, self.params.EMB_DIM],
                                               dtype=tf.float32,
                                               regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                               trainable=self.params.is_word_trainable)

        self.ctx_resp_word_emb = tf.nn.embedding_lookup(params=self.word_emb_matrix,
                                                        ids=self.ctx_with_resp,
                                                        name='ctx_resp_word_emb',
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
                rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.params.RNN_HIDDEN_DIM, forget_bias=1.0)
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.params.keep_prob)
                return rnn_cell

    def extract_hidden_embedding(self, name):
        with tf.variable_scope('rnn_layer') as scope:
            self.rnn_cell = self.create_rnn_cell(name, self.params.rnn)
            reshaped_input = tf.reshape(self.ctx_resp_word_emb, shape=[-1, self.params.MAX_UTT_SEQ_LENGTH, self.params.EMB_DIM])
            reshaped_length = tf.reshape(self.seq_len_placeholders, shape=[-1])
            rnn_output, rnn_state = tf.nn.dynamic_rnn(self.rnn_cell,
                                                      reshaped_input,
                                                      reshaped_length,
                                                      dtype=tf.float32)
            self.rnn_output = tf.reshape(rnn_output,
                                         shape=[-1, self.num_context_with_resp, self.params.MAX_UTT_SEQ_LENGTH, self.params.EMB_DIM],
                                         name='layer1_output')
            self.rnn_state = tf.reshape(rnn_state,
                                        shape=[-1, self.num_context_with_resp, self.params.RNN_HIDDEN_DIM],
                                        name='layer1_state')

            print 'Extracted rnn hidden states.'


def main():
    SMN(set_params.ParamsClass(), set_dir.Directory('TR'))


if __name__ == '__main__':
    main()
