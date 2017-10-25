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
        self.get_initial_hidden_state()
        word_matching_matrix, hidden_emb_matching_matrix = self.compute_matching_matrix()
        conv_hidden_output, conv_word_output = self.get_cnn_output(hidden_emb_matching_matrix, word_matching_matrix)
        accumulated_word_match, accumulated_hidden_match = self.get_accumulated_match(conv_word_output, conv_hidden_output)
        self.get_final_hidden_state(accumulated_word_match, accumulated_hidden_match)
        # self.compute_loss()

    def get_cnn_output(self, hidden_emb_matching_matrix, word_matching_matrix):
        conv_word_output = self.conv_pipeline_init(word_matching_matrix, 'word_conv')
        conv_hidden_output = self.conv_pipeline_init(hidden_emb_matching_matrix, 'hidden_conv')
        return conv_hidden_output, conv_word_output

    def get_initial_hidden_state(self):
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

    def create_rnn_cell(self, name, option='lstm'):
        if option == 'lstm':
            with tf.variable_scope(name):
                rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.params.RNN_HIDDEN_DIM, forget_bias=1.0)
                rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=self.params.keep_prob)
                return rnn_cell

        elif option == 'gru':
            with tf.variable_scope(name):
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
        return word_matching_matrix, hidden_matching_matrix

    def conv_layer(self, conv_input, filter_shape, num_filters, stride, padding, name):
        with tf.variable_scope(name) as scope:
            try:
                weights = tf.get_variable(name='weights', shape=filter_shape, regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                biases = tf.get_variable(name='biases', shape=[num_filters], regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                         initializer=tf.constant_initializer(0.0))
            except ValueError:
                scope.reuse_variables()
                weights = tf.get_variable(name='weights', shape=filter_shape, regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                          initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
                biases = tf.get_variable(name='biases', shape=[num_filters], regularizer=tf.contrib.layers.l2_regularizer(0.0),
                                         initializer=tf.constant_initializer(0.0))
            # conv_input_nhwc = tf.transpose(conv_input, perm=[1,2,3,0], name='conv_input_nhwc')
            # kernel = tf.transpose(weights, perm=[0,2,1,3], name='kernel_nhwc')
            # conv = tf.nn.conv2d(conv_input, filter=weights, strides=stride, padding=padding)
            conv = tf.nn.conv2d(conv_input, filter=weights, strides=stride, padding=padding)
            # scope.reuse_variables()
            return tf.nn.relu(conv + biases)

    def conv_output(self, conv_input, kernel, strides, num_filters, padding, name):
        return self.conv_layer(conv_input, kernel, num_filters, strides, padding, name)

    def pool_output(self, pool_input, ksize, stride, padding, name):
        return tf.nn.max_pool(pool_input, ksize, stride, padding, name='pool')

    def conv_pipeline_init(self, cnn_input, conv_name):
        num_filters = self.params.num_filters

        all_context_pool_output = []

        with tf.variable_scope(conv_name):
            for i in range(len(cnn_input)):
                # curr_context_utt_view_word_feature = tf.nn.embedding_lookup(self.word_emb_matrix, tf.squeeze(cnn_input[i]), name='utt_context_emb_' + str(i))
                pool_output = []
                for layer_num in range(len(self.params.filter_width)):
                    curr_pool_output = self.conv_layer_pipeline(layer_num, num_filters, tf.expand_dims(cnn_input[i], -1), conv_name)
                    # scope.reuse_variables()
                    pool_output.append(curr_pool_output)
                print('Context ' + str(i) + ' convolution and max-pool ' + conv_name + ': DONE')
                all_context_pool_output.append(pool_output)
            return all_context_pool_output

    def conv_layer_pipeline(self, layer_num, num_filters, cnn_input, conv_name):

        filter_width = self.params.filter_width[layer_num]
        # num_filters = self.params.num_filters[layer_num]

        # with tf.variable_scope('utt_view_kernel_' + str(layer_num)) as scope:
        conv1_output = self.conv_output(cnn_input, [filter_width, self.params.MAX_RESP_UTT_LENGTH, 1, num_filters], [1, 1, 1, 1], num_filters, 'VALID', 'conv1_' + str(layer_num))
        # print('Context ' + str(layer_num) + ' convolution ' + conv_name + ': DONE')
        # scope.reuse_variables()

        pool2_output = self.pool_output(conv1_output, ksize=[1, 10, 1, 1], stride=[1, 3, 1, 1], padding='VALID', name='pool2')
        # print('Context' + str(layer_num) + ' max pooling ' + conv_name + ': DONE')
        # scope.reuse_variables()

        # print(conv_name + ' CNN pipeline: DONE')

        return pool2_output

    def get_accumulated_match(self, word_input, hidden_input):
        with tf.variable_scope('matching_accumulation'):
            concat_word_input = []
            concat_hidden_input = []
            for i in range(len(word_input)):
                concat_word_input.append(tf.concat(word_input[i], axis=1))
                concat_hidden_input.append(tf.concat(hidden_input[i], axis=1))

            flat_word_input = []
            flat_hidden_input = []

            word_input_axis_1_dim = concat_word_input[0][0].shape.dims[0].value
            hidden_input_axis_1_dim = concat_hidden_input[0][0].shape.dims[0].value

            for i in range(len(word_input)):
                flat_word_input.append(tf.expand_dims(tf.reshape(concat_word_input, [-1, word_input_axis_1_dim * self.params.num_filters]), 1))
                flat_hidden_input.append(tf.expand_dims(tf.reshape(concat_hidden_input, [-1, hidden_input_axis_1_dim * self.params.num_filters]), 1))

            accumulated_word_match = tf.concat(flat_word_input, axis=1)
            accumulated_hidden_match = tf.concat(flat_hidden_input, axis=1)

            return accumulated_word_match, accumulated_hidden_match

    def get_final_hidden_state(self, word_input, hidden_input):
        with tf.variable_scope('final_layer'):
            final_input = tf.concat([word_input, hidden_input], axis=2)
            rnn_cell = self.create_rnn_cell('last_layer', option=self.params.rnn)

            final_output, final_state = tf.nn.dynamic_rnn(rnn_cell,
                                                                          final_input,
                                                                          self.num_ctx_placeholders,
                                                                          dtype=tf.float32)

            if self.params.rnn == 'lstm':
                final_state = final_state[0]

            print 'Extracted rnn hidden states.'



def main():
    SMN(set_params.ParamsClass(), set_dir.Directory('TR'))


if __name__ == '__main__':
    main()
