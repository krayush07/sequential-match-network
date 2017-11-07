import numpy as np
from global_module.settings_module import ParamsClass, Directory, Dictionary


class DataReader:
    def __init__(self, params):
        self.params = params

    def get_index_string(self, utt, word_dict):
        index_string = ''
        for each_token in utt.split():
            if (ParamsClass('TR').all_lowercase):
                if (word_dict.has_key(each_token.lower())):
                    each_token = each_token.lower()
                elif (word_dict.has_key(each_token)):
                    each_token = each_token
                elif (word_dict.has_key(each_token.title())):
                    each_token = each_token.title()
                elif (word_dict.has_key(each_token.upper())):
                    each_token = each_token.upper()
                else:
                    each_token = each_token.lower()

            index_string += str(word_dict.get(each_token, word_dict.get("UNK"))) + '\t'
        return len(index_string.strip().split()), index_string.strip()

    def pad_string(self, id_string, curr_len, max_seq_len):
        id_string = id_string.strip() + '\t'
        while curr_len < max_seq_len:
            id_string += '0\t'
            curr_len += 1
        return id_string.strip()

    def add_dummy_context_string(self, curr_context_string, curr_num_context, max_num_context, indiv_max_seq_len):
        for i in range(curr_num_context, max_num_context):
            context_string = ''
            for j in range(indiv_max_seq_len):
                context_string += '0\t'
            curr_context_string.append(context_string.strip())
        return curr_context_string

    def strip_extra_sequence(self, feature_id_string_arr, max_seq_len):
        for i in range(len(feature_id_string_arr)):
            string_id_split = feature_id_string_arr[i].split('\t')[:max_seq_len]
            string_id_string = '\t'.join(string_id_split)
            feature_id_string_arr[i] = string_id_string.strip()
        return feature_id_string_arr

    def format_string(self, inp_string, curr_string_len, max_len):
        if curr_string_len > max_len:
            print('Maximum SEQ LENGTH reached. Stripping extra sequence.\n')
            op_string = '\t'.join(inp_string.split('\t')[:max_len])
        else:
            op_string = self.pad_string(inp_string, curr_string_len, max_len)
        return op_string

    def generate_id_map(self, data_filename, label_filename, index_arr, dict_obj):
        data_file_arr = open(data_filename, 'r').readlines()
        label_file_arr = open(label_filename, 'r').readlines()

        global_ctx_arr = []
        global_ctx_len_arr = []
        global_num_ctx_arr = []
        global_resp_arr = []
        global_resp_len_arr = []
        global_label_arr = []

        for curr_id, each_idx in enumerate(index_arr):
            curr_line = data_file_arr[each_idx].strip()
            curr_label = label_file_arr[each_idx].strip()

            data_line_split = curr_line.split('\t')
            curr_num_context = len(data_line_split) - 1

            curr_ctx_seq_arr = []
            curr_ctx_len_arr = [0 for _ in range(self.params.NUM_CONTEXT)]

            for idx in range(len(data_line_split) - 1):
                each_split = data_line_split[idx]
                curr_string_len, curr_index_string = self.get_index_string(each_split, dict_obj.word_dict)
                curr_utt_string = self.format_string(curr_index_string, curr_string_len, self.params.MAX_CTX_UTT_LENGTH)  # format each utt string
                curr_ctx_seq_arr.append(curr_utt_string)
                curr_ctx_len_arr[idx] = curr_string_len

            curr_ctx_seq_arr = self.add_dummy_context_string(curr_ctx_seq_arr, curr_num_context, self.params.NUM_CONTEXT, self.params.MAX_CTX_UTT_LENGTH)

            resp_string_len, resp_index_string = self.get_index_string(data_line_split[curr_num_context], dict_obj.word_dict)
            resp_index_string = self.format_string(resp_index_string, resp_string_len, self.params.MAX_RESP_UTT_LENGTH)  # format resp string

            global_ctx_arr.append(curr_ctx_seq_arr)
            global_ctx_len_arr.append(curr_ctx_len_arr)
            global_num_ctx_arr.append(curr_num_context)
            global_resp_arr.append(resp_index_string)
            global_resp_len_arr.append(resp_string_len)
            global_label_arr.append(curr_label)

        print('Reading: DONE')
        return global_ctx_arr, global_ctx_len_arr, global_num_ctx_arr, global_resp_arr, global_resp_len_arr, global_label_arr

    def data_iterator(self, data_filename, label_filename, index_arr, dict_obj):
        ctx_arr, ctx_len_arr, num_ctx_arr, resp_arr, resp_len_arr, label_arr = self.generate_id_map(data_filename, label_filename, index_arr, dict_obj)

        batch_size = self.params.batch_size
        num_batches = len(index_arr) / self.params.batch_size

        for i in range(num_batches):
            curr_ctx_arr = ctx_arr[i * batch_size: (i + 1) * batch_size]
            curr_ctx_len_arr = ctx_len_arr[i * batch_size: (i + 1) * batch_size]

            curr_ctx_list = []
            curr_ctx_len_list = []

            for p in range(batch_size):
                curr_ctx_list.append(np.loadtxt(curr_ctx_arr[p], dtype=np.int32))
                curr_ctx_len_list.append(np.loadtxt(curr_ctx_len_arr[p], dtype=np.int32))

            curr_ctx_arr = np.array(curr_ctx_list)
            curr_ctx_len_arr = np.array(curr_ctx_len_list)
            curr_num_ctx_arr = np.array(num_ctx_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
            curr_resp_arr = np.loadtxt(resp_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
            curr_resp_len_arr = np.array(resp_len_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)
            curr_label_arr = np.array(label_arr[i * batch_size: (i + 1) * batch_size], dtype=np.int32)

            if batch_size == 1:
                curr_resp_arr = np.expand_dims(curr_resp_arr, axis=0)

            yield (curr_ctx_arr, curr_ctx_len_arr, curr_num_ctx_arr, curr_resp_arr, curr_resp_len_arr, curr_label_arr)
            # print("A")


def getLength(fileName):
    print('Reading :', fileName)
    dataFile = open(fileName, 'r')
    count = 0
    for line in dataFile:
        count += 1
    dataFile.close()
    return count, np.arange(count)


def main():
    data_file = '/home/aykumar/aykumar_home/self/sequential_match/global_module/utility_dir/folder1/data/raw_tokenized_train.txt'
    label_file = '/home/aykumar/aykumar_home/self/sequential_match/global_module/utility_dir/folder1/data/label_train.txt'
    count, utterance_idx = getLength(data_file)
    dictObj = Dictionary()
    config = ParamsClass()
    flag = 'TR'

    a, b, c, d, e, f = DataReader(config).data_iterator(config, data_file, label_file, utterance_idx, dictObj)

    # for step, (a, b, c, d, e, f, g, h) in enumerate(data_iterator(config, data_file, label_file, utterance_idx, dictObj)):
    #     print 1
    # np.savetxt('a.txt', a, fmt='%d')
    # np.savetxt('b.txt', b, fmt='%d')
    # np.savetxt('c.txt', c, fmt='%d')
    # np.savetxt('d.txt', d, fmt='%d')
    # np.savetxt('e.txt', e, fmt='%d')
    # np.savetxt('f.txt', f, fmt='%d')
    # print g
    # np.savetxt('g.txt', np.squeeze(g), fmt='%d')
    # np.savetxt('h.txt', h)


if __name__ == '__main__':
    main()
