import random
def create_negative_sampled_data(training_filename, input_filename):
    training_file = open(training_filename, 'r').readlines()
    neg_examples = 5

    op_utt_file = open(input_filename + '_UTT_' + str(neg_examples) + '_NEGATIVE.txt', 'w')
    op_label_file = open(input_filename + '_LABEL_' + str(neg_examples) + '_NEGATIVE.txt', 'w')

    agent_resp_to_index_map = {}
    agent_index_to_resp_map = {}

    count = 0

    for each_line in training_file:
        agent_resp = each_line.strip().split('\t')[-1]
        if(not agent_resp_to_index_map.has_key(agent_resp)):
            agent_resp_to_index_map[agent_resp] = count
            agent_index_to_resp_map[count] = agent_resp
            count += 1

    print len(agent_resp_to_index_map)

    end = len(agent_resp_to_index_map)


    input_file = open(input_filename, 'r')
    for each_line in input_file:
        line_split = each_line.strip().split('\t')
        agent_index = agent_resp_to_index_map[line_split[-1]]

        r = range(0, agent_index) + range(agent_index+1, end)

        random.shuffle(r)

        context = '\t'.join(line_split[:-1]).strip()

        op_utt_file.write(each_line.strip() + '\n')
        op_label_file.write('1\n')

        for i in range(neg_examples):
            op_utt_file.write(context + '\t' + agent_index_to_resp_map[r[i]] + '\n')
            op_label_file.write('0\n')

    op_utt_file.close()
    op_label_file.close()

training_filename = '/home/aykumar/aykumar_home/self/sequential_match/global_module/utility_dir/folder1/data/base_data/train/jpmc_demo_train_raw_formatted.txt_tokenized_4_context_seq.txt'
valid_data = '/home/aykumar/aykumar_home/self/sequential_match/global_module/utility_dir/folder1/data/base_data/valid/jpmc_demo_valid_raw_transformed.txt_tokenized_4_context_seq.txt'


create_negative_sampled_data(training_filename, valid_data)