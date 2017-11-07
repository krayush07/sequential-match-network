def generate_4context_seq():
    input_file = '/home/aykumar/aykumar_home/self/sequential_match/global_module/utility_dir/folder1/data/base_data/test/jpmc_demo_test_raw_transformed.txt_tokenized'
    conv_file = open(input_file, 'r')
    end_str = '======================='

    seq_file = open(input_file + '_4_context_seq.txt', 'w')

    conv_list = []
    total_4context_seq = 0

    for line in conv_file:
        line = line.strip()
        if(line.startswith(end_str)):
            conv_len = len(conv_list)
            print(conv_len)
            total_4context_seq += (conv_len - 5) / 2 + 1
            seq_file.write(conv_list[0] + "\t" + conv_list[1] + "\t" + conv_list[2] + "\n")
            for i in range(0, conv_len, 2):
                if(i+4 < conv_len):
                    seq_file.write(conv_list[i] + "\t" + conv_list[i+1] + "\t" + conv_list[i+2] + "\t" + conv_list[i+3] + "\t" + conv_list[i+4] + "\n")
            conv_list = []
        else:
            conv_list.append(line)
    print(total_4context_seq)
    seq_file.close()
    conv_file.close()

generate_4context_seq()