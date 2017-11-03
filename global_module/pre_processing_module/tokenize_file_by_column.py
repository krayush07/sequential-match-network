# -*- coding: utf-8 -*-
import nltk

def tokenize_by_column(filename, column_num=0):
    lines = open(filename, 'r')
    op_file = open(filename + '_tokenized', 'w')

    for line in lines:
        line_split = line.strip().split('\t')
        print (line)
        initial_string = '\t'.join(line_split[0:column_num]).strip()
        tokenized_string = str(' '.join(nltk.word_tokenize(line_split[column_num]))).strip()
        last_string = '\t'.join(line_split[column_num+1:]).strip()
        final_string = initial_string + '\t' + str(tokenized_string) + '\t' + last_string
        op_file.write(final_string.strip() + '\n')

    op_file.close()

tokenize_by_column('/home/aykumar/aykumar_home/self/sequential_match/global_module/utility_dir/folder1/data/base_data/valid/jpmc_demo_valid_raw_transformed.txt', 0)