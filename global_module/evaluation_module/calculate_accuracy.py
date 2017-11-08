import sys
from global_module.settings_module import set_dir

# root_folder =

dir_obj = set_dir.Directory('TE')
cost_file = open(dir_obj.test_cost_path, 'r')
output = open(dir_obj.test_pred_path, 'w')
output_seq = open(dir_obj.test_seq_op_path, 'w')

test_seq_file = open(dir_obj.data_filename, 'r')

count_iter = 1
min_pos = -1
step_val = 6
pred_ans = ''
pred_goal = ''
pred_slot = ''
multiplier = -1.0 # if probability, 1.0 if cost

min_cost = sys.float_info.max
for costLine, pred_line in zip(cost_file, test_seq_file):
    costLine = costLine.rstrip()
    ans = pred_line.rstrip()

    cost = multiplier * float(costLine)
    if (count_iter < step_val):
        if (min_cost > cost):
            min_cost = cost
            min_pos = count_iter
            pred_ans = ans
        count_iter += 1
    else:
        if (min_cost > cost):
            min_pos = step_val
            pred_ans = ans
        #print min_cost
        output.write(str(min_pos) + '\t' + str(min_cost) + "\n")
        output_seq.write(pred_ans + "\n")
        count_iter = 1
        min_cost = sys.float_info.max
        min_pos = -1

cost_file.close()
output.close()
output_seq.close()
test_seq_file.close()


output_seq = open(dir_obj.test_seq_op_path, 'r')
test_act_file = open(dir_obj.gold_data, 'r')
unmatched_file = open(dir_obj.output_path + '/unmatched.file', 'w')
output_seq_info = open(dir_obj.output_path + '/seq_selected_with_info.txt', 'w')
correct_incorrect_file = open(dir_obj.output_path + '/correct_incorrect.txt', 'w')

total = 0.0
correct = 0.0

for l1, l2 in zip(output_seq, test_act_file):
    total += 1
    l1 = l1.rstrip()
    l2 = l2.rstrip()
    if(l1.lower() == l2.lower()):
        correct += 1
        output_seq_info.write(l1 + "\t" + "CORRECT" + "\n")
        correct_incorrect_file.write("correct\n")
    else:
        output_seq_info.write(l1 + "\t" + "INCORRECT" + "\n")
        unmatched_file.write(str(int(total)) + "\t" + "==PREDICTED==\t" + l1.split("\t")[-1] + "\t" + "==CORRECT==\t" + l2.split("\t")[-1] + "\n")
        correct_incorrect_file.write("incorrect\n")

print(correct, total)
output_seq.close()
output_seq_info.close()
test_act_file.close()
unmatched_file.close()
correct_incorrect_file.close()
