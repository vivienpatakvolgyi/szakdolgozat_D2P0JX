import argparse
import os
import pickle
import shutil

import numpy as np


with open('./train_val_paths_labels_21.pkl', 'rb') as f:
    train_test_paths_labels = pickle.load(f)

    test_labels = train_test_paths_labels[5]
    test_num_each = train_test_paths_labels[8]

    parser = argparse.ArgumentParser(description='lstm testing')
    parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 4')
    parser.add_argument('-n', '--name',
                        default='TeCNO50_trans1_3_5_1_length_30_epoch_5_train_9779_val_8830_test_54.pkl',
                        type=str, help='name of model')
    parser.add_argument('-a', '--test', default=2, type=int, help='number of test video(s)')
    parser.add_argument('-r', '--root', default='Trans-SV', type=str, help='Root directory for videos')

    args, unknown = parser.parse_known_args()
    sequence_length = args.seq
    pred_name = 'best_model/Trans-SV/'+input('Please enter the name of the TeCNO prediction (.pkl file)\n').split('.')[0]+'.pkl'
    num_test = args.test
    root_dir = args.root

    test_labels = np.asarray(test_labels, dtype=np.int64)

    with open(pred_name, 'rb') as f:
        ori_preds = pickle.load(f)

    num_labels = len(test_labels)
    num_preds = len(ori_preds)

    print('num of labels  : {:6d}'.format(num_labels))
    print("num ori preds  : {:6d}".format(num_preds))
    print("num test  : {:6d}".format(num_test))
    print('(num_preds + (sequence_length - 1) * 2) : {:6d}'.format(num_preds + (sequence_length - 1) * num_test))
    print('num test labels : {:6d}'.format(len(test_labels)))
    print(test_num_each)
    print("labels example : ", test_labels[5000][0])
    print("preds example  : ", ori_preds[1])
    print(len(test_num_each) - 1)

if num_labels == (num_preds + (sequence_length - 1) * num_test):

    phase_dir = 'best_model/preds/'
    print(phase_dir)
    shutil.rmtree(phase_dir, ignore_errors=True)
    os.mkdir(phase_dir)
    phase_dict_key = ['Preparation', 'Dividing Ligament and Peritoneum', 'Dividing Uterine Vessels and Ligament',
                      'Transecting the Vagina',
                      'Specimen Removal', 'Suturing', 'Washing']
    preds_all = []
    count = 0
    for i in range(num_test):
        filename = phase_dir + 'transSV_video' + str(i + 1) + '-phase.txt'
        print(filename)
        f = open(filename, 'a')
        f.write('Frame  Phase')
        f.write('\n')
        preds_each = []
        for j in range(count, count + test_num_each[i] - (sequence_length - 1)):
            if j == count:
                for k in range(sequence_length - 1):
                    preds_each.append(ori_preds[j])
                    preds_all.append(ori_preds[j])
            preds_each.append(ori_preds[j])
            preds_all.append(ori_preds[j])
        for k in range(len(preds_each)):
            f.write(str(1 * k))
            f.write('\t')
            f.write(str(preds_each[k]))  # f.write(phase_dict_key[preds_each[k]])
            f.write('\n')
        f.close()
        count += test_num_each[i] - (sequence_length - 1)
    test_corrects = 0

    for i in range(len(test_labels)):
        if test_labels[i][0] == preds_all[i]:  # if test_labels[i][len(test_num_each) - 1] == preds_all[i]:
            test_corrects += 1

    print('last video num label: {:6d}'.format(test_num_each[-1]))
    print('last video num preds: {:6d}'.format(len(preds_each)))
    print('num of labels       : {:6d}'.format(num_labels))
    print('rsult of all preds  : {:6d}'.format(len(preds_all)))
    print('right number preds  : {:6d}'.format(test_corrects))
    print('test accuracy       : {:.4f}'.format(test_corrects / num_labels))
else:
    print('number error, please check')
