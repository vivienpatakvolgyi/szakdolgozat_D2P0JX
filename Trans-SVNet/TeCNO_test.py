import pickle
import time
from datetime import datetime

import numpy as np
import torch
from torch.nn import DataParallel

import _functions
import mstcn
from _functions import log
import os


def test_model(model_name, test_labels_80, test_num_each_80, test_start_vidx):


    model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features,
                                  mstcn_causal_conv)


    model.load_state_dict(torch.load(model_name))
    model.cuda()

    test_we_use_start_idx_80 = [x for x in range(2)]

    progressmax = len(test_we_use_start_idx_80)
    torch.cuda.empty_cache()
    prg_i = 0

    start_time = time.time()
    test_corrects_phase = 0
    test_all_preds_phase = []
    test_all_labels_phase = []
    test_acc_each_video = []
    test_start_time = time.time()

    with torch.no_grad():
        for i in test_we_use_start_idx_80:
            labels_phase = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                labels_phase.append(test_labels_80[j][0])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
            else:
                labels_phase = labels_phase

            long_feature = _functions.get_long_feature(start_index=test_start_vidx[i],
                                                       lfb=g_LFB_test, LFB_length=test_num_each_80[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_classes = model.forward(video_fe)
            stages = y_classes.shape[0]
            _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
            p_classes = y_classes[-1].squeeze().transpose(1, 0)

            test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / test_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

            prg_i += 1
            prg_s = prg_i / progressmax * 100
            el_time, el_time_s, _, _ = _functions.run_stats(prg_s, start_time)
            if i == 0:
                rem_time_s = '0:00:00'
            stats = '\rStatus: {:4.2f}% (test)  Elapsed time: {:8s}  Est time: {:8s} '.format(
                prg_s, el_time_s, rem_time_s)
            print(stats, end=' ')

    test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
    test_acc_video = np.mean(test_acc_each_video)
    test_elapsed_time = time.time() - test_start_time

    log('leng of all preds: {}'.format(len(test_all_preds_phase)))
    save_test = int("{:4.0f}".format(test_accuracy_phase * 10000))
    model_pure_name, _ = os.path.splitext(model_name)
    print(model_pure_name)
    pred_name = model_pure_name + '_test_' + str(save_test) + '.pkl'

    with open(pred_name, 'wb') as f:
        pickle.dump(test_all_preds_phase, f)

    el_time, el_time_s, rem_time_s, speed = _functions.run_stats(100, start_time)
    epoch_stat = '\rElapsed time: {}    Estimated remaining time: {} '.format(
        el_time_s,
        rem_time_s)
    for x in range(len(stats) - len(epoch_stat) + 1):
        epoch_stat = epoch_stat + ' '
    log(epoch_stat)

    log(' test in: {:2.0f}m{:2.0f}s\n'
        ' test accu(phase): {:.4f}\n'
        ' test accu(video): {:.4f}\n'
        .format(test_elapsed_time // 60,
                test_elapsed_time % 60,
                test_accuracy_phase,
                test_acc_video))



if __name__ == "__main__":
    out_features = 7
    batch_size = 1
    mstcn_causal_conv = True
    mstcn_layers = 8
    mstcn_f_maps = 32
    mstcn_f_dim = 2048
    mstcn_stages = 2

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    logfile = '{}\\{}'.format('log', datetime.now().strftime("TeCNO_test_%Y%m%d_%H%M%S.log"))
    _functions.logging.basicConfig(filename=logfile, level=_functions.logging.INFO)

    _, _, _, _, _, _, test_labels_80, test_num_each_80, test_start_vidx = _functions.get_data2(
        './train_val_paths_labels_21.pkl')

    with open("./LFB/g_LFB50_test0.pkl", 'rb') as f:
        g_LFB_test = pickle.load(f)

    log("g_LFB_test shape: {}".format(g_LFB_test.shape))

    pthname = input('Please enter the model name (*.pth filename):\n')
    model_name = "./best_model/TeCNO/" + pthname.split('.')[0] + ".pth"

    torch.cuda.empty_cache()
    test_model(model_name, test_labels_80, test_num_each_80, test_start_vidx)
