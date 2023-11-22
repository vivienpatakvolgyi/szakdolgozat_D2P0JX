# some codes adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet
# and https://github.com/tobiascz/TeCNO
import copy
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torchinfo import summary

import _functions
import mstcn
from _functions import log


def train_model(train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx):

    criterion_phase = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device))
    # criterion_phase = nn.CrossEntropyLoss()

    model = mstcn.MultiStageModel(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features,
                                  mstcn_causal_conv)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    from torchinfo import summary
    log(summary(model, verbose=0))
    log('\n\n')
    input('waithere')

    train_we_use_start_idx_80 = [x for x in range(17)]
    val_we_use_start_idx_80 = [x for x in range(2)]

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy_phase = 0.0
    correspond_train_acc_phase = 0.0
    best_epoch = 0
    progressmax = max_epochs * (len(train_we_use_start_idx_80) + len(val_we_use_start_idx_80))
    torch.cuda.empty_cache()
    prg_i = 0

    start_time = time.time()
    for epoch in range(max_epochs):
        torch.cuda.empty_cache()
        train_idx_80 = []
        model.train()
        train_loss_phase = 0.0
        train_corrects_phase = 0
        batch_progress = 0.0
        running_loss_phase = 0.0
        minibatch_correct_phase = 0.0
        train_start_time = time.time()
        for i in train_we_use_start_idx_80:
            optimizer.zero_grad()
            labels_phase = []
            for j in range(train_start_vidx[i], train_start_vidx[i] + train_num_each_80[i]):
                labels_phase.append(train_labels_80[j][0])
            labels_phase = torch.LongTensor(labels_phase)
            if use_gpu:
                labels_phase = labels_phase.to(device)
                if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                    print(torch.cuda.memory_reserved() // (1024 * 1024 * 1024))
                    return -1
            else:
                labels_phase = labels_phase
            long_feature = _functions.get_long_feature(start_index=train_start_vidx[i],
                                                       lfb=g_LFB_train, LFB_length=train_num_each_80[i])

            long_feature = (torch.Tensor(long_feature)).to(device)
            video_fe = long_feature.transpose(2, 1)

            y_classes = model.forward(video_fe)
            stages = y_classes.shape[0]
            clc_loss = 0
            for j in range(stages):  ### make the interuption free stronge the more layers.
                p_classes = y_classes[j].squeeze().transpose(1, 0)
                ce_loss = criterion_phase(p_classes, labels_phase)
                clc_loss += ce_loss
            clc_loss = clc_loss / (stages * 1.0)

            _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)

            loss = clc_loss
            # print(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()

            running_loss_phase += clc_loss.data.item()
            train_loss_phase += clc_loss.data.item()

            batch_corrects_phase = torch.sum(preds_phase == labels_phase.data)
            train_corrects_phase += batch_corrects_phase
            minibatch_correct_phase += batch_corrects_phase

            prg_i += 1
            batch_progress += 1
            prg_s = prg_i / progressmax * 100
            el_time, el_time_s, _, _ = _functions.run_stats(prg_s, start_time)
            percent = batch_progress * batch_size / len(train_we_use_start_idx_80) * 100
            if epoch == 0:
                rem_time_s = '0:00:00'
            stats = '\rStatus: {:4.2f}% (training)  Epoch: {:4.2f}%  Elapsed time: {:8s}  Est time: {:8s} '.format(
                prg_s, min(100, percent), el_time_s, rem_time_s)
            print(stats, end=' ')

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_phase = float(train_corrects_phase) / len(train_labels_80)
        train_average_loss_phase = train_loss_phase

        # Sets the module in evaluation mode.
        model.eval()
        val_loss_phase = 0.0
        val_corrects_phase = 0
        val_start_time = time.time()
        val_progress = 0
        val_all_preds_phase = []
        val_all_labels_phase = []
        val_acc_each_video = []
        with torch.no_grad():
            for i in val_we_use_start_idx_80:
                labels_phase = []
                for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
                    labels_phase.append(val_labels_80[j][0])
                labels_phase = torch.LongTensor(labels_phase)
                if use_gpu:
                    labels_phase = labels_phase.to(device)
                    if torch.cuda.memory_reserved() > 0.8 * 8 * 1024 * 1024 * 1024:
                        print(torch.cuda.memory_reserved() // (1024 * 1024 * 1024))
                        return -2
                else:
                    labels_phase = labels_phase

                long_feature = _functions.get_long_feature(start_index=val_start_vidx[i],
                                                           lfb=g_LFB_val, LFB_length=val_num_each_80[i])

                long_feature = (torch.Tensor(long_feature)).to(device)
                video_fe = long_feature.transpose(2, 1)

                y_classes = model.forward(video_fe)
                stages = y_classes.shape[0]
                clc_loss = 0
                for j in range(stages):  ### make the interuption free stronge the more layers.
                    p_classes = y_classes[j].squeeze().transpose(1, 0)
                    ce_loss = criterion_phase(p_classes, labels_phase)
                    clc_loss += ce_loss
                clc_loss = clc_loss / (stages * 1.0)

                _, preds_phase = torch.max(y_classes[stages - 1].squeeze().transpose(1, 0).data, 1)
                p_classes = y_classes[-1].squeeze().transpose(1, 0)
                loss_phase = criterion_phase(p_classes, labels_phase)

                val_loss_phase += loss_phase.data.item()

                val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
                val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / val_num_each_80[i])
                # TODO

                for j in range(len(preds_phase)):
                    val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
                for j in range(len(labels_phase)):
                    val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

                prg_i += 1
                val_progress += 1
                prg_s = prg_i / progressmax * 100
                el_time, el_time_s, _, _ = _functions.run_stats(prg_s, start_time)
                percent = val_progress * batch_size / len(val_we_use_start_idx_80) * 100
                if i == 0:
                    rem_time_s = '0:00:00'
                stats = '\rStatus: {:4.2f}% (validation)  Epoch: {:4.2f}%  Elapsed time: {:8s}  Est time: {:8s} '.format(
                    prg_s, min(100, percent), el_time_s, rem_time_s)
                print(stats, end=' ')

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
        val_acc_video = np.mean(val_acc_each_video)
        val_average_loss_phase = val_loss_phase

        val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro',
                                                zero_division=0)
        val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro',
                                                      zero_division=0)
        val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro',
                                                  zero_division=0)
        val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None,
                                                           zero_division=0)
        val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None,
                                                     zero_division=0)
        el_time, el_time_s, rem_time_s, speed = _functions.run_stats(100, start_time)
        epoch_stat = '\rEpoch:  {:3d} /{:3d}   Elapsed time: {}    Estimated remaining time: {} '.format(
            epoch,
            max_epochs - 1,
            el_time_s,
            rem_time_s)
        for x in range(len(stats) - len(epoch_stat) + 1):
            epoch_stat = epoch_stat + ' '
        log(epoch_stat)

        log(' train in          : {:2.0f}m{:2.0f}s\n'
            ' train loss(phase) : {:4.4f}\n'
            ' train accu(phase) : {:.4f}\n'
            ' valid in          : {:2.0f}m{:2.0f}s\n'
            ' valid loss(phase) : {:4.4f}\n'
            ' valid accu(phase) : {:.4f}\n'
            .format(train_elapsed_time // 60,
                    train_elapsed_time % 60,
                    train_average_loss_phase,
                    train_accuracy_phase,
                    val_elapsed_time // 60,
                    val_elapsed_time % 60,
                    val_average_loss_phase,
                    val_accuracy_phase))

        log("val_precision_each_phase   : {}".format(val_precision_each_phase))
        log("val_recall_each_phase      : {}".format(val_recall_each_phase))
        log("val_precision_phase        : {}".format(val_precision_phase))
        log("val_recall_phase           : {}".format(val_recall_phase))
        log("val_jaccard_phase          : {}\n".format(val_jaccard_phase))

        if val_accuracy_phase > best_val_accuracy_phase:
            best_val_accuracy_phase = val_accuracy_phase
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        save_val_phase = int("{:4.0f}".format(best_val_accuracy_phase * 10000))
        save_train_phase = int("{:4.0f}".format(correspond_train_acc_phase * 10000))
        base_name = "TeCNO50" \
                    + "_epoch_" + str(best_epoch) \
                    + "_train_" + str(save_train_phase) \
                    + "_val_" + str(save_val_phase)
    torch.save(best_model_wts, "./best_model/TeCNO/" + base_name + ".pth")
    log("best_epoch: {}".format(str(best_epoch)))
    log(summary(model, verbose=0))
    return 1


if __name__ == "__main__":
    out_features = 7
    batch_size = 56
    mstcn_causal_conv = True
    learning_rate = 5e-4
    max_epochs = 20
    mstcn_layers = 8
    mstcn_f_maps = 32
    mstcn_f_dim = 2048
    mstcn_stages = 2

    weights_train = np.asarray([1.6411019141231247,
                                0.19090963801041133,
                                1.0,
                                0.2502662616859295,
                                1.9176363911137977,
                                0.9840248158200853,
                                2.174635818337618, ])

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    logfile = '{}\\{}'.format('log', datetime.now().strftime("TransSV_p3_%Y%m%d_%H%M%S.log"))
    _functions.logging.basicConfig(filename=logfile, level=_functions.logging.INFO)

    train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx, _, _, _ = _functions.get_data2(
        './train_val_paths_labels_21.pkl')

    with open("./LFB/g_LFB50_train0.pkl", 'rb') as f:
        g_LFB_train = pickle.load(f)
    with open("./LFB/g_LFB50_val0.pkl", 'rb') as f:
        g_LFB_val = pickle.load(f)

    log("g_LFB_train shape:".format(g_LFB_train.shape))
    log("g_LFB_val shape:".format(g_LFB_val.shape))

    suc = 0
    while suc != 1:
        torch.cuda.empty_cache()
        suc = train_model(train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80,
                          val_start_vidx)
        ntrbatch = batch_size - int(min((batch_size) / 2, 8) // 1)
        nvbatch = batch_size - int(min((batch_size) / 2, 8) // 1)
        if suc == -1:
            log('\nTrain batch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(
                batch_size,
                ntrbatch))
            batch_size = ntrbatch
        if suc == -2:
            log(
                '\nValidation batch number ({}) was too high to fit in memory. Trying lower ({}) number.\n'.format(
                    batch_size, nvbatch))
            batch_size = nvbatch

        if batch_size < 16:
            print(
                '\nToo large images for training to input into the network, please configure the resize parameters lower')
            break
        if batch_size < 1:
            print(
                '\nToo large images for validation to input into the network, please configure the resize parameters lower')
            break
