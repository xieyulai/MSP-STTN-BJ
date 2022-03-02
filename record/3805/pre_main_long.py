import warnings
warnings.filterwarnings('ignore')
import numpy as np
import time
import sys
import cv2
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse
import os
import math
from time import localtime, strftime
from sklearn import metrics

torch.backends.cudnn.benchmark = True
from util.util import timeSince, get_yaml_data
from util.util import weights_init, VALRMSE
from tensorboardX import SummaryWriter
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

TORCH_VERSION = torch.__version__

log_name = 'logs/taxibj_'

# seed = 777


class DataConfiguration:
    def __init__(self, Len_close, Len_period, Len_trend):
        super().__init__()

        # Data
        self.name = 'TaxiBJ'
        self.portion = 1.  # portion of data

        self.len_close = Len_close
        self.len_period = Len_period
        self.len_trend = Len_trend
        self.pad_forward_period = 0
        self.pad_back_period = 0
        self.pad_forward_trend = 0
        self.pad_back_trend = 0

        self.len_all_close = self.len_close * 1
        self.len_all_period = self.len_period * (1 + self.pad_back_period + self.pad_forward_period)
        self.len_all_trend = self.len_trend * (1 + self.pad_back_trend + self.pad_forward_trend)

        self.len_seq = self.len_all_close + self.len_all_period + self.len_all_trend
        self.cpt = [self.len_all_close, self.len_all_period, self.len_all_trend]

        self.interval_period = 1
        self.interval_trend = 7

        self.ext_flag = True
        self.ext_time_flag = True
        self.rm_incomplete_flag = True
        self.fourty_eight = True
        self.previous_meteorol = True

        self.dim_h = 32
        self.dim_w = 32


def run(mcof):
    IS_TRAIN = 0
    IS_VAL = 0
    ####SETTING####
    TASK_TYPE = mcof.task
    INP_TYPE = mcof.inp_type
    DATA_TYPE = mcof.dataset_type
    RECORD_ID = mcof.record
    PRESUME_RECORD_ID = mcof.presume_record
    EPOCH_S = mcof.epoch_s
    PRESUME_EPOCH_S = mcof.presume_epoch_s
    IS_REMOVE = mcof.is_remove

    if len(mcof.mode) > 1:
        if mcof.mode == 'train':
            IS_TRAIN = 1
            setting = get_yaml_data("./pre_setting_bj_long.yaml")
            BATCH_SIZE = setting['TRAIN']['BATCH_SIZE']
        if mcof.mode == 'val':
            IS_VAL = 1
            RECORD_ID = mcof.record
            setting = get_yaml_data(f"./record/{RECORD_ID}/pre_setting_bj_long.yaml")
            EVAL_BATCH = setting['TRAIN']['EVAL_BATCH']

    ####SETTING####
    IS_BEST_EVAL = setting['TRAIN']['IS_BEST_EVAL']
    DROPOUT = setting['TRAIN']['DROPOUT']
    MERGE = setting['TRAIN']['MERGE']
    PATCH_LIST = setting['TRAIN']['PATCH_LIST']
    IS_USING_SKIP = setting['TRAIN']['IS_USING_SKIP']
    MODEL_DIM = setting['TRAIN']['MODEL_DIM']
    ATT_NUM = setting['TRAIN']['ATT_NUM']
    CROSS_ATT_NUM = setting['TRAIN']['CROSS_ATT_NUM']
    IS_MASK_ATT = setting['TRAIN']['IS_MASK_ATT']
    IS_SCALE = setting['TRAIN']['IS_SCALE']
    LR_G = setting['TRAIN']['LR_G']
    EPOCH_E = setting['TRAIN']['EPOCH']
    WARMUP_EPOCH = setting['TRAIN']['WARMUP_EPOCH']
    MILE_STONE = setting['TRAIN']['MILE_STONE']
    LOSS_GEN = setting['TRAIN']['LOSS_GEN']
    LOSS_TIME = setting['TRAIN']['LOSS_TIME']
    LOSS_TYP = setting['TRAIN']['LOSS_TYP']
    LEN_CLOSE = setting['TRAIN']['LEN_CLOSE']
    LEN_PERIOD = setting['TRAIN']['LEN_PERIOD']
    LEN_TREND = setting['TRAIN']['LEN_TREND']
    LENGTH = setting['TRAIN']['LENGTH']
    IS_SEQ = setting['TRAIN']['IS_SEQ']
    #IS_REDUCE = setting['TRAIN']['IS_REDUCE']
    OUT_STYLE = setting['TRAIN']['OUT_STYLE']
    CAT_STYLE = setting['TRAIN']['CAT_STYLE']
    IS_AUX = setting['TRAIN']['IS_AUX']
    IS_C3D = setting['TRAIN']['IS_C3D']
    ONLY_CONV6 = setting['TRAIN']['ONLY_CONV6']
    ITERATION_STEP = setting['TRAIN']['ITERATION_STEP']
    EVAL_MODE = setting['TRAIN']['EVAL_MODE']
    BATCH_SIZE = setting['TRAIN']['BATCH_SIZE']
    EVAL_START_EPOCH = setting['TRAIN']['EVAL_START_EPOCH']
    print(setting)

    C = 2
    H = 32
    W = 32

    from dataset.dataset_long import DatasetFactory

    dconf = DataConfiguration(Len_close=LEN_CLOSE,
                              Len_period=LEN_PERIOD,
                              Len_trend=LEN_TREND,
                              )
    ds_factory = DatasetFactory(dconf, INP_TYPE, DATA_TYPE, LENGTH, IS_SEQ)

    if IS_TRAIN:

        try:
            if os.path.exists('./record/{}/'.format(RECORD_ID)):
                shutil.rmtree('./record/{}/'.format(RECORD_ID))
            os.makedirs('./record/{}/'.format(RECORD_ID))

            oldname = os.getcwd() + os.sep
            newname = f'./record/{RECORD_ID}/'
            shutil.copyfile(oldname + 'pre_setting_bj_long.yaml', newname + 'pre_setting_bj_long.yaml')
            shutil.copyfile(oldname + 'pre_main_long.py', newname + 'pre_main_long.py')
            shutil.copytree(oldname + 'net', newname + 'net')
            shutil.copytree(oldname + 'dataset', newname + 'dataset')
        except:
            raise print('record directory not find!')

        record = open("record/{}/log.txt".format(RECORD_ID), "w")

        curr_time = strftime('%y%m%d%H%M%S', localtime())
        Keep_Train = mcof.keep_train

        #### 数据加载和预处理 ###
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        train_ds = ds_factory.get_train_dataset()

        train_loader = DataLoader(
            dataset=train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=1
        )

        ####MODEL####
        input_channels = C

        P_list = eval(PATCH_LIST)
        # P_list = [[1,1],[3,4],[6,8],[12,16]]
        Is_scaling = IS_SCALE

        from net.imp_pos_cl_heat2heat import Prediction_Model as Model
        #from net.niu_imp_pos_cl_heat2heat import Prediction_Model as Model

        net = Model(
            mcof,
            Length=LENGTH,  # 8
            Width=W,  # 200
            Height=H,  # 200
            Input_dim=input_channels,  # 1
            Patch_list=P_list,  # 小片段的大小
            Dropout=DROPOUT,
            Att_num=ATT_NUM,  # 2
            Cross_att_num=CROSS_ATT_NUM,  # 2
            Using_skip=IS_USING_SKIP,  # 1
            Encoding_dim=MODEL_DIM,  # 256
            Embedding_dim=MODEL_DIM,  # 256
            #Is_reduce=IS_REDUCE,
            Is_mask=IS_MASK_ATT,  # 1
            Is_scaling=Is_scaling,  # 1
            Debugging=0,  # 0
            Merge=MERGE,  # cross-attention
            Out_style=OUT_STYLE,
            Cat_style=CAT_STYLE,
            Is_aux=IS_AUX,
            IS_C3D=IS_C3D,
            ONLY_CONV6=ONLY_CONV6,
        )

        ####TRAINING####
        print('LONG TRAINING START')
        print('-' * 30)

        writer_gen = SummaryWriter(f'runs/exp_gen/{curr_time[2:]}')
        writer_val = SummaryWriter(f'runs/exp_val/{curr_time[2:]}')
        print(f'Runs procedure in {curr_time[2:]} file')

        start = time.time()

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        device_ids = [i for i in range(torch.cuda.device_count())]

        #### Optimizer ####
        optimizer = optim.Adam(net.parameters(), lr=LR_G)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=eval(MILE_STONE), gamma=0.1)

        gamma = 0.1
        warm_up_with_multistep_lr = lambda epoch: epoch / int(WARMUP_EPOCH) if epoch <= int(
            WARMUP_EPOCH) else gamma ** len([m for m in eval(MILE_STONE) if m <= epoch])
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

        #### Loss Function ####
        # criterion = torch.nn.MSELoss()
        criterion = torch.nn.L1Loss()
        class_criterion = nn.CrossEntropyLoss()

        #if Keep_Train:
            #path = './model/{}/MinMax/Long/Imp_{}/pre_model_{}.pth'.format(DATA_TYPE, PRESUME_RECORD_ID,
                                                                           #PRESUME_EPOCH_S)
            #net.load_state_dict(torch.load(path))
            ## pretrained_dict = torch.load(path)
            ## net_dict = net.state_dict()
            ## pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}
            ## net_dict.update(pretrained_dict)
            ## net.load_state_dict(net_dict)
        #else:
            #pass

        if Keep_Train:
            path = './model/All/MinMax/Short/Imp_{}/pre_model_ep_{}.pth'.format(PRESUME_RECORD_ID, PRESUME_EPOCH_S)
            #path = './model/{}/MinMax/Short/Imp_{}/pre_model_{}.pth'.format(DATA_TYPE, PRESUME_RECORD_ID, PRESUME_EPOCH_S)
            #net.load_state_dict(torch.load(path))

            pretrained_dict = torch.load(path)

            net_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict}


            ignore_weights = []
            #ignore_weights.append('linear_tim_aux.0.weight')
            #ignore_weights.append('linear_tim_aux.0.bias')
            #ignore_weights.append('linear_typ_aux.0.weight')
            #ignore_weights.append('linear_typ_aux.0.bias')

            for key in ignore_weights:
                if pretrained_dict.pop(key, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(key))
                else:
                    print('Can Not Remove Weights: {}.'.format(key))




            net_dict.update(pretrained_dict)
            net.load_state_dict(net_dict)
            print('LOAD',path)
        else:
            pass

        #### 训练设备准备
        net = net.to(device)
        net = nn.DataParallel(net, device_ids=device_ids)

        #### Training ####
        it = 0
        for epoch in range(0, EPOCH_E):

            epoch_start = time.time()
            net.train()
            epoch_rmse_list = []
            for i, data in enumerate(train_loader):

                con, ave, ave_q, label, tim_cls, typ_cls = data

                B, T, C, H, W = con.shape
                ave = ave.to(device)
                ave_q = ave_q.to(device)
                con = con.to(device)
                label = label.to(device)
                tim_cls = tim_cls.squeeze().to(device)
                typ_cls = typ_cls.squeeze().to(device)

                optimizer.zero_grad()

                out, tim_out, typ_out = net(ave, ave_q, con)

                out = out.reshape(B, T, C, H, W)

                #### 将模型输出进行均值处理 ####
                if not IS_SEQ:
                    oup = out[:, 0].to(device)
                else:
                    oup = out
                # oup = output[:, 0].to(device)
                loss_gen = criterion(oup, label)
                loss_tim = class_criterion(tim_out, tim_cls.long())
                loss_typ = class_criterion(typ_out, typ_cls.long())

                loss = LOSS_GEN * loss_gen + LOSS_TIME * loss_tim + LOSS_TYP * loss_typ

                loss.backward()
                optimizer.step()

                net.eval()
                out, tim_out, typ_out = net(ave, ave_q, con)

                _, out_tim = torch.max(torch.softmax(tim_out, 1), 1)
                out_tim = out_tim.cpu().numpy()
                cls_tim = tim_cls.long().cpu().numpy()
                tim_score = round(metrics.accuracy_score(out_tim, cls_tim) * 100, 2)

                _, out_typ = torch.max(torch.softmax(typ_out, 1), 1)
                out_typ = out_typ.cpu().numpy()
                cls_typ = typ_cls.long().cpu().numpy()
                typ_score = round(metrics.accuracy_score(out_typ, cls_typ) * 100, 2)

                net.train()
                rmse = VALRMSE(oup, label, ds_factory.ds, ds_factory.dataset.m_factor)
                epoch_rmse_list.append(rmse.item())

                #if it % 20 == 0:
                if i % 100 == 0:
                    c_lr = scheduler.get_last_lr()
                    loss_info = 'TT:{:.6f},G: {:.6f},C:{:.6f},T:{:.6f}'.format(loss.item(), loss_gen.item(),loss_tim.item(), loss_typ.item())
                    _it = '{0:03d}'.format(i)
                    info = '-- It:{}/{},<bat_RMSE>:{:.2f},{},Tim:{},Typ:{},lr:{}'.format(_it,len(train_loader),rmse, loss_info, tim_score, typ_score, c_lr)
                    print(info)
                    record.write(info + '\n')

                    info_matrix = "[epoch %d][%d/%d] mse: %.4f rmse: %.4f RECORD:%s" % (
                        epoch, i + 1, len(train_loader), loss_gen.item(), rmse.item(),RECORD_ID)
                    record.write(info_matrix + '\n')
                    #print(info_matrix)
                    writer_val.add_scalar('mse', loss_gen.item(), it)
                    writer_val.add_scalar('rmse', rmse.item(), it)

                    writer_gen.add_scalar('Union', loss.item() * 10, it)
                    writer_gen.add_scalar('Generator', loss_gen.item() * 10, it)
                    writer_gen.add_scalar('Tim Classifier', loss_tim.item() * 10, it)
                    writer_gen.add_scalar('TYP Classifier', loss_typ.item() * 10, it)
                    writer_gen.add_scalar('lr', optimizer.param_groups[0]['lr'], it)

                if it % ITERATION_STEP == 0:
                    dirs = './model/All/MinMax/Short/Imp_{}'.format(RECORD_ID)
                    if not os.path.exists(dirs):os.makedirs(dirs)
                    model_path = os.path.join(dirs, f'pre_model_it_{it}.pth')
                    if TORCH_VERSION == '1.6.0' or TORCH_VERSION == '1.7.0':
                        torch.save(net.cpu().module.state_dict(), model_path, _use_new_zipfile_serialization=False)
                    else:
                        torch.save(net.cpu().module.state_dict(), model_path)
                    net = net.to(device)

                it += 1

            mean_rmse = np.mean(np.array(epoch_rmse_list))
            t = timeSince(start)
            epoch_t = timeSince(epoch_start)
            loss_info = 'D:{:.6f}'.format(loss.item())
            info = 'EPOCH:{}/{},Mean_RMSE {},Loss {} Time {},Epoch_Time {}'.format(epoch+1, EPOCH_E, mean_rmse,loss_info, t,epoch_t)
            print(info)
            record.write(info + '\n')
            scheduler.step()

            if (epoch + 1) % 1 == 0:

                dirs = './model/{}/MinMax/Short/Imp_{}'.format(DATA_TYPE, RECORD_ID)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                model_path = os.path.join(dirs, f'pre_model_ep_{epoch + 1}.pth')

                if TORCH_VERSION == '1.6.0' or TORCH_VERSION == '1.7.0':
                    torch.save(net.cpu().module.state_dict(), model_path, _use_new_zipfile_serialization=False)
                else:
                    torch.save(net.cpu().module.state_dict(), model_path)

                net = net.to(device)

        record.close()

    if IS_VAL:



        #train_len = 13728
        train_len = 13668
        epoch_iteartion = train_len//BATCH_SIZE
        total_iteation = EPOCH_E * epoch_iteartion
        iteration_step = ITERATION_STEP
        iteration_test_num = total_iteation // iteration_step

        if EVAL_MODE == 'Iteration':
            EPOCH_E = iteration_test_num
            EVAL_START_EPOCH = (EVAL_START_EPOCH * epoch_iteartion) // iteration_step
            NAME = "Iter"
            MUL = iteration_step
        else:
            NAME = "Epoch"
            MUL = 1



        ### TEST DATASET ###
        test_ds = ds_factory.get_test_dataset()

        test_loader = DataLoader(
            dataset=test_ds,
            batch_size=EVAL_BATCH,
            shuffle=False,
            num_workers=1
        )

        #### MODEL ####
        input_channels = C

        P_list = eval(PATCH_LIST)
        Is_scaling = IS_SCALE

        from net.imp_pos_cl_heat2heat import Prediction_Model as Model
        #from net.niu_imp_pos_cl_heat2heat import Prediction_Model as Model

        print('EVALUATION START')
        print('-' * 30)

        record = open("record/{}/log_eval.txt".format(RECORD_ID), "w")  ###xie

        if 1:

            rmse_list = []  ###xie
            mae_list = []  ###xie
            for epoch in range(0, EPOCH_E):

                net = Model(
                    mcof,
                    Length=LENGTH,
                    Width=W,
                    Height=H,
                    Input_dim=input_channels,
                    Patch_list=P_list,
                    Dropout=DROPOUT,
                    Att_num=ATT_NUM,
                    Cross_att_num=CROSS_ATT_NUM,
                    Using_skip=IS_USING_SKIP,
                    Encoding_dim=MODEL_DIM,
                    Embedding_dim=MODEL_DIM,
                    Is_mask=IS_MASK_ATT,
                    #Is_reduce=IS_REDUCE,
                    Is_scaling=Is_scaling,
                    Debugging=0,
                    Merge=MERGE,
                    Out_style=OUT_STYLE,
                    Cat_style=CAT_STYLE,
                    Is_aux=IS_AUX,
                    IS_C3D=IS_C3D,
                    ONLY_CONV6=ONLY_CONV6,
                )

                if IS_BEST_EVAL:
                    model_path = 'model/best_model/pre_model_best_{}.pth'.format('B')
                else:
                    if EVAL_MODE == 'Iteration':
                        model_path = './model/All/MinMax/Short/Imp_{}/pre_model_it_{}.pth'.format(RECORD_ID,(epoch + 1)*ITERATION_STEP)
                    else:
                        model_path = './model/All/MinMax/Short/Imp_{}/pre_model_ep_{}.pth'.format(RECORD_ID,epoch + 1)
                        #model_path = './model/All/MinMax/Short/Imp_{}/pre_model_{}.pth'.format(RECORD_ID,epoch + 1)
                    print(model_path)

                #try:
                    net.load_state_dict(torch.load(model_path))
                #except:
                    #print('error!')
                    #net = torch.load(model_path)

                device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
                net = net.to(device)
                net = nn.DataParallel(net)

                criterion = nn.MSELoss().to(device)

                net.eval()
                mse = 0.0
                mse_in = 0.0
                mse_out = 0.0
                mae = 0.0
                target = []
                pred = []
                mmn = ds_factory.ds.mmn
                with torch.no_grad():
                    for i, data in enumerate(test_loader, 0):

                        # (1,28,6,2,32,32) (1,28,6,2,32,32) (1,28,2,32,32)or(1,28,6,2,32,32) (1,28), (1,28)
                        con, ave, ave_q, label, tim_cls, typ_cls = data

                        if IS_SEQ:
                            tar = label[:, 0]
                        else:
                            tar = label  ##niu

                        ave = ave.to(device)
                        ave_q = ave_q.to(device)
                        tar = tar.to(device)
                        con = con.to(device)

                        gen_out, tim_out, typ_out = net(ave, ave_q, con)
                        # (2,32,32)
                        oup = gen_out[:, 0]

                        loss = criterion(oup, tar)  # 所有样本损失的平均值

                        if IS_REMOVE and i in range(772, 796):
                            pass
                        else:
                            mse += con.shape[0] * loss.item()  # 所有样本损失的总和
                            mae += con.shape[0] * torch.mean(
                                torch.abs(oup - tar)).item()  # mean()不加维度时，返回所有值的平均

                            ##niu
                            mse_in += con.shape[0] * torch.mean(
                                (tar[:, 0] - oup[:, 0]) * (tar[:, 0] - oup[:, 0])).item()
                            mse_out += con.shape[0] * torch.mean(
                                (tar[:, 1] - oup[:, 1]) * (tar[:, 1] - oup[:, 1])).item()

                        _, out_cls = torch.max(torch.softmax(tim_out, 1), 1)
                        out_class = out_cls.cpu().numpy()
                        lab_class = tim_cls.long().cpu().numpy()

                        target.append(lab_class)
                        pred.append(out_class)

                ## Validation
                target = np.concatenate(target)
                pred = np.concatenate(pred)
                tim_acc = metrics.accuracy_score(pred, target) * 100

                if IS_REMOVE:
                    cnt = ds_factory.ds.X_con_tes.shape[0] - 24
                else:
                    cnt = ds_factory.ds.X_con_tes.shape[0]

                mae /= cnt
                mae = mae * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor

                mse /= cnt
                rmse = math.sqrt(mse) * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor
                #print("mae: %.4f" % (mae))
                #print("rmse: %.4f" % (rmse))

                rmse_list.append(rmse)  ##xie
                mae_list.append(mae)  ##xie

                mse_in /= cnt
                rmse_in = math.sqrt(mse_in) * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor
                mse_out /= cnt
                rmse_out = math.sqrt(mse_out) * (mmn.max - mmn.min) / 2. * ds_factory.dataset.m_factor

                #info = "inflow rmse: %.5f    outflow rmse: %.4f" % (rmse_in, rmse_out)  ###xie
                info = "EPOCH:<%d> MAE: %.4f RMSE: [%.4f] IN: %.4f OUT: %.4f TIM: %.4f" % (epoch+1,mae,rmse,rmse_in, rmse_out,tim_acc)###xie
                #record.write(info + '\n')  ###xie

                min_idx = rmse_list.index(min(rmse_list))  ###xie
                rmse_min = round(rmse_list[min_idx], 2)  ###xie
                mae_min = round(mae_list[min_idx], 2)  ###xie
                info = '{} Best:RMSE:{},MAE:{},epoch:{}'.format(info,rmse_min, mae_min, min_idx + 1)  ###xie
                print('---------------------------------')  ###xie
                print(info)  ###xie
                record.write('-----------------------' + '\n')  ###xie
                record.write(info + '\n')  ###xie

            record.close()  ###xie


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass in some training parameters')
    parser.add_argument('--mode', type=str, default='train', help='The processing phase of the model')
    parser.add_argument('--record', type=str, help='Recode ID')
    parser.add_argument('--presume_record', type=str, help='Presume Recode ID')
    parser.add_argument('--task', type=str, default='B', help='Processing task type')
    parser.add_argument('--keep_train', type=int, default=0, help='Model keep training')
    parser.add_argument('--epoch_s', type=int, default=0, help='Continue training on the previous model')
    parser.add_argument('--presume_epoch_s', type=int, default=0, help='Continue training on the previous model')
    parser.add_argument('--inp_type', type=str, default='external',
                        choices=['external', 'train', 'accumulate', 'accumulate_avg', 'holiday', 'windspeed', 'weather',
                                 'temperature'])
    parser.add_argument('--patch_method', type=str, default='STTN', choices=['EINOPS', 'UNFOLD', 'STTN'])
    parser.add_argument('--dataset_type', type=str, default='Sub', choices=['Sub', 'All'],
                        help='datasets type is sub_datasets or all_datasets')
    parser.add_argument('--context_type', type=str, default='cpt', choices=['cpt', 'cpte'],
                        help='components of contextual data')
    parser.add_argument('--is_remove', default=0, help='whether to remove the problematic label')

    parser.add_argument('--ext_inp_type', type=str, default='external', choices=['external'])
    parser.add_argument('--debug', type=int, default=0, help='Model debug')
    parser.add_argument('--pretrained_class_model_path', type=str, default=None,
                        help='freeze encoder param,using pretrain param')
    parser.add_argument('--finetune_class_encoder', dest='finetune_class_encoder',
                        action='store_true', default=False)
    parser.add_argument('--pos_en', type=int, default=1, help='positional encoding')
    parser.add_argument('--pos_en_mode', type=str, default='cat', help='positional encoding mode')
    mcof = parser.parse_args()

    run(mcof)
