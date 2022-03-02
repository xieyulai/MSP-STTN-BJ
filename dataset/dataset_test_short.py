import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data
import pdb
import time
from einops import rearrange
import matplotlib.pyplot as plt

# from dataset.external import external_taxibj, external_bikenyc, external_taxinyc
from minmax_normalization import MinMaxNormalization
from data_fetcher import DataFetcher

import sys
sys.path.append('./data')


class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    print('*' * 10 + 'DEBUG' + '*' * 10)
    print(datapath)

    def __init__(self, dconf, Inp_type, Data_type, Length, Is_seq, Is_correct, test_days=-1, datapath=datapath):
        self.dconf = dconf
        self.dataset = dconf.name
        self.len_close = dconf.len_close
        self.len_period = dconf.len_period
        self.len_trend = dconf.len_trend
        self.datapath = datapath
        self.inp_type = Inp_type
        self.data_type = Data_type
        self.length = Length
        self.is_seq = Is_seq
        self.is_revise = Is_correct

        if self.dataset == 'TaxiBJ':
            self.datafolder = 'TaxiBJ/dataset'
            if self.data_type == 'Sub':
                self.dataname = [
                    'BJ16_M32x32_T30_InOut.h5'
                    # 'BJ13_M32x32_T30_InOut.h5',
                ]
            else:
                self.dataname = [
                    'BJ13_M32x32_T30_InOut.h5',
                    'BJ14_M32x32_T30_InOut.h5',
                    'BJ15_M32x32_T30_InOut.h5',
                    'BJ16_M32x32_T30_InOut.h5'
                ]
            self.nb_flow = 2
            self.dim_h = 32
            self.dim_w = 32
            self.T = 48
            if self.len_close == 4 and self.len_period == 2 and self.len_trend == 2:
                test_days = 24 if test_days == -1 else test_days
            else:
                test_days = 28 if test_days == -1 else test_days

            self.m_factor = 1.

        elif self.dataset == 'BikeNYC':
            self.datafolder = 'BikeNYC'
            self.dataname = ['NYC14_M16x8_T60_NewEnd.h5']
            self.nb_flow = 2
            self.dim_h = 16
            self.dim_w = 8
            self.T = 24
            test_days = 10 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 16 * 8 / 81)

        elif self.dataset == 'TaxiNYC':
            self.datafolder = 'TaxiNYC'
            self.dataname = ['NYC2014.h5']
            self.nb_flow = 2
            self.dim_h = 15
            self.dim_w = 5
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 15 * 5 / 64)

        else:
            raise ValueError('Invalid dataset')

        self.len_test = test_days * self.T
        self.portion = dconf.portion

    def get_raw_data(self):
        """
         data:
         np.array(n_sample * n_flow * height * width)
         ts:
         np.array(n_sample * length of timestamp string)
        """
        raw_data_list = list()
        raw_ts_list = list()
        print("  Dataset: ", self.datafolder)

        for filename in self.dataname:
            f = h5py.File(os.path.join(self.datapath, self.datafolder, filename), 'r')
            _raw_data = f['data'][()]
            _raw_ts = f['date'][()]
            f.close()

            raw_data_list.append(_raw_data)
            raw_ts_list.append(_raw_ts)
        # delete data over 2channels

        return raw_data_list, raw_ts_list

    @staticmethod
    def remove_incomplete_days(data, timestamps, t=48):
        print("before removing", len(data))
        # remove a certain day which has not 48 timestamps
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + t - 1 < len(timestamps) and int(timestamps[i + t - 1][8:]) == t:
                days.append(timestamps[i][:8])
                i += t
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        print("after removing", len(data))
        return data, timestamps

    def trainset_of(self, vec):
        return vec[:math.floor((len(vec) - self.len_test) * self.portion)]

    def testset_of(self, vec):
        return vec[-math.floor(self.len_test * self.portion):]

    def split(self, x, y, x_ave, y_cls, y_typ):
        x_tra = self.trainset_of(x)
        x_tes = self.testset_of(x)

        x_ave_tra = self.trainset_of(x_ave)
        x_ave_tes = self.testset_of(x_ave)

        y_tra = self.trainset_of(y)
        y_tes = self.testset_of(y)

        y_tra_cls = self.trainset_of(y_cls)
        y_tes_cls = self.testset_of(y_cls)

        y_tra_typ = self.trainset_of(y_typ)
        y_tes_typ = self.testset_of(y_typ)

        return x_tra, x_ave_tra, y_tra, y_tra_cls, y_tra_typ, x_tes, x_ave_tes, y_tes, y_tes_cls, y_tes_typ

    def load_data(self):
        """
        return value:
            X_train & X_test: [XC, XP, XT, Xext]
            Y_train & Y_test: vector
        """
        # read file and place all of the raw data in np.array. 'ts' means timestamp
        # without removing incomplete days
        print('Preprocessing: Reading HDF5 file(s)')
        raw_data_list, ts_list = self.get_raw_data()

        # filter dataset
        data_list, ts_new_list = [], []
        for idx in range(len(ts_list)):
            raw_data = raw_data_list[idx]
            ts = ts_list[idx]

            if self.dconf.rm_incomplete_flag:
                raw_data, ts = self.remove_incomplete_days(raw_data, ts, self.T)

            data_list.append(raw_data)  # 列表套列表套数组，最外层长度为4
            ts_new_list.append(ts)

        print(np.array(data_list[0]).shape)

        # np.save('../data/TaxiBJ/Subset/Plot/plot_all_date_13.npy', np.array(ts_new_list[0]))
        # np.save('../data/TaxiBJ/Subset/Plot/plot_all_data_13.npy', np.array(data_list[0]))
        #
        # pdb.set_trace()

        print(f'=============={self.inp_type} 输入加载成功！=============')
        inp_path = f'../data/TaxiBJ/{self.data_type}set/AVG{self.length}/{self.inp_type}_inp_average.npy'
        all_average_data = np.load(inp_path, allow_pickle=True)
        new_average_data_list = list(all_average_data)

        ext_cls_path = f'../data/TaxiBJ/{self.data_type}set/AVG{self.length}/{self.inp_type}_cls.npy'
        all_ext_cls = np.load(ext_cls_path, allow_pickle=True)
        new_all_ext_cls = list(all_ext_cls)

        print('Preprocessing: Min max normalizing')
        raw_data = np.concatenate(data_list)
        mmn = MinMaxNormalization()
        # # (21360, 2, 32, 32), len(ts_new_list)=4
        train_dat = self.trainset_of(raw_data)
        mmn.fit(train_dat)
        new_data_list = [
            mmn.transform(data).astype('float32', copy=False)
            for data in data_list
        ]
        print('Context data min max normalizing processing finished!')

        x_list, y_list, x_ave_list, y_typ_list, ts_x_list, ts_y_list = [], [], [], [], [], []
        for idx in range(len(ts_new_list)):
            # x, x_ave, x_ave_q, y, y_cls, ts_x, ts_y
            x, x_ave, x_ave_q, y, y_cls, ts_x, ts_y = \
                DataFetcher(new_data_list[idx], ts_new_list[idx], new_average_data_list[idx], new_all_ext_cls[idx], self.T).fetch_data(self.dconf)
            x_list.append(x)
            y_list.append(y)

            x_ave_list.append(x_ave)
            y_typ_list.append(y_cls)

            ts_x_list.append(ts_x)  # list nest list nest list nest numpy.datetime64 class
            ts_y_list.append(ts_y)  # list nest list nest numpy.datetime64 class
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        x_ave = np.concatenate(x_ave_list)
        y_typ = np.concatenate(y_typ_list)
        ts_y = np.concatenate(ts_y_list)

        all_ts = []
        for ts in ts_y:
            wday = time.strptime(str(ts)[:10], '%Y-%m-%d')
            ts_arr = np.array([wday[0], wday[1], wday[2]])
            all_ts.append(ts_arr)
        all_ts_array = np.stack(all_ts)
        ts_tra = self.trainset_of(all_ts_array)
        ts_tes = self.testset_of(all_ts_array)

        print(ts_y[-self.len_test:])
        # print(ts_y[-self.len_test:][768:768+24])
        #
        #### 画出有问题的天及前一天的heatmap
        # for i in range(48):
        #     lab = y[-self.len_test:][768+i]
        #     lab1 = y[-self.len_test:][768-48+i]
        #
        #     fig = plt.gcf()
        #     fig.set_size_inches(32/10.0, 32/10.0)  #输出width*height像素
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace =0, wspace =0)#输出图像#边框设置
        #     plt.margins(0,0)
        #
        #     lab_inp = lab[0:1]
        #     lab_inp = rearrange(lab_inp, '1 h w  -> h (w)')
        #     # plt.imshow(lab_inp)
        #     # plt.savefig(f'./image/inp_{i + 1}.jpg')
        #
        #     # lab_out = lab[1:]
        #     # lab_out = rearrange(lab_out, '1 h w  -> h (w)')
        #     # plt.imshow(lab_out)
        #     # plt.savefig(f'./image/out_{i + 1}.jpg')
        #
        #     lab_out = lab1[0:1]
        #     lab_out = rearrange(lab_out, '1 h w  -> h (w)')
        #
        #     heatmap = np.concatenate([lab_inp,lab_out])
        #     plt.imshow(heatmap)
        #     plt.savefig(f'./image/heatmap_tes_{i + 1}.jpg')


        #### 画出有问题的天及前后天的每个时刻的平均值
        res_before = []
        res = []
        res_after = []
        dif = mmn.max - mmn.min
        for i in range(48):
            y_meta = (y[-self.len_test:]+ 1.) * dif / 2.
            lab_before = y_meta[768-48+i]
            lab = y_meta[768+i]
            lab_after = y_meta[768+48+i]
            res_before.append(np.mean(lab_before))
            res.append(np.mean(lab))
            res_after.append(np.mean(lab_after))

        plt.plot(res_before, label='2016-03-27')
        plt.plot(res, label='2016-03-28')
        plt.plot(res_after, label='2016-03-29')

        plt.legend()
        plt.xlabel("interval")
        plt.ylabel("mean")
        plt.show()


        Y_Class = []
        for i in enumerate(ts_y[::48]):
            Y_Class.append(np.array(range(0, 48)))
        y_cls = np.concatenate(Y_Class, axis=0).reshape(-1, 1)

        y_typ = y_typ.reshape(-1, 1)

        # (16464, 12, 32, 32) (16464, 2, 32, 32) (16464, 6) (16464,)
        x_tra, x_ave_tra, y_tra, y_tra_cls, y_tra_typ, x_tes, x_ave_tes, y_tes, y_tes_cls, y_tes_typ = self.split(
            x, y, x_ave, y_cls, y_typ)

        # 对异常tar进行修正
        if self.is_revise:
            for i, ts in enumerate(ts_tes[::48]):
                if int(ts[2]) == 21:
                    y_tes[i*48+22] = (y_tes[i*48+21] + y_tes[i*48+20]) / 2
                elif int(ts[2]) == 22:
                    y_tes[i*48] = (y_tes[i*48-1] + y_tes[i*48-2]) / 2
                elif int(ts[2]) == 28:
                    y_tes[i*48] = (y_tes[i*48-1] + y_tes[i*48-2]) / 2
                    y_tes[i*48+24] = (y_tes[i*48+23] + y_tes[i*48+22]) / 2
                elif int(ts[2]) == 31:
                    y_tes[i*48] = (y_tes[i*48-1] + y_tes[i*48-2]) / 2

        # 是否使用多个序列长度求loss
        if self.is_seq:
            x_tra = x_tra[:-self.length+1]
            x_tes = x_tes[:-self.length+1]
            x_ave_tra = x_ave_tra[:-self.length+1]
            x_ave_tes = x_ave_tes[:-self.length+1]
            y_tra_cls = y_tra_cls[:-self.length+1]
            y_tes_cls = y_tes_cls[:-self.length+1]
            y_tra_typ = y_tra_typ[:-self.length+1]
            y_tes_typ = y_tes_typ[:-self.length+1]

            y_seq_tra = []
            for i, _ in enumerate(y_tra[:-self.length+1]):
                y_seq_tra.append(y_tra[i:i+self.length])
            y_seq_tra = np.stack(y_seq_tra)

            y_seq_tes = []
            for i, _ in enumerate(y_tes[:-self.length+1]):
                y_seq_tes.append(y_tes[i:i+self.length])
            y_seq_tes = np.stack(y_seq_tes)

        class TempClass:
            def __init__(self_2):
                self_2.X_tra = x_tra
                self_2.X_ave_tra = x_ave_tra
                if self.is_seq:
                    self_2.Y_tra = y_seq_tra
                else:
                    self_2.Y_tra = y_tra
                self_2.Y_tra_cls = y_tra_cls
                self_2.Y_tra_typ = y_tra_typ

                self_2.X_tes = x_tes
                self_2.X_ave_tes = x_ave_tes
                if self.is_seq:
                    self_2.Y_tes = y_seq_tes
                else:
                    self_2.Y_tes = y_tes
                self_2.Y_tes_cls = y_tes_cls
                self_2.Y_tes_typ = y_tes_typ

                self_2.img_mean = np.mean(train_dat, axis=0)
                self_2.img_std = np.std(train_dat, axis=0)
                self_2.mmn = mmn
                self_2.ts_Y_train = ts_tra
                self_2.ts_Y_test = ts_tes

            def show(self_2):
                print(
                    "Run: X inputs shape: ", self_2.X_tra.shape, self_2.X_ave_tra.shape,
                    self_2.X_tes.shape, self_2.X_ave_tes.shape,
                    "Y inputs shape: ", self_2.Y_tra.shape, self_2.Y_tra_cls.shape, self_2.Y_tra_typ.shape,
                    self_2.Y_tes.shape, self_2.Y_tes_cls.shape, self_2.Y_tes_typ.shape
                )
                print("Run: min~max: ", self_2.mmn.min, '~', self_2.mmn.max)

        return TempClass()


class TorchDataset(data.Dataset):
    def __init__(self, ds, mode='train'):
        super(TorchDataset, self).__init__()
        self.ds = ds
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            X = torch.from_numpy(self.ds.X_tra[index])
            X_AVE = torch.from_numpy(self.ds.X_ave_tra[index])
            Y = torch.from_numpy(self.ds.Y_tra[index])
            Y_tim = torch.Tensor(self.ds.Y_tra_cls[index])
            Y_typ = torch.Tensor(self.ds.Y_tra_typ[index])
            Y_ts = torch.from_numpy(self.ds.ts_Y_train[index])
        else:
            X = torch.from_numpy(self.ds.X_tes[index])
            X_AVE = torch.from_numpy(self.ds.X_ave_tes[index])
            Y = torch.from_numpy(self.ds.Y_tes[index])
            Y_tim = torch.Tensor(self.ds.Y_tes_cls[index])
            Y_typ = torch.Tensor(self.ds.Y_tes_typ[index])
            Y_ts = torch.Tensor(self.ds.ts_Y_test[index])

        return X.float(), X_AVE.float(), Y.float(), Y_tim.float(), Y_typ.float(), Y_ts

    def __len__(self):
        if self.mode == 'train':
            return self.ds.X_tra.shape[0]
        else:
            return self.ds.X_tes.shape[0]


class DatasetFactory(object):
    def __init__(self, dconf, Inp_type, Data_type, Length, Is_seq, Is_correct):
        self.dataset = Dataset(dconf, Inp_type, Data_type, Length, Is_seq, Is_correct)
        self.ds = self.dataset.load_data()
        print('Show a list of dataset!')
        print(self.ds.show())

    def get_train_dataset(self):
        return TorchDataset(self.ds, 'train')

    def get_test_dataset(self):
        return TorchDataset(self.ds, 'test')


if __name__ == '__main__':
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

    df = DatasetFactory(DataConfiguration(4, 1, 1), Inp_type='external', Data_type='Sub', Length=6, Is_seq=0, Is_correct=1)
    ds = df.get_train_dataset()
    X, X_ave, Y, Y_cls, Y_ext_cls, Ts_y = next(iter(ds))
    print('train:')
    print(X.size())
    print(X_ave.size())
    print(Y.size())
    print(Y_cls)
    print(Y_ext_cls)
    print(Ts_y)

    # ds = df.get_train_dataset()
    # X, X_ave, X_ext, Y, Y_ext = next(iter(ds))
    # print('test:')
    # print(X.size())
    # print(X_ave.size())
    # print(X_ext.size())
    # print(Y.size())
    # print(Y_ext.size())