import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data
import pdb

# from dataset.external import external_taxibj, external_bikenyc, external_taxinyc
from dataset.minmax_normalization import MinMaxNormalization
from dataset.data_fetcher_cpte import DataFetcher


class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    print('*' * 10 + 'DEBUG' + '*' * 10)
    print(datapath)

    def __init__(self, dconf, inp_type, DATA_TYPE, test_days=-1, datapath=datapath):
        self.dconf = dconf
        self.inp_type = inp_type
        self.dataset = dconf.name
        self.datapath = datapath
        self.data_type = DATA_TYPE
        if self.dataset == 'TaxiBJ':
            self.datafolder = 'TaxiBJ/dataset'

            if self.data_type == 'Sub':
                self.dataname = [
                    'BJ16_M32x32_T30_InOut.h5'
                ]
                self.extname = [
                    'BJ16_SUB_EXT.h5'
                ]
            else:
                self.dataname = [
                    'BJ13_M32x32_T30_InOut.h5',
                    'BJ14_M32x32_T30_InOut.h5',
                    'BJ15_M32x32_T30_InOut.h5',
                    'BJ16_M32x32_T30_InOut.h5'
                ]
                self.extname = [
                    'BJ13_SUB_EXT.h5',
                    'BJ14_SUB_EXT.h5',
                    'BJ15_SUB_EXT.h5',
                    'BJ16_SUB_EXT.h5'
                ]
            self.nb_flow = 2
            self.dim_h = 32
            self.dim_w = 32
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.m_factor = 1.
            ext_dim = 28
            if dconf.ext_time_flag:
                ext_dim += 25
                if dconf.fourty_eight:
                    ext_dim += 24
            self.ext_dim = ext_dim

        elif self.dataset == 'BikeNYC':
            self.datafolder = 'BikeNYC'
            self.dataname = ['NYC14_M16x8_T60_NewEnd.h5']
            self.nb_flow = 2
            self.dim_h = 16
            self.dim_w = 8
            self.T = 24
            test_days = 10 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 16 * 8 / 81)
            self.ext_dim = 33 if dconf.ext_time_flag else 8

        elif self.dataset == 'TaxiNYC':
            self.datafolder = 'TaxiNYC'
            self.dataname = ['NYC2014.h5']
            self.nb_flow = 2
            self.dim_h = 15
            self.dim_w = 5
            self.T = 48
            test_days = 28 if test_days == -1 else test_days

            self.m_factor = math.sqrt(1. * 15 * 5 / 64)
            self.ext_dim = 77

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
        return vec[:int(math.floor((len(vec) - self.len_test) * self.portion))]

    def testset_of(self, vec):
        return vec[int(-math.floor(self.len_test * self.portion)):]

    def split(self, context, y, x_all_ave, y_class):
        x_tra = self.trainset_of(context)
        x_tes = self.testset_of(context)

        x_all_ave_tra = self.trainset_of(x_all_ave)
        x_all_ave_tes = self.testset_of(x_all_ave)

        y_tra = self.trainset_of(y)
        y_tes = self.testset_of(y)

        y_tra_cls = self.trainset_of(y_class)
        y_tes_cls = self.testset_of(y_class)

        return x_tra, x_all_ave_tra, y_tra, y_tra_cls, x_tes, x_all_ave_tes, y_tes, y_tes_cls

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

        # EXT
        new_ext_list, cls_list = [], []
        for filename in self.extname:
            f = h5py.File(os.path.join(self.datapath, self.datafolder, filename), 'r')
            _new_ext = f['data'][()]
            _new_cls = f['cls'][()]
            f.close()

            new_ext_list.append(_new_ext)
            cls_list.append(_new_cls)

        # 1、归一化数据如何求方差和均值，在整个数据集上还是训练集上
        # 2、求平均是在整个数据集上还是训练集上
        # (21360, 6, 2, 32, 32)   21360/48 = 445

        inp_path = f'./data/TaxiBJ/{self.data_type}set/MinMax/{self.inp_type}_inp_average.npy'
        all_average_data = np.load(inp_path, allow_pickle=True)
        new_average_data_list = list(all_average_data)

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

        x_list, y_list, x_all_ave_list, x_ext_ave_list, ts_x_list, ts_y_list = [], [], [], [], [], []
        for idx in range(len(ts_new_list)):
            x, x_all_ave, x_ext_ave, y, ts_x, ts_y = \
                DataFetcher(new_data_list[idx], ts_new_list[idx], new_average_data_list[idx], new_ext_list[idx],
                            self.T).fetch_data(self.dconf)
            x_list.append(x)
            y_list.append(y)
            x_all_ave_list.append(x_all_ave)
            x_ext_ave_list.append(x_ext_ave)
            ts_x_list.append(ts_x)  # list nest list nest list nest numpy.datetime64 class
            ts_y_list.append(ts_y)  # list nest list nest numpy.datetime64 class
        x = np.concatenate(x_list)
        y = np.concatenate(y_list)
        x_all_ave = np.concatenate(x_all_ave_list)
        x_ext_ave = np.concatenate(x_ext_ave_list)
        x_ext_ave = x_ext_ave.reshape((x.shape[0], 1, 2, 32, 32))
        context = np.concatenate([x, x_ext_ave], axis=1)
        ts_y = np.concatenate(ts_y_list)

        Y_Class = []
        for i in enumerate(ts_y[::48]):
            Y_Class.append(np.array(range(0, 48)))
        y_all_class = np.concatenate(Y_Class, axis=0).reshape(-1, 1)

        # (16464, 12, 32, 32) (16464, 2, 32, 32) (16464, 6) (16464,)
        x_tra, x_all_ave_tra, y_tra, y_all_cls_tra, x_tes, x_all_ave_tes, y_tes, y_all_cls_tes = self.split(
            context, y, x_all_ave, y_all_class)

        class TempClass:
            def __init__(self_2):
                self_2.X_tra = x_tra
                self_2.X_all_ave_tra = x_all_ave_tra
                self_2.Y_tra = y_tra
                self_2.Y_all_cls_tra = y_all_cls_tra

                self_2.X_tes = x_tes
                self_2.X_all_ave_tes = x_all_ave_tes
                self_2.Y_tes = y_tes
                self_2.Y_all_cls_tes = y_all_cls_tes

                self_2.img_mean = np.mean(train_dat, axis=0)
                self_2.img_std = np.std(train_dat, axis=0)
                self_2.mmn = mmn
                self_2.ts_Y_train = self.trainset_of(ts_y)
                self_2.ts_Y_test = self.testset_of(ts_y)

            def show(self_2):
                print(
                    "Run: X inputs shape: ", self_2.X_tra.shape, self_2.X_all_ave_tra.shape,
                    self_2.X_tes.shape, self_2.X_all_ave_tes.shape,
                    "Y inputs shape: ", self_2.Y_tra.shape, self_2.Y_all_cls_tra.shape,
                    self_2.Y_tes.shape, self_2.Y_all_cls_tes.shape,
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
            X_ALL_AVE = torch.from_numpy(self.ds.X_all_ave_tra[index])
            Y = torch.from_numpy(self.ds.Y_tra[index])
            Y_all_cls = torch.Tensor(self.ds.Y_all_cls_tra[index])
        else:
            X = torch.from_numpy(self.ds.X_tes[index])
            X_ALL_AVE = torch.from_numpy(self.ds.X_all_ave_tes[index])
            Y = torch.from_numpy(self.ds.Y_tes[index])
            Y_all_cls = torch.Tensor(self.ds.Y_all_cls_tes[index])

        return X.float(), X_ALL_AVE.float(), Y.float(), Y_all_cls.float()

    def __len__(self):
        if self.mode == 'train':
            return self.ds.X_tra.shape[0]
        else:
            return self.ds.X_tes.shape[0]


class DatasetFactory(object):
    def __init__(self, dconf, inp_type, data_type):
        self.dataset = Dataset(dconf, inp_type, data_type)
        self.ds = self.dataset.load_data()
        print('Show a list of dataset!')
        print(self.ds.show())

    def get_train_dataset(self):
        return TorchDataset(self.ds, 'train')

    def get_test_dataset(self):
        return TorchDataset(self.ds, 'test')


if __name__ == '__main__':
    class DataConfiguration:
        # Data
        name = 'TaxiBJ'
        portion = 1.  # portion of data

        len_close = 6
        len_period = 3
        len_trend = 2
        pad_forward_period = 0
        pad_back_period = 0
        pad_forward_trend = 0
        pad_back_trend = 0

        len_all_close = len_close * 1
        len_all_period = len_period * (1 + pad_back_period + pad_forward_period)
        len_all_trend = len_trend * (1 + pad_back_trend + pad_forward_trend)

        len_seq = len_all_close + len_all_period + len_all_trend
        cpt = [len_all_close, len_all_period, len_all_trend]

        interval_period = 1
        interval_trend = 7

        ext_flag = True
        ext_time_flag = True
        rm_incomplete_flag = True
        fourty_eight = True
        previous_meteorol = True


    df = DatasetFactory(DataConfiguration(), inp_type='train', data_type='Sub')
    ds = df.get_train_dataset()
    X, X_all_ave, Y, Y_cls = next(iter(ds))
    print('train:')
    print(X.size())
    print(X_all_ave.size())
    print(Y.size())
    print(Y_cls)

    # ds = df.get_train_dataset()
    # X, X_ave, X_ext, Y, Y_ext = next(iter(ds))
    # print('test:')
    # print(X.size())
    # print(X_ave.size())
    # print(X_ext.size())
    # print(Y.size())
    # print(Y_ext.size())
