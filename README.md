# traffic

## Data
**Allset:(21360,2,32,32)**
- ./data/Allset/MinMax/train_inp_average.npy   训练集的平均填充

**Subset:(6624,2,32,32)**
- ./data/Subset/MinMax/train_inp_average.npy   训练集的平均填充

## 短期流量预测任务--有context和avg两个输入
- 主文件为pre_main.py
- 配置文件为pre_setting_bj.yml
- 数据处理文件 dateset.py 和 data_fetcher.py
- 数据处理文件 dateset_cpte.py 和 data_fetcher_cpte.py

### 模型改进
- ./net/imp_heat2heat.py
- ./net/imp_heat.py

### 训练
    python pre_main.py
           --mode train
           --record 306
           --dataset_type Sub
           --patch_method STTN
           --stream_num 1
           --context_type cpt
           --pos_en 1


### 测试
    python pre_main.py
           --mode val
           --record 306
           --dataset_type Sub
           --patch_method STTN
           --stream_num 1
           --context_type cpt
           --pos_en 1

### 原模型
- 主文件为pre_main_short.py
- 配置文件为pre_setting_bj.yml
- 数据处理文件 dateset.py 和 data_fetcher.py

- ./net/imp_pos_cl_heat2heat.py

### 训练
    python pre_main_short.py
           --mode train
           --record 306
           --dataset_type Sub
           --patch_method STTN
           --pos_en 1

### 继续训练
    python pre_main_short.py
           --mode train 
           --record 408 
           --keep_train 1 
           --presume_record 401 
           --presume_epoch_s 30


### 测试
    python pre_main_short.py
           --mode val
           --record 306
           --dataset_type Sub
           --patch_method STTN
           --pos_en 1



## 长期流量预测任务--有context和avg两个输入
- 主文件为pre_main_long.py
- 配置文件为pre_setting_bj_l.yml
- 数据处理文件 dateset_l.py 和 data_fetcher_l.py

### 模型同短期

### 训练测试也同短期
