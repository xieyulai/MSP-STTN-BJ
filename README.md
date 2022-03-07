# MSP-STTN

Code and data for the paper [Multi-Size Patched Spatial-Temporal Transformer Network for Short- and Long-Term Grid-based Crowd Flow Prediction]()

Please cite the following paper if you use this repository in your research.
```
Under construction
```

This repo is for **TaxiBJ**, more information can be found in [MSP-STTN](https://github.com/xieyulai/MSP-STTN). 

## TaxiBJ

### Package
```
PyTorch > 1.07
```
Please refer to `requirements.txt`

### Data Preparation
- Processing data according to [MSP-STTN-DATA](https://github.com/xieyulai/MSP-STTN-DATA).
- The `data\` should be like this:
```bash
data
___ TaxiBJ
```
- Or the processed data can be downloaded from [BAIDU_PAN](https://pan.baidu.com/s/1aXkP1NgGPCPSjSkus8rpdw),PW:`p3r0`.


### Pre-trained Models
- Several pre-trained models can be downloaded from [BAIDU_PAN](https://pan.baidu.com/s/1HRzBa6L-HQtuSxNTUkA2vQ), PW:`9ius`.
- The `model\` should be like this:
```bash
model
___ Imp_0547
___ ___ pre_model_ep_19.pth
___ Imp_0548
___ ___ pre_model_ep_41.pth
___ Imp_1543
___ ___ pre_model_ep_0.pth
___ ___ pre_model_it_14700.pth
___ Imp_1545
___ ___ pre_model_ep_23.pth
___ Imp_3548
___ ___ pre_model_ep_22.pth
___ Imp_3805
___ ___ pre_model_ep_22.pth
___ Imp_5547
    ___ pre_model_ep_27.pth
```
- Use `sh BEST.sh` for short-term prediction.
- Use `sh BEST_long.sh` for short-term prediction.

### Train and Test
- Use `sh TRAIN.sh` for short-term prediction.
- Use `sh TRAIN_long.sh` for short-term prediction.

### Repo Structure
```bash
___ BEST_long.sh
___ BEST.sh
___ data # Data
___ dataset
___ model # Store the training weights
___ net # Network struture
___ pre_main_short.py # Main function for shot-term prediction
___ pre_setting_bj_long.yaml # Configuration for long-term prediction
___ pre_setting_bj.yaml # Configuration for short-term prediction
___ README.md
___ record # Recording the training and the test
___ TRAIN_long.sh
___ TRAIN.sh
___ util
```
