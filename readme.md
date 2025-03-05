[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/collaborative-discrepancy-optimization-for-1/anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad?p=collaborative-discrepancy-optimization-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/collaborative-discrepancy-optimization-for-1/anomaly-detection-on-mvtec-3d-ad-1)](https://paperswithcode.com/sota/anomaly-detection-on-mvtec-3d-ad-1?p=collaborative-discrepancy-optimization-for-1)

## Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization
[IEEE Transactions on Industrial Informatics 2023](https://ieeexplore.ieee.org/document/10034849)

# 🔥 News
- *2025.03*: 🎉🎉 Check out our updated version of CDO! You can find significant improvements versus our original version! The current version also supports both semi-supervised and unsupervised anomaly detection.
  
- *2023.03*: 🎉🎉 We published a new paper related to point cloud anomaly detection, [Complementary Pseudo Multimodal Feature for Point Cloud Anomaly Detection](https://arxiv.org/ftp/arxiv/papers/2303/2303.13194.pdf).  
  [🔗 Paper](https://arxiv.org/ftp/arxiv/papers/2303/2303.13194.pdf) | [🔗 Code](https://github.com/caoyunkang/CPMF)

---

## Abstract
Most unsupervised image anomaly localization methods suffer from overgeneralization due to the high generalization abilities of convolutional neural networks, leading to unreliable predictions. To mitigate this, we propose Collaborative Discrepancy Optimization (CDO), which optimizes normal and abnormal feature distributions with synthetic anomalies. CDO introduces a margin optimization module and an overlap optimization module to maximize the margin and minimize the overlap between the discrepancy distributions (DDs) of normal and abnormal samples. Experiments on MVTec2D and MVTec3D demonstrate that CDO effectively mitigates overgeneralization, achieving excellent anomaly localization performance with real-time computation efficiency.

## BibTex Citation
If you find our [paper](https://ieeexplore.ieee.org/document/10034849) or code useful, please cite us using the following BibTex:
```bibtex
@ARTICLE{10034849,
  author={Cao, Yunkang and Xu, Xiaohao and Liu, Zhaoge and Shen, Weiming},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Collaborative Discrepancy Optimization for Reliable Image Anomaly Localization}, 
  year={2023},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/TII.2023.3241579}}
```

---

## Quick Start
1. Create a new conda environment and install the required packages:
   ```bash
   conda create -n cdo_env python=3.9.12
   conda activate cdo_env
   pip install -r requirements.txt
   ```

---

## Data Preparation
We support the [MVTec AD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) and [VisA dataset](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar) for anomaly localization in factory settings. Unzip the files to the following directories:
```
datasets/
├── mvtec_anomaly_detection/
│   ├── bottle/
│   ├── cable/
│   ├── ...
│   └── meta.json
└── VisA_20220922/
    ├── candle/
    ├── capsules/
    ├── ...
    └── meta.json
```
To generate the required JSON files for training, run the following scripts:
```bash
python ./data/gen_metadata/mvtec.py
python ./data/gen_metadata/visa.py
```

---

## Training Models
### Execute Different Training Tasks
Modify the `self.train_data.mode` and `self.test_data.mode` parameters in `./configs/benchmark/cdo/cdo_256_100e.py` to select different training tasks. The available tasks and their corresponding metadata files are:
```python
EXPERIMENTAL_SETUP = {
    'zero_shot': 'meta_zero_shot.json',
    'few_shot1': 'meta_few_shot1.json',
    'few_shot2': 'meta_few_shot2.json',
    'few_shot4': 'meta_few_shot4.json',
    'few_shot8': 'meta_few_shot8.json',
    'unsupervised': 'meta_unsupervised.json',
    'semi1': 'meta_semi1.json',
    'semi5': 'meta_semi5.json',
    'semi10': 'meta_semi10.json',
}
```
For example, to run unsupervised training:
```python
self.train_data.mode = 'unsupervised'
self.test_data.mode = 'unsupervised'
```

### Run Experiment
```bash
python run_dataset.py
```

---

## CDO Architecture
![CDO Framework](./pngs/framework.png)

## Reference CDO Quantitative Results 
![Quantitative Results](./pngs/quantitative_results.png)


The original code has been further refined and modified. The following are the quantitative results of the current version on the MVTec Dataset and VisA Dataset.

Table 1: Quantitative Results of Unsupervised Experiment on MVTec Dataset

|                | **mAUROC_sp_max** | **mAP_sp_max** | **mF1_max_sp_max** | **mAUROC_px** | **mAP_px** | **mF1_max_px** | **mAUPRO_px** |
|:--------------:|:-----------------:|:--------------:|:------------------:|:-------------:|:----------:|:--------------:|:-------------:|
|   **carpet**   |      100.00       |     100.00     |       100.00       |     98.81     |   57.30    |     58.92      |     95.95     |
|    **grid**    |      100.00       |     100.00     |       100.00       |     99.24     |   45.88    |     49.85      |     97.64     |
|   **leather**  |      100.00       |     100.00     |       100.00       |     99.16     |   43.34    |     43.58      |     98.35     |
|    **tile**    |      100.00       |     100.00     |       100.00       |     96.59     |   71.68    |     64.09      |     88.93     |
|    **wood**    |       99.47       |     99.84      |       98.33        |     95.84     |   51.63    |     53.69      |     92.06     |
|   **bottle**   |      100.00       |     100.00     |       100.00       |     98.95     |   80.95    |     77.05      |     96.65     |
|    **cable**   |       99.57       |     99.74      |       97.83        |     97.51     |   61.10    |     62.99      |     92.05     |
|   **capsule**  |       97.57       |     99.48      |       97.72        |     98.78     |   41.99    |     47.50      |     96.14     |
|  **hazelnut**  |      100.00       |     100.00     |       100.00       |     99.11     |   65.72    |     65.56      |     95.94     |
|  **metal_nut** |      100.00       |     100.00     |       100.00       |     97.80     |   77.84    |     82.10      |     94.99     |
|    **pill**    |       99.15       |     99.86      |       98.56        |     99.05     |   79.95    |     75.37      |     96.36     |
|    **screw**   |       95.57       |     98.48      |       94.96        |     99.31     |   37.04    |     40.50      |     96.54     |
| **toothbrush** |       96.67       |     98.73      |       95.24        |     98.99     |   52.60    |     58.28      |     91.37     |
| **transistor** |       99.83       |     99.76      |       97.56        |     95.61     |   59.92    |     55.13      |     90.60     |
|   **zipper**   |       99.55       |     99.88      |       98.74        |     98.45     |   56.23    |     60.51      |     95.21     |
|   **average**   |       99.13       |     99.68      |       98.44        |     98.12     |   57.46    |     58.77      |     94.30     |


Table 2: Quantitative Results of Unsupervised Experiment on VisA Dataset

|                | **mAUROC_sp_max** | **mAP_sp_max** | **mF1_max_sp_max** | **mAUROC_px** | **mAP_px** | **mF1_max_px** | **mAUPRO_px** |
|:--------------:|:-----------------:|:--------------:|:------------------:|:-------------:|:----------:|:--------------:|:-------------:|
|    **pcb1**    |       96.07       |     94.43      |       93.78        |     99.77     |   86.12    |     78.36      |     95.83     |
|    **pcb2**    |       98.02       |     97.97      |       95.38        |     99.04     |   17.81    |     25.88      |     91.62     |
|    **pcb3**    |       97.33       |     97.28      |       92.23        |     99.27     |   31.25    |     30.53      |     93.76     |
|    **pcb4**    |       99.66       |     99.64      |       98.49        |     99.22     |   49.87    |     53.25      |     95.07     |
|  **macaroni1** |       93.83       |     90.93      |       86.91        |     99.78     |   19.02    |     23.76      |     98.29     |
|  **macaroni2** |       92.62       |     91.75      |       84.04        |     99.73     |   12.66    |     21.89      |     99.33     |
|  **capsules**  |       93.80       |     96.33      |       91.08        |     99.28     |   63.91    |     60.77      |     95.72     |
|   **candle**   |       98.06       |     98.03      |       94.63        |     99.42     |   23.62    |     34.59      |     96.92     |
|   **cashew**   |       97.36       |     98.75      |       94.42        |     98.75     |   51.07    |     52.52      |     95.54     |
| **chewinggum** |       99.43       |     99.68      |       98.99        |     99.14     |   57.44    |     56.03      |     90.52     |
|    **fryum**   |       97.48       |     98.79      |       94.12        |     96.94     |   44.90    |     50.58      |     93.29     |
| **pipe_fryum** |       99.12       |     99.56      |       97.00        |     99.08     |   52.60    |     56.90      |     95.37     |
| **average** |       96.54       |     96.20      |       92.88        |     99.07     |   40.34    |     43.65      |     94.80     |


Table 3: Quantitative Results of semi10 Experiment on MVTec Dataset

|                | **mAUROC_sp_max** | **mAP_sp_max** | **mF1_max_sp_max** | **mAUROC_px** | **mAP_px** | **mF1_max_px** | **mAUPRO_px** |
|:--------------:|:-----------------:|:--------------:|:------------------:|:-------------:|:----------:|:--------------:|:-------------:|
|   **carpet**   |      100.00       |     100.00     |       100.00       |     99.44     |   77.26    |     69.03      |     98.06     |
|    **grid**    |      100.00       |     100.00     |       100.00       |     99.31     |   47.25    |     51.51      |     97.81     |
|   **leather**  |      100.00       |     100.00     |       100.00       |     99.58     |   64.79    |     60.75      |     98.90     |
|    **tile**    |      100.00       |     100.00     |       100.00       |     99.03     |   93.34    |     85.09      |     94.19     |
|    **wood**    |       99.89       |     99.96      |       99.01        |     97.15     |   75.59    |     67.42      |     95.12     |
|   **bottle**   |      100.00       |     100.00     |       100.00       |     99.49     |   90.90    |     84.40      |     97.97     |
|    **cable**   |       99.39       |     99.60      |       97.01        |     97.20     |   72.03    |     63.97      |     91.51     |
|   **capsule**  |       98.81       |     99.71      |       98.49        |     98.86     |   44.79    |     49.35      |     96.72     |
|  **hazelnut**  |      100.00       |     100.00     |       100.00       |     99.47     |   83.28    |     74.39      |     96.35     |
|  **metal_nut** |      100.00       |     100.00     |       100.00       |     99.45     |   96.04    |     88.89      |     95.91     |
|    **pill**    |       98.88       |     99.79      |       97.73        |     99.24     |   85.89    |     77.39      |     96.82     |
|    **screw**   |       95.55       |     98.37      |       94.44        |     99.36     |   41.10    |     44.65      |     96.70     |
| **toothbrush** |       97.92       |     98.81      |       95.24        |     99.42     |   69.43    |     65.18      |     91.69     |
| **transistor** |      100.00       |     100.00     |       100.00       |     99.80     |   95.81    |     88.25      |     98.16     |
|   **zipper**   |      100.00       |     100.00     |       100.00       |     99.27     |   78.41    |     70.39      |     96.92     |
|   **average**   |      99.18       |     99.65     |       98.55       |     99.01     |   73.73    |     68.73      |     95.83     |


Table 4: Quantitative Results of semi10 Experiment on VisA Dataset

|                | **mAUROC_sp_max** | **mAP_sp_max** | **mF1_max_sp_max** | **mAUROC_px** | **mAP_px** | **mF1_max_px** | **mAUPRO_px** |
|:--------------:|:-----------------:|:--------------:|:------------------:|:-------------:|:----------:|:--------------:|:-------------:|
|    **pcb1**    |       97.20       |     95.45      |       94.57        |     99.80     |   89.77    |     82.90      |     95.46     |
|    **pcb2**    |       98.17       |     98.06      |       96.09        |     99.41     |   43.23    |     47.07      |     93.30     |
|    **pcb3**    |       97.34       |     97.11      |       92.13        |     99.27     |   29.59    |     29.12      |     93.03     |
|    **pcb4**    |       99.70       |     99.64      |       98.32        |     99.48     |   59.06    |     57.72      |     96.44     |
|  **macaroni1** |       95.17       |     91.12      |       89.89        |     99.78     |   18.09    |     22.49      |     98.42     |
|  **macaroni2** |       92.06       |     90.26      |       83.13        |     99.71     |   12.12    |     20.18      |     99.15     |
|  **capsules**  |       94.87       |     96.74      |       91.21        |     99.71     |   72.69    |     65.49      |     97.32     |
|   **candle**   |       97.72       |     97.77      |       93.33        |     99.55     |   30.29    |     39.28      |     96.89     |
|   **cashew**   |       98.71       |     99.33      |       96.17        |     99.84     |   94.73    |     89.91      |     96.12     |
| **chewinggum** |       99.78       |     99.87      |       99.44        |     99.27     |   70.83    |     64.77      |     90.38     |
|    **fryum**   |       97.29       |     98.66      |       93.71        |     97.25     |   56.69    |     54.73      |     92.16     |
| **pipe_fryum** |       99.60       |     99.78      |       97.80        |     99.50     |   76.15    |     68.20      |     95.33     |
| **average** |       96.88       |     96.54      |       93.00        |     99.31     |   52.78    |     51.95      |     94.46     |




## Reference CDO Qualitative Results 
![CDO](./pngs/qualitative_results.png)