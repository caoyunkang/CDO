import os
import random
from .anomaly_generator import WhiteNoiseGenerator
import json

### custom_cover: 需要额外造缺陷
### screen: 需要额外造缺陷

def get_data_root(dataset):
    assert dataset in DATA_SUBDIR, f"Only support {DATA_SUBDIR.keys()}, but entered {dataset}"
    return os.path.join(DATA_ROOT, DATA_SUBDIR[dataset])


DATA_ROOT = '/home/user/zyx/datasets'

DATA_SUBDIR = {
    'mvtec': 'mvtec_anomaly_detection',
    'visa': 'VisA_20220922',
}

CLASS_NAMES = {
    'mvtec': [
        'carpet', 'grid', 'leather', 'tile', 'wood',
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper',
    ],
    'visa': [
        'pcb1', 'pcb2', 'pcb3', 'pcb4',
        'macaroni1', 'macaroni2', 'capsules', 'candle',
        'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
    ],

}

ANOMALY_GENERATOR = {
    'white_noise': WhiteNoiseGenerator
}

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

# Train and Test Set Configurations:
#
# - Zero-shot:
#     - Training set: original training set
#     - Testing set: original testing set
#
# - Few-shot:
#     - Training set: 1/2/4/8 normal samples from the training set
#     - Testing set: original testing set
#
# - Full-shot:
#     - Training set: original training set
#     - Testing set: original testing set
#
# - Semi-supervised:
#     - Training set: original training set + 1/5/10 anomalous samples from the testing set
#     - Testing set: original testing set (excluding selected anomalous samples)

def _split_meta_zero_shot(info: dict):
    setup_info = dict(train={}, test={})
    setup_info['test'] = info['test']

    for cls in info['train'].keys():
        # setup_info['train'][cls] = info['train'][cls] + info['test'][cls]  # for cross-dataset training
        setup_info['train'][cls] = info['test'][cls]  # for cross-dataset training

    return [setup_info], [EXPERIMENTAL_SETUP['zero_shot']]

def _split_meta_few_shot(info: dict, shots=[1, 2, 4, 8]):
    max_shot = max(shots)
    test_info = info['test']
    max_train_info = dict()

    for cls in info['train'].keys():
        assert max_shot < len(info['train'][cls]), f"{cls} only have {len(info['train'][cls])} samples, " \
                                                         f"but want to select {max_shot} shots"
        max_train_info[cls] = random.sample(info['train'][cls], max_shot)

    results_info = []
    results_json = []

    for shot in shots:
        cur_train_info = dict()

        for cls in info['train'].keys():
            cur_train_info[cls] = random.sample(max_train_info[cls] , shot)

        results_info.append({'train': cur_train_info, 'test': test_info})
        fmode = f'few_shot{shot}'
        results_json.append(EXPERIMENTAL_SETUP[fmode])

    return results_info, results_json

def _split_meta_unsupervised(info: dict):
    return [info], [EXPERIMENTAL_SETUP['unsupervised']]

def _split_meta_semi(info: dict, shots=[1, 5, 10]):
    max_shot = max(shots)
    train_anomalous_info = {}
    test_info = {}

    for cls in info['test'].keys():
        anomalous_samples = [item for item in info['test'][cls] if item['anomaly'] == 1]
        assert max_shot < len(anomalous_samples), f"{cls} only have {anomalous_samples} anomalous samples, " \
                                                         f"but want to select {max_shot} shots"
        selected_samples = random.sample(anomalous_samples, max_shot)
        remain_samples = [item for item in info['test'][cls] if item not in selected_samples]
        test_info[cls] = remain_samples
        train_anomalous_info[cls] = selected_samples

    results_info = []
    results_json = []

    for shot in shots:
        cur_train_info = {}
        for cls in train_anomalous_info.keys():
            cur_train_info[cls] = info['train'][cls] + random.sample(train_anomalous_info[cls] , shot)

        results_info.append({'train': cur_train_info, 'test': test_info})
        fmode = f'semi{shot}'
        results_json.append(EXPERIMENTAL_SETUP[fmode])

    return results_info, results_json

def split_meta(info: dict, root: str):
    results_info = []
    results_json = []

    _info, _json = _split_meta_zero_shot(info)
    results_info += _info
    results_json += _json


    _info, _json = _split_meta_few_shot(info)
    results_info += _info
    results_json += _json

    _info, _json = _split_meta_unsupervised(info)
    results_info += _info
    results_json += _json

    _info, _json = _split_meta_semi(info)
    results_info += _info
    results_json += _json

    for _info, _json in zip(results_info, results_json):
        full_path = os.path.join(root, _json)

        with open(full_path, 'w') as f:
            f.write(json.dumps(_info, indent=4) + "\n")

        print(f'Save {_json}...')