import json
import os
from pathlib import Path

import yaml
from easydict import EasyDict as edict


def get_config_regression(model_name, dataset_name, config_file=""):
    """
    Get the regression config of given dataset and model from config file.

    Parameters:
        config_file (str): Path to config file, if given an empty string, will use default config file.
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.

    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == "":
        config_file = Path(__file__).parent / "config" / "config.json"#从def——run运行下来的话，config文件都是存在的，要么默认，要么用自己的
    with open(config_file, 'r') as f:#这里的r模式无法创建文件
        config_all = json.load(f)#这里的
    model_common_args = config_all[model_name]['commonParams']#这里的model_name在train代码中被赋予DLF，字典
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']
#dataset_args['aligned']就是对dataset_args再一次筛选
    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(dataset_args)#这里使得config可以访问到featurepath，根据dataset_name和数据集是否对齐可以访问每一个featurepath
    config.update(model_common_args)#这里使得config可以访问到"use_bert"键，这里使得config可以访问到need_data_aligned"
    config.update(model_dataset_args)
    config['featurePath'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['featurePath'])
    config['train_mode']=config_all['base']['train_mode']
    config['missing_rate_eval_test']=config_all['base']['missing_rate_eval_test']
    config['missing_seed']=config_all['base']['seed']
    config['batch_size']=config_all['base']['batch_size']
    config['num_workers']=config_all['base']['num_workers']
    config = edict(config)#利用edict，config里面的元素能用点号访问，此时config还是一个字典

    return config
