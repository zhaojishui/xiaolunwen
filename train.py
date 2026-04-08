from run import DLF_run

DLF_run(model_name='STUDENT', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="MOSEIxunlianjieguo",
        res_save_dir="./result", log_dir="./log", mode='train', is_training=True)