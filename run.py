import gc
import logging
import os
import time
import numpy as np
import pandas as pd
import torch
from config.config import get_config_regression
from utils.ATIO import ATIO
from utils.dataset import MMDataLoader
from pathlib import Path
from utils.functions import assign_gpu, setup_seed
from StudentModel import student
from TeacherModel import teacher
import sys
from datetime import datetime

now = datetime.now()
format = "%Y/%m/%d %H:%M:%S"
formatted_now = now.strftime(format)  # 把 datetime 对象，按指定格式，转成字符串
formatted_now = str(formatted_now) + " - "  # 字符串拼接

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 固定写法，固定 GPU 编号顺序，避免多卡环境下设备映射混乱
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"  # 固定写法，强制 cuBLAS 使用确定性算法，提高实验结果可复现性
logger = logging.getLogger('MMSA')  # 查找一个名字叫 "MMSA" 的 Logger,如果已经存在 → 直接返回（单例）,如果不存在 → 创建一个新的 Logger 并返回,logging.getLogger是固定写法

def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger,根据传进来的log_dir在加上model_name, dataset_name构造出一条路径，
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)  # logging.DEBUG是固定写法，当然还有其它的

    # file handler，上一段代码建立了一个日志路径，这段代码应用这个日志执行处理操作,
    fh = logging.FileHandler(log_file_path)  # ！！！！！！！！注意log_file_path长什么样，带‘’号
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')  # name是 logger = logging.getLogger('MMSA')中的MMSA
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)  # MMSA的日志的保存路径是 log_file_path，处理方法是fh
    # 先建立一个logger，再建立一个文件处理器fh，最后用logger.addHandler把fh加入进去。
    # stream handler，以下这段代码的作用是根据用户指定的 verbose_level，控制终端里打印多少日志，并规定打印格式”
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()  ## StreamHandler()是写入控制台的。logging.FileHandler是写入对应文件的。
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')  # 按照 "%(name)s - %(message)s" 的格式打印到终端，%(name)s是建立logging.getLogger(__name__)时所起的名字
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)  # MMSA的日志输出到控制台时遵循ch方法
    return logger


def DLF_run(
        model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
        tune_times=500, feature_T="", feature_A="", feature_V="",
         res_save_dir="", log_dir="",
        gpu_ids=[0], num_workers=1, verbose_level=1, mode='', is_training=False
        # 运行train代码调用这个函数时传入is_training=true，即确实是训练模式
):  # 调用这个函数时，里面的参数有一部分可能指定，也可能没有指定
    # Initialization
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()

    if config_file != "":  # 说明函数的参数也可以是一个文件
        config_file = Path(config_file)
    else:  # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():  # p.is_file()     # 判断是不是文件，Path类中的方法
        raise ValueError(f"Config file {str(config_file)} not found.")

    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name,verbose_level)  # 函数_set_logger的参数在运行DLF_run时被传入，作为DLF_run函数参数的一部分
    config_file=str(config_file)
    args = get_config_regression(model_name, dataset_name,config_file)  # 这个函数get_config_regression的参数也作为DLF_run参数的一部分。默认情况下config路径是config_file = Path(__file__).parent / "config" / "config.json"
    args.is_training = is_training  # 上面的model的名字是DLF，args是一个能用点号访问的字典，也能用点号新增键值对，比如这两句代码，运行train代码时args新增属性is_training，值是true
    args.mode = mode  # train or test
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:  # train代码中运行dlf——run中config默认是空，所以config=none，此代码不执行。
        args.update(config)

    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []  # 在 Python 里创建（初始化）一个空列表。
    for i, seed in enumerate(seeds):  #
        setup_seed(seed)
        args['cur_seed'] = i + 1  # args是一个字典
        result = _run(args, num_workers, is_tune)  # 字典也可以作为一个参数传入函数，默认is_tune是False，运行train代码时is_tunr是true
        model_results.append(result)
    if args.is_training:  # 运行train代码时执行此条代码
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"  # 后缀是csv而已，具体是不是还是要看里面的内容。
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time"] + ["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values) * 100, 2)
            std = round(np.std(values) * 100, 2)
            res.append((mean, std))

        res = [formatted_now] + res
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    dataloader = MMDataLoader(args, num_workers)  # 调用函数_run时传入args是一个字典，因此这里args仍然是一个字典
    # 这里把某个数据集的train，valid，test部分全部准备好
    if args.is_training:  # 运行train代码时此代码执行
        print("training for DLF")

        args.gd_size_low = 64  # args字典增加下列参数
        args.w_losses_low = [1, 10]
        args.metric_low = 'l1'

        args.gd_size_high = 32
        args.w_losses_high = [1, 10]
        args.metric_high = 'l1'

        to_idx = [0, 1, 2]
        from_idx = [0, 1, 2]
        assert len(from_idx) >= 1  # 断言，from_idx 这个列表至少包含 1 个元素，否则程序就报错。assert后面的表达式为真，程序继续执行，否则断言失败，则报错

        model = []
        net_student=getattr(student, 'studentmodel')(args)  # DLF.py 文件中取出类 DLF，然后用 args 创建这个类的实例，从DLF.py文件里面获取DLF这个类，并实例化。
        net_teacher=getattr(teacher, 'teachermodel')(args)

        net_student=net_student.cuda()  # 把模型从 CPU 移到 GPU 上，以便用显卡加速计算
        net_teacher = net_teacher.cuda()
        model = [net_student, net_teacher]#model[0]是学生，model[1]是老师。
    else:
        print("testing phase for studentmodel")
        model = getattr(student, 'studentmodel')(args)
        model = model.cuda()

      # ATIO()相当于实例化类ATIO()的对象，然后由对象去调用类的方法。
    trainer = ATIO().getTrain(args)
    # test
    if args.mode == 'test':
        base = Path(r'F:\zhengliuxiangmu\studentxunlianjieguo')
        best_path = base /  args.dataset_name / 'best_model.pth'
        model.load_state_dict(torch.load(best_path),strict=False)  # 加载保存的模型。best_path = os.path.join(save_dir, "best_model.pth")
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()  # 把还没显示的输出内容立刻打印出来，不再等待
        input('[Press Any Key to start another run]')
        return results
    # train
    else:
        trainer.do_train(model, dataloader,return_epoch_results=from_sena)  # model = [model_DLF]，这里model是一个列表。
        #model[0].load_state_dict(torch.load('./MOSIxunlianjieguo/DLF' + str(args.dataset_name) + '.pth'))
        #results = trainer.do_test(model[0], dataloader['test'], mode="TEST")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
