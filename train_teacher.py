import os
import torch
from config.config import get_config_regression
from utils.dataset import MMDataLoader
from pathlib import Path
from utils.functions import assign_gpu, setup_seed
from TeacherModel.teacher import teachermodel  #
from teacher_only_run import TeacherTrainer
from teacher_only_run import _set_logger


def train_teacher_main():
    dataset_name = 'mosi'
    model_name = 'STUDENT'  # 复用 STUDENT 的配置参数

    # 获取配置
    config_file = Path(__file__).parent / "config" / "config.json"
    args = get_config_regression(model_name, dataset_name, str(config_file))

    args.is_training = True
    args.mode = 'train'
    args['device'] = assign_gpu([0])
    args['train_mode'] = 'regression'

    setup_seed(1111)

    # 设置日志
    log_dir = Path("utils/log/teacher")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger=_set_logger(str(log_dir), "TEACHER_ONLY", dataset_name, verbose_level=1)

    print("Initializing DataLoaders...")
    dataloader = MMDataLoader(args, num_workers=args.num_workers)

    print("Initializing Teacher Model...")
    # 实例化老师模型并放入 GPU
    model = teachermodel(args).cuda()

    print("Starting Phase 1: Training Teacher on Full Data...")
    trainer = TeacherTrainer(args)
    trainer.do_train(model, dataloader)


if __name__ == '__main__':
    train_teacher_main()