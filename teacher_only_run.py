import logging
import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.metric import MetricsTop, dict_to_str
from utils.loss import diff_loss  # 引入你写好的正交损失
from TeacherModel.teacher import teachermodel
logger = logging.getLogger('train-teacher')
def _set_logger(log_dir, model_name, dataset_name, verbose_level):
    # base logger,根据传进来的log_dir在加上model_name, dataset_name构造出一条路径，
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('train-teacher')
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

class TeacherTrainer():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.cosine = nn.CosineEmbeddingLoss()

    def do_train(self, model, dataloader ):
        # 注意：这里传进来的 model 直接是 Teacher 模型实例
        net_teacher = model.cuda()
        optimizer = optim.Adam(net_teacher.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        best_valid = float('inf')
        best_epoch = 0
        best_state_dict = copy.deepcopy(net_teacher.state_dict())

        # 专门给 Teacher 建一个保存权重的文件夹
        save_dir = os.path.join(r'teacher_best_weights', self.args.dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        num_epochs = self.args.update_epochs

        for epoch in range(num_epochs):
            y_pred, y_true = [], []
            net_teacher.train()
            train_loss = 0.0

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    optimizer.zero_grad()

                    # 【关键点 1：只提取完整数据，完全不用 vision_m, audio_m 等】
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    # 【关键点 2：前向传播】
                    output_tea = net_teacher(text, audio, vision)

                    # 【关键点 3：计算任务损失】
                    loss_task_all = self.criterion(output_tea['output_logit'], labels)
                    loss_task_l_hetero = self.criterion(output_tea['logits_l_hetero'], labels)
                    loss_task_v_hetero = self.criterion(output_tea['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output_tea['logits_a_hetero'], labels)

                    loss_task_c = self.criterion(output_tea['logits_c'], labels)
                    loss_task = 1 * (1 * loss_task_all + 1 * loss_task_c + 3 * loss_task_l_hetero + 1 * loss_task_v_hetero + 1 * loss_task_a_hetero)

                    # 【关键点 4：计算正交解耦损失】
                    dim_l = output_tea['s_l'].size(-1)
                    dim_v = output_tea['s_v'].size(-1)
                    dim_a = output_tea['s_a'].size(-1)

                    s_l = output_tea['s_l'].contiguous().view(-1, dim_l)
                    c_l = output_tea['c_l'].contiguous().view(-1, dim_l)
                    s_v = output_tea['s_v'].contiguous().view(-1, dim_v)
                    c_v = output_tea['c_v'].contiguous().view(-1, dim_v)
                    s_a = output_tea['s_a'].contiguous().view(-1, dim_a)
                    c_a = output_tea['c_a'].contiguous().view(-1, dim_a)
                    cosine_similarity_s_c_l = self.cosine(output_tea['s_l'].reshape(-1, dim_l),
                                                          output_tea['c_l'].reshape(-1, dim_l),
                                                          torch.tensor([-1]).cuda())  # 标签是-1，最小化它们之间的余弦相似度。
                    cosine_similarity_s_c_v = self.cosine(output_tea['s_v'].reshape(-1, dim_v),
                                                          output_tea['c_v'].reshape(-1, dim_v), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_a = self.cosine(output_tea['s_a'].reshape(-1, dim_a),
                                                          output_tea['c_a'].reshape(-1, dim_a), torch.tensor([-1]).cuda())


                    loss_ort = diff_loss(s_l, c_l) + diff_loss(s_v, c_v) + diff_loss(s_a, c_a)+cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # 【关键点 5：加总与反向传播，没有复杂的蒸馏预热机制】
                    total_loss = loss_task + 0.1 * loss_ort

                    total_loss.backward()
                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_(net_teacher.parameters(), self.args.grad_clip)
                    optimizer.step()

                    train_loss += total_loss.item()
                    y_pred.append(output_tea['output_logit'].cpu().detach())
                    y_true.append(labels.cpu().detach())
            y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(y_pred, y_true)
            logger.info(
                f">> Epoch: {epoch + 1} "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # 验证阶段 (do_test 的写法和上面类似，只是不需要梯度)
            val_results = self.do_test(net_teacher, dataloader['valid'], mode="VAL")
            scheduler.step(val_results['Loss'])
            logger.info(
                f">>验证集测试结果"
                f"{dict_to_str(val_results)}"
            )
            # 保存验证集上表现最好的模型
            if val_results['Loss'] < best_valid:
                best_valid = val_results['Loss']
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(net_teacher.state_dict())

            # Early stopping...
        logger.info(f"Best Epoch: {best_epoch}")
        # 训练结束后，保存最强 Teacher 的权重
        best_path = os.path.join(save_dir, "best_teacher_full_data.pth")
        torch.save(best_state_dict, best_path)
        net_teacher.load_state_dict(best_state_dict)
        logger.info(f"Teacher Pre-training Done! Best weights saved to {best_path}")
        best_test = self.do_test(net_teacher, dataloader['test'], mode="TEST")
        logger.info(
            f">>测试集测试结果"
            f"{dict_to_str(best_test)}"
        )
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Best Test: {best_test}")
    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    # 测试时同样只用完整数据
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)

                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        return eval_results