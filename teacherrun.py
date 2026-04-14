import copy
import logging
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils.metric import MetricsTop, dict_to_str
from utils.HingeLoss import HingeLoss
import numpy as np
logger = logging.getLogger('MMSA')
from utils.loss import feature_distillation_loss, kd_loss, diff_loss
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)#这里diffs=real-pred
        n = torch.numel(diffs.data)#计算diffs里面有多少个数,diffs是tensor类型的数据，diffs.data就是访问diffs里面的元素，然后用torch.numel计算数量
        mse = torch.sum(diffs.pow(2)) / n#torch.pow就是完成指数运算的，这里就是把diffs里面每个元素都平方，然后把所有元素的平方和加起来再除以n
        return mse#这个是MSE（Mean Squared Error，均方误差）损失函数代码的实现

class studentmodel():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()#L1Loss = 预测值和真实值的“绝对误差”
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)#MetricsTop(args.train_mode)相当于初始化类MetricsTop的对象，然后由对象去调用类的方法getMetics
        self.MSE = MSE()#在另一个类的初始化方法里面，可以实例化其他的类并创建类的对象。
        self.sim_loss = HingeLoss()
        # ⭐ 新增：EMA参数
        self.ema_decay = 0.995
        self.warmup_epoch = 4  # ⭐ 延迟蒸馏
        self.kd_ramp_epochs = 4
        self.feat_kd_max_weight = 0.15
        self.logit_kd_max_weight = 0.05
        self.warmup_epoch = 4
        self.kd_ramp_epochs = 4
        self.bert_freeze_epochs = 3
        self.bert_lr_mult = 0.1

    def _logit_kd_loss(self, stu_logits, tea_logits):
        # 当前任务是回归（输出维度=1）时，KL蒸馏没有信息量，改为回归蒸馏更稳定
        if stu_logits.size(-1) == 1:
            return F.smooth_l1_loss(stu_logits, tea_logits)
        return kd_loss(stu_logits, tea_logits)

    def _consistency_loss(self, logits_missing, logits_full, temperature=2.0):
        # 同一样本在“完整模态/缺失模态”下输出应一致；回归任务用L1更合适
        if logits_missing.size(-1) == 1:
            return F.smooth_l1_loss(logits_missing, logits_full.detach())
        p_missing = F.log_softmax(logits_missing / temperature, dim=-1)
        p_full = F.softmax(logits_full.detach() / temperature, dim=-1)
        return F.kl_div(p_missing, p_full, reduction='batchmean') * (temperature * temperature)

    def _get_kd_scale(self,epoch):
        if epoch < self.warmup_epoch:
            return 0.0
        progress = (epoch - self.warmup_epoch + 1) / max(1, self.kd_ramp_epochs)
        return min(1.0, max(0.0, progress))

    def _set_bert_trainable(self, model, trainable):
        if not hasattr(model, "text_model"):
            return
        for param in model.text_model.parameters():
            param.requires_grad = trainable


    def do_train(self, model, dataloader, return_epoch_results=False):
        # 0: DLF model
        named_params = list(model[0].named_parameters())
        bert_params = [p for n, p in named_params if "text_model" in n]
        other_params = [p for n, p in named_params if "text_model" not in n]
        optimizer = optim.Adam(
            [
                {"params": other_params, "lr": self.args.learning_rate},
                {"params": bert_params, "lr": self.args.learning_rate * self.bert_lr_mult},
            ],
            weight_decay=self.args.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=self.args.patience)
        net = []
        net_teacher = model[1]
        net_student=model[0]##0是学生，1是老师。
        net.append(net_student)#net[0]是学生，
        net.append(net_teacher)#net[1]是老师
        model = net
        base_save_dir = r'studentxunlianjieguo'
        # 生成子文件夹（用数据集名称命名）
        save_dir = os.path.join(base_save_dir, self.args.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        best_valid=float('inf')
        best_epoch = 0
        best_state_dict = copy.deepcopy(net_student.state_dict())
        num_epochs = self.args.update_epochs
        teacher_weights_path = os.path.join('teacher_best_weights', self.args.dataset_name,
                                            "best_teacher_full_data.pth")
        net_teacher.load_state_dict(torch.load(teacher_weights_path))
        for param in net_teacher.parameters():
            param.requires_grad = False
        from copy import deepcopy
        ema_student = deepcopy(net_student)
        for param in ema_student.parameters():
            param.requires_grad = False
        ema_decay = self.ema_decay
        ema_student.eval()
        for epoch in range(num_epochs):
            y_pred, y_true = [], []
            net[1].eval()  # 老师
            for param in net_teacher.parameters():
                param.requires_grad = False
            self._set_bert_trainable(net_student, epoch >= self.bert_freeze_epochs)
            net[0].train()  # 学生
            train_loss = 0.0

            with tqdm(dataloader['train']) as td:
                for batch_data in td:  # batchsize_size=16
                    optimizer.zero_grad()

                    vision = batch_data['vision'].to(self.args.device)  #
                    audio = batch_data['audio'].to(self.args.device)  #
                    text = batch_data['text'].to(self.args.device)  #

                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)
                    with torch.no_grad():
                        output_tea = net[1](text, audio, vision)
                    output_stu = net[0](text_m, audio_m, vision_m)

                    # task loss
                    loss_task_all = self.criterion(output_stu['output_logit'], labels)#三种模态不变特征和特定特征拼接起来再预测的损失
                    loss_task_c = self.criterion(output_stu['logits_c'], labels)#三种模态不变特征预测的损失
                    loss_task =loss_task_all + 0.5* loss_task_c#损失1


                    # ort loss L_o
                    dim_l = output_stu['s_l'].size(-1)
                    dim_v = output_stu['s_v'].size(-1)
                    dim_a = output_stu['s_a'].size(-1)

                    s_l = output_stu['s_l'].contiguous().view(-1, dim_l)
                    c_l = output_stu['c_l'].contiguous().view(-1, dim_l)
                    s_v = output_stu['s_v'].contiguous().view(-1, dim_v)
                    c_v = output_stu['c_v'].contiguous().view(-1, dim_v)
                    s_a = output_stu['s_a'].contiguous().view(-1, dim_a)
                    c_a = output_stu['c_a'].contiguous().view(-1, dim_a)

                    # 使用 utils/loss.py 中正确的 diff_loss 让它们正交（相互独立）
                    loss_ort = diff_loss(s_l, c_l) + diff_loss(s_v, c_v) + diff_loss(s_a, c_a)

                    # 1. 获取当前 epoch 的蒸馏比例 (0.0 ~ 1.0)
                    kd_scale = self._get_kd_scale(epoch)

                    # ========== 核心修改 1：动态衰减任务损失权重 ==========
                    # 随着 kd_scale 从 0 增加到 1，task_weight 会从 1.0 平滑下降到 0.2
                    task_weight = max(1.0 - 0.8 * kd_scale, 0.2)

                    if kd_scale == 0.0:
                        # 预热期：主要关注任务损失和正交解耦损失
                        total_loss = task_weight * loss_task + (loss_ort * 0.1)
                    else:
                        # 准备学生特征
                        stu_feats = [
                            output_stu['c_l'], output_stu['c_v'], output_stu['c_a'],
                            output_stu['s_l'], output_stu['s_v'], output_stu['s_a']
                        ]

                        # ========== 核心修改 2：给老师的所有特征加上 detach() ==========
                        tea_feats = [
                            output_tea['c_l'].detach(), output_tea['c_v'].detach(), output_tea['c_a'].detach(),
                            output_tea['s_l'].detach(), output_tea['s_v'].detach(), output_tea['s_a'].detach()
                        ]

                        # 计算特征蒸馏损失
                        loss_feat_kd = feature_distillation_loss(stu_feats, tea_feats)

                        # 计算结构一致性损失 (强迫学生特定的模态特征向老师对齐)
                        loss_structure = (
                                F.mse_loss(output_stu['s_l'], output_tea['s_l'].detach()) +
                                F.mse_loss(output_stu['s_v'], output_tea['s_v'].detach()) +
                                F.mse_loss(output_stu['s_a'], output_tea['s_a'].detach())
                        )

                        # 计算一致性损失 (老师的 logits 需要 detach)
                        loss_cons = self._consistency_loss(
                            output_stu['output_logit'],
                            output_tea['output_logit'].detach()
                        )

                        # 计算逻辑蒸馏损失 (使用误差感知权重，老师的 logits 需要 detach)
                        reg_weight = torch.exp(-torch.abs(output_tea['output_logit'].detach() - labels))
                        loss_logit_kd = (self._logit_kd_loss(output_stu['output_logit'],
                                                             output_tea['output_logit'].detach()) * reg_weight.mean())

                        loss_feat_kd = torch.mean(loss_feat_kd)
                        loss_structure = torch.mean(loss_structure)

                        # ========== 核心修改 3：调整最终 Total Loss 的融合 ==========
                        total_loss = (
                                task_weight * loss_task  # <--- 使用动态衰减的任务损失权重
                                + 0.1 * loss_ort  # 正交解耦损失保持恒定
                                + kd_scale * (
                                        0.5 * loss_feat_kd +  # 特征对齐 (权重 0.4 -> 0.5)
                                        0.3 * loss_logit_kd +  # 逻辑对齐 (权重 0.3)
                                        0.3 * loss_structure +  # 结构对齐 (权重 0.2 -> 0.3)
                                        0.2 * loss_cons  # 一致性对齐 (权重 0.2)
                                )
                        )

                    total_loss.backward()
                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_norm_(params, self.args.grad_clip)
                    optimizer.step()#更新梯度
                    # ⭐ 只在前期更新 teacher（防止后期过拟合）

                    train_loss += total_loss.item()  # train_loss是累加的
                    y_pred.append(output_stu['output_logit'].cpu())
                    y_true.append(labels.cpu())#什么时候for循环退出，也就是把某个数据集中用于train的数据全部摸了一遍。
                    with torch.no_grad():
                        for ema_param, s_param in zip(ema_student.parameters(), net_student.parameters()):
                            ema_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1 - self.ema_decay)
            train_loss = train_loss / len(dataloader['train'])#len(dataloader) = batch 数量
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epoch+1} "
                f"TRAIN -({self.args.model_name}) [{epoch+1}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(ema_student, dataloader['valid'], mode="VAL")
                #cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
                #torch.save(model[0].state_dict(),
                       #r'F:\zhengliuxiangmu\studentxunlianjieguo' + str(self.args.dataset_name) + '_' + str(epoch+1) + '.pth')#？修改
            # save best model
            if val_results['Loss'] < best_valid:
                    best_valid = val_results['Loss']
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(ema_student.state_dict())
            if (epoch+1) - best_epoch >= self.args.early_stop:
                logger.info(f"Early stop at epoch {epoch + 1}")
                break
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(best_state_dict, best_path)
        net_student.load_state_dict(best_state_dict)
        best_test = self.do_test(net_student, dataloader['test'], mode="TEST")
        logger.info(f"Best Epoch: {best_epoch}")
        logger.info(f"Best Test: {best_test}")


    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()#student模型。
        y_pred, y_true = [], []

        eval_loss = 0.0

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                            # !!! 测试阶段使用带缺失标志的模态输入 !!!
                    vision_m = batch_data['vision_m'].to(self.args.device)
                    audio_m = batch_data['audio_m'].to(self.args.device)
                    text_m = batch_data['text_m'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                            # 测试模型表现 (此时 model 就是 net_student)
                    output = model(text_m, audio_m, vision_m)#这里的模型是学生模型。

                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()  # item（）Returns the value of this tensor as a standard Python number
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)  # 这个 dataloader 一共会返回多少个 batch（循环会跑多少次），用于计算整个流程的平均 loss
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)  # 保留eval_loss的4位小数。
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        return eval_results