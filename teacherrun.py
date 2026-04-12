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
from utils.loss import feature_distillation_loss
from utils.loss import kd_loss
logger = logging.getLogger('MMSA')
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

    def do_train(self, model, dataloader, return_epoch_results=False):
        # 0: DLF model
        params = model[0].parameters()

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)
        net = []
        net_teacher = model[1]
        net_student=model[0]##0是学生，1是老师。
        net.append(net_student)#net[0]是学生，
        net.append(net_teacher)#net[1]是老师
        model = net
        base_save_dir = r'/studentxunlianjieguo'
        # 生成子文件夹（用数据集名称命名）
        save_dir = os.path.join(base_save_dir, self.args.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        best_valid = float('inf')
        best_epoch = 0

        num_epochs=10
        for epoch in range(num_epochs):
            y_pred, y_true = [], []
            net[1].eval()  # 老师
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
                    output_stu_full=net[0](text,audio,vision)
                    # task loss
                    loss_task_all = self.criterion(output_stu['output_logit'], labels)#三种模态不变特征和特定特征拼接起来再预测的损失
                    loss_task_c = self.criterion(output_stu['logits_c'], labels)#三种模态不变特征预测的损失
                    loss_task =loss_task_all + 0.5* loss_task_c#损失1

                    logits_missing = output_stu['output_logit']
                    logits_full = output_stu_full['output_logit']
                    T = 2.0  # 温度（推荐 2~4）
                    p_missing = F.log_softmax(logits_missing / T, dim=-1)
                    p_full = F.softmax(logits_full / T, dim=-1)
                    loss_consistency = F.kl_div(p_missing, p_full, reduction='batchmean') * (T * T)#损失2，同一个样本，在“缺失模态”和“完整模态”两种输入下，模型输出应该一致。

                # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10
                    cosine_similarity_s_c_l = self.cosine(output_stu['s_l'].reshape(-1, num),output_stu['c_l'].reshape(-1, num),torch.tensor([-1]).cuda())  # 拉开语言模态的特定特征和不变特征的距离，让它们不相似。
                    cosine_similarity_s_c_v = self.cosine(output_stu['s_v'].reshape(-1, num),output_stu['c_v'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_a = self.cosine(output_stu['s_a'].reshape(-1, num),output_stu['c_a'].reshape(-1, num), torch.tensor([-1]).cuda())
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a#损失3

                    stu_feats = [output_stu['s_l'], output_stu['s_v'], output_stu['s_a'], output_stu['c_l'],
                             output_stu['c_v'], output_stu['c_a']]
                    tea_feats = [output_tea['s_l'], output_tea['s_v'], output_tea['s_a'], output_tea['c_l'],
                             output_tea['c_v'], output_tea['c_a']]
                    loss_feat_kd = feature_distillation_loss(stu_feats, tea_feats)#损失4
                # 逻辑蒸馏 (Logit Distillation)
                    loss_logit_kd = kd_loss(output_stu['output_logit'], output_tea['output_logit'])#损失5
                # --- 7. 总损失回传 ---
                    # 0.1 和 0.5 为蒸馏超参，你可以根据实验结果调整 (也可以写进 args 里)
                    total_loss = loss_task + ( loss_ort * 0.1) + 0.05 * loss_feat_kd + 0.3 * loss_logit_kd+0.2 * loss_consistency
                    total_loss.backward()
                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)
                    optimizer.step()
                    train_loss += total_loss.item()  # train_loss是累加的
                    y_pred.append(output_stu['output_logit'].cpu())
                    y_true.append(labels.cpu())#什么时候for循环退出，也就是把某个数据集中用于train的数据全部摸了一遍。


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
                val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
                #cur_valid = val_results[self.args.KeyEval]
                scheduler.step(val_results['Loss'])
            # save each epoch model
                #torch.save(model[0].state_dict(),
                       #r'F:\zhengliuxiangmu\studentxunlianjieguo' + str(self.args.dataset_name) + '_' + str(epoch+1) + '.pth')#？修改
            # save best model
                if val_results['Loss'] < best_valid:
                    best_valid = val_results['Loss']
                    best_epoch = epoch + 1

            if epoch - best_epoch >= self.args.early_stop:
                logger.info(f"Early stop at epoch {epoch + 1}")
                break
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(net_student.state_dict(), best_path)
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