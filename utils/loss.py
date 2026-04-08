import torch
import torch.nn.functional as F

def diff_loss(shared_feat, specific_feat):
    """正交损失 (Difference Loss)：强制 shared 和 specific 相互独立"""
    # 归一化后计算内积，期望正交（内积趋于0）
    shared_norm = F.normalize(shared_feat, p=2, dim=1)
    specific_norm = F.normalize(specific_feat, p=2, dim=1)
    correlation_matrix = torch.bmm(shared_norm.unsqueeze(1), specific_norm.unsqueeze(2))
    return torch.mean(correlation_matrix ** 2)


def sim_loss(s_t, s_a, s_v):
    """相似度损失 (Similarity Loss)：强制不同模态的 shared 空间对齐"""
    # 这里使用简单的 MSE，也可以替换为 CMD (Central Moment Discrepancy) 或 Contrastive Loss
    loss = F.mse_loss(s_t, s_a) + F.mse_loss(s_t, s_v) + F.mse_loss(s_a, s_v)
    return loss / 3.0


def calculate_decoupling_loss(shared_feats, specific_feats):
    """计算单个模型内部的解耦标准损失"""
    s_t, s_a, s_v = shared_feats
    p_t, p_a, p_v = specific_feats

    # 差异损失：每个模态的 shared 和 specific 必须正交
    l_diff = diff_loss(s_t, p_t) + diff_loss(s_a, p_a) + diff_loss(s_v, p_v)

    # 相似损失：三个模态的 shared 必须相似
    l_sim = sim_loss(s_t, s_a, s_v)

    return l_diff + l_sim


def kd_loss(stu_logits, tea_logits, temperature=2.0):
    """逻辑层蒸馏损失 (Logit Distillation)"""
    loss_kl = F.kl_div(
        F.log_softmax(stu_logits / temperature, dim=1),
        F.softmax(tea_logits / temperature, dim=1),
        reduction='batchmean'
    ) * (temperature ** 2)
    return loss_kl


def feature_distillation_loss(stu_feats, tea_feats):
    """特征层蒸馏损失 (Feature Distillation)"""
    loss = 0
    for s_f, t_f in zip(stu_feats, tea_feats):
        loss += F.mse_loss(s_f, t_f)
    return loss