import torch
import torch.nn as nn


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def compute_cosine(self, x, y):
        # x = self.compute_compact_s(x)
        # y = self.compute_compact_s(y)
        x_norm = torch.sqrt(torch.sum(torch.pow(x, 2), 1) + 1e-8)  # torch.sum求和时dim=1，如果没有dim=1，那么他会把所有元素加起来得到一个数，加上1e-8：防止数值为 0
        x_norm = torch.max(x_norm, 1e-8 * torch.ones_like(x_norm))  # 防止范数过小，避免后面除以 0，
        y_norm = torch.sqrt(torch.sum(torch.pow(y, 2), 1) + 1e-8)
        y_norm = torch.max(y_norm, 1e-8 * torch.ones_like(y_norm))
        cosine = torch.sum(x * y, 1) / (x_norm * y_norm)  # dim=1进行求和
        return cosine  # 这段代码在 batch 维度上手动计算余弦相似度，并通过多重下界保护避免数值不稳定。

    def forward(self, ids, feats, margin=0.1):
        B, F = feats.shape  #

        s = feats.repeat(1, B).view(-1, F)  # s的形状是B**2，F
        s_ids = ids.view(B, 1).repeat(1, B)  # s_ids的形状是B，B

        t = feats.repeat(B, 1)  # t的形状是 B**2，F
        t_ids = ids.view(1, B).repeat(B, 1)  # 形状是B，B

        cosine = self.compute_cosine(s, t)  # 结果形状是B*B，1
        equal_mask = torch.eye(B, dtype=torch.bool)  # equal_mask形状是B X B，里面对角线元素是True，除对角线之外元素全部是False
        s_ids = s_ids[~equal_mask].view(B, B - 1)  # 形状是B,(B-1)
        t_ids = t_ids[~equal_mask].view(B, B - 1)  # 形状是B,(B-1)
        cosine = cosine.view(B, B)[~equal_mask].view(B, B - 1)  # B X (B-1)

        sim_mask = (s_ids == t_ids)  # B，(B-1),s_ids == t_ids返回的是bool类型的矩阵，里面的元素要么是true，要么是false
        margin = 0.15 * abs(s_ids - t_ids)  # [~sim_mask].view(B, B - 3)，abs是计算绝对值的

        loss = 0
        loss_num = 0

        for i in range(B):
            sim_num = sum(sim_mask[i])  # sim_mask矩阵是bool类型的，形状是B，(B-1)，这里用到了矩阵[]取元素问题，sum（）表示统计出true的元素有多少个,比如i=0时，统计出sim_mask第一行中的true有多少个，i=2时统计第二行的true有多少个，以此类推
            dif_num = B - 1 - sim_num  # 统计出每一行的false有多少个
            if not sim_num or not dif_num:  # 当 sim_num=0或者dif_num=0时，跳出本次循环，也就是说下面的代码不执行了，i重新给值.
                continue
            sim_cos = cosine[i, sim_mask[i]].reshape(-1, 1).repeat(1, dif_num)
            dif_cos = cosine[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)
            t_margin = margin[i, ~sim_mask[i]].reshape(-1, 1).repeat(1, sim_num).transpose(0, 1)

            loss_i = torch.max(torch.zeros_like(sim_cos), t_margin - sim_cos + dif_cos).mean()
            loss += loss_i
            loss_num += 1

        if loss_num == 0:
            loss_num = 1  # 避免 loss_num = 0 的数值问题。否则在下面求平均损失的代码中，0可能被除，出现报错

        loss = loss / loss_num  # 如果一次性送来5个样本，那就计算5个样本的平均损失。
        return loss

# 这段代码：batch 内构造 所有 pairwise cosine，✔ 用 ids 区分 同类 / 异类，✔ 用 pairwise hinge ranking loss 约束：
# 同类要更像，异类要更不像，差距 ≥ 自适应 margin