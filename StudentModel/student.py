"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.bert import BertTextEncoder
from utils.transformer import TransformerEncoder


class studentmodel(nn.Module):
    def __init__(self, args):
        super(studentmodel, self).__init__()
        if args.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, transformers=args.transformers,
                                              pretrained=args.pretrained)  # 这一步只是完成类BertTextEncoder的初始化，还没有调用类的forward方法。
        self.use_bert = args.use_bert
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        if args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims  # orig_d_*原始输入特征维度，768，74，35
        self.d_l = self.d_a = self.d_v = dst_feature_dims  # d_*Transformer 的隐空间维度，50
        self.num_heads = nheads
        self.layers = args.nlevels  # 2
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask
        combined_dim_low = self.d_a  # 50
        combined_dim_high = self.d_a  # 50
        combined_dim = (self.d_l + self.d_a + self.d_v) + self.d_l * 3  # 300

        output_dim = 1

        # 1. Temporal convolutional layers for initial feature
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0,
                                bias=False)  # 定义了Conv1d卷积网络，后面如果使用的话就往self.proj_l 里面传入输入参数
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 2. Modality-specific encoder
        self.encoder_s_l = self.get_network(self_type='l', layers=self.layers)
        self.encoder_s_v = self.get_network(self_type='v', layers=self.layers)
        self.encoder_s_a = self.get_network(self_type='a', layers=self.layers)

        #   Modality-shared encoder
        self.encoder_c = self.get_network(self_type='l', layers=self.layers)

        # 3. Decoder for reconstruct three modalities
        self.decoder_l = nn.Conv1d(self.d_l * 2, self.d_l, kernel_size=1, padding=0, bias=False)
        self.decoder_v = nn.Conv1d(self.d_v * 2, self.d_v, kernel_size=1, padding=0, bias=False)
        self.decoder_a = nn.Conv1d(self.d_a * 2, self.d_a, kernel_size=1, padding=0, bias=False)

        # for calculate cosine sim between s_x
        self.proj_cosine_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                       combined_dim_low)  # （2400，50）
        self.proj_cosine_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                       combined_dim_low)  # （2400，50）
        self.proj_cosine_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                       combined_dim_low)  # （2400，50）

        # for align c_l, c_v, c_a
        self.align_c_l = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                   combined_dim_low)  # （2400，50）
        self.align_c_v = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                   combined_dim_low)  # （2400，50）
        self.align_c_a = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                   combined_dim_low)  # （2400，50）

        self.self_attentions_c_l = self.get_network(self_type='l')
        self.self_attentions_c_v = self.get_network(self_type='v')
        self.self_attentions_c_a = self.get_network(self_type='a')

        self.proj1_c = nn.Linear(self.d_l * 3, self.d_l * 3)  # （150，150）
        self.proj2_c = nn.Linear(self.d_l * 3, self.d_l * 3)
        self.out_layer_c = nn.Linear(self.d_l * 3, output_dim)  # （150，1）

        # 4 Multimodal Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la', layers=self.layers)
        self.trans_l_with_v = self.get_network(self_type='lv', layers=self.layers)
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=self.layers)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)

        # 5. fc layers for shared features
        self.proj1_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                     combined_dim_low)  # （2400.50）
        self.proj2_l_low = nn.Linear(combined_dim_low,
                                     combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1))  # （50，2400）
        self.out_layer_l_low = nn.Linear(combined_dim_low * (self.len_l - args.conv1d_kernel_size_l + 1),
                                         output_dim)  # （2400，1）
        self.proj1_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                     combined_dim_low)  # （2400.50）
        self.proj2_v_low = nn.Linear(combined_dim_low,
                                     combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1))  # （50，2400）
        self.out_layer_v_low = nn.Linear(combined_dim_low * (self.len_v - args.conv1d_kernel_size_v + 1),
                                         output_dim)  # （2400，1）
        self.proj1_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                     combined_dim_low)  # （2400.50）
        self.proj2_a_low = nn.Linear(combined_dim_low,
                                     combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1))  # （50，2400）
        self.out_layer_a_low = nn.Linear(combined_dim_low * (self.len_a - args.conv1d_kernel_size_a + 1),
                                         output_dim)  # （2400，1）

        # 6. fc layers for specific features
        self.proj1_l_high = nn.Linear(combined_dim_high, combined_dim_high)  # （50，50）
        self.proj2_l_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_l_high = nn.Linear(combined_dim_high, output_dim)  # （50，1）
        self.proj1_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_v_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_v_high = nn.Linear(combined_dim_high, output_dim)
        self.proj1_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.proj2_a_high = nn.Linear(combined_dim_high, combined_dim_high)
        self.out_layer_a_high = nn.Linear(combined_dim_high, output_dim)

        # 7. project for fusion
        self.projector_l = nn.Linear(self.d_l, self.d_l)  # （50，50）
        self.projector_v = nn.Linear(self.d_v, self.d_v)
        self.projector_a = nn.Linear(self.d_a, self.d_a)
        self.projector_c = nn.Linear(3 * self.d_l, 3 * self.d_l)

        # 8. final project
        self.proj1 = nn.Linear(combined_dim, combined_dim)  # （300，300）
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)  # （300，1）

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout  # 50，0.5
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,  # 这里的TransformerEncoder是一个类，是自己编写的，所以参数不遵循标准的TransformerEncoder
                                  num_heads=self.num_heads,  # 10
                                  layers=max(self.layers, layers),  # 2
                                  attn_dropout=attn_dropout,  # 0.5
                                  relu_dropout=self.relu_dropout,  # 0
                                  res_dropout=self.res_dropout,  # 0
                                  embed_dropout=self.embed_dropout,  # 0
                                  attn_mask=self.attn_mask)  # true

    def forward(self, text, audio, video):
        # extraction
        if self.use_bert:
            text = self.text_model(text)  # 这里的text是text_bert，送入bert模型进行处理。text的形状：batch_size,seq-len,hidden-size
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout,
                        training=self.training)  # x_l形状：batch_size，hidden-size，seq-len
        x_a = audio.transpose(1, 2)  # 使用transpose是为了让数据进入卷积层。
        x_v = video.transpose(1, 2)

        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)  # 输出的形状还是（(batch, hidden, seq_len)

        proj_x_l = proj_x_l.permute(2, 0, 1)  # 为了让数据进入transformer层，因为它要求的输入形状是(src_len, batch, embed_dim)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)

        # disentanglement
        s_l = self.encoder_s_l(proj_x_l)
        s_v = self.encoder_s_v(proj_x_v)
        s_a = self.encoder_s_a(proj_x_a)

        c_l = self.encoder_c(proj_x_l)
        c_v = self.encoder_c(proj_x_v)
        c_a = self.encoder_c(proj_x_a)  # 形状是(src_len, batch, embed_dim)

        s_l = s_l.permute(1, 2, 0)  # 形状变成（batch，embed_dim,src_len)
        s_v = s_v.permute(1, 2, 0)
        s_a = s_a.permute(1, 2, 0)

        c_l = c_l.permute(1, 2, 0)
        c_v = c_v.permute(1, 2, 0)
        c_a = c_a.permute(1, 2, 0)
        c_list = [c_l, c_v, c_a]

        c_l_sim = self.align_c_l(c_l.contiguous().view(x_l.size(0),
                                                       -1))  # c_l.contiguous().view(x_l.size(0), -1)形状是batch，hidden-size*seq-len，，也就是变成二维的
        c_v_sim = self.align_c_v(c_v.contiguous().view(x_l.size(0), -1))  # 得到的结果还是batch_size,combined_dim_low
        c_a_sim = self.align_c_a(c_a.contiguous().view(x_l.size(0), -1))

        recon_l = self.decoder_l(
            torch.cat([s_l, c_list[0]], dim=1))  # 把共享和特殊空间的l特征拼接起来，decoder_l是一个卷积，正好形状是（batch，embed_dim,src_len)
        recon_v = self.decoder_v(torch.cat([s_v, c_list[1]], dim=1))
        recon_a = self.decoder_a(torch.cat([s_a, c_list[2]], dim=1))

        recon_l = recon_l.permute(2, 0, 1)  # 形状变成（src_len，batch，embed_dim）
        recon_v = recon_v.permute(2, 0, 1)
        recon_a = recon_a.permute(2, 0, 1)

        s_l_r = self.encoder_s_l(recon_l).permute(1, 2,
                                                  0)  # encoder_s_l是一个transformer，要求输入的形状是（src_len，batch，embed_dim），再经过permute(1, 2, 0)，形状变成：batch，embed_dim，src_len
        s_v_r = self.encoder_s_v(recon_v).permute(1, 2, 0)
        s_a_r = self.encoder_s_a(recon_a).permute(1, 2, 0)

        s_l = s_l.permute(2, 0, 1)  # 形状变成：src_len，batch，embed_dim
        s_v = s_v.permute(2, 0, 1)
        s_a = s_a.permute(2, 0, 1)

        c_l = c_l.permute(2, 0, 1)  ##形状变成：src_len，batch，embed_dim
        c_v = c_v.permute(2, 0, 1)
        c_a = c_a.permute(2, 0, 1)

        # enhancement
        hs_l_low = c_l.transpose(0, 1).contiguous().view(x_l.size(0),
                                                         -1)  # 形状变成：batch，hidden-size*seq-len，再送入proj1_l_low
        repr_l_low = self.proj1_l_low(hs_l_low)
        hs_proj_l_low = self.proj2_l_low(
            F.dropout(F.relu(repr_l_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_low += hs_l_low
        logits_l_low = self.out_layer_l_low(hs_proj_l_low)

        hs_v_low = c_v.transpose(0, 1).contiguous().view(x_v.size(0), -1)
        repr_v_low = self.proj1_v_low(hs_v_low)
        hs_proj_v_low = self.proj2_v_low(
            F.dropout(F.relu(repr_v_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_low += hs_v_low
        logits_v_low = self.out_layer_v_low(hs_proj_v_low)

        hs_a_low = c_a.transpose(0, 1).contiguous().view(x_a.size(0), -1)
        repr_a_low = self.proj1_a_low(hs_a_low)
        hs_proj_a_low = self.proj2_a_low(
            F.dropout(F.relu(repr_a_low, inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_a_low += hs_a_low
        logits_a_low = self.out_layer_a_low(hs_proj_a_low)

        c_l_att = self.self_attentions_c_l(c_l)  # 形状是src_len，batch，embed_dim，可以送入transformer
        if type(c_l_att) == tuple:  # 这里额外处理返回值是元组的情况。
            c_l_att = c_l_att[0]
        c_l_att = c_l_att[-1]

        c_v_att = self.self_attentions_c_v(c_v)
        if type(c_v_att) == tuple:
            c_v_att = c_v_att[0]
        c_v_att = c_v_att[-1]

        c_a_att = self.self_attentions_c_a(c_a)
        if type(c_a_att) == tuple:
            c_a_att = c_a_att[0]
        c_a_att = c_a_att[-1]

        c_fusion = torch.cat([c_l_att, c_v_att, c_a_att], dim=1)

        c_proj = self.proj2_c(
            F.dropout(F.relu(self.proj1_c(c_fusion), inplace=True), p=self.output_dropout,  # proj1_c是一个线性层。
                      training=self.training))
        c_proj += c_fusion  # 残差连接。
        logits_c = self.out_layer_c(c_proj)  # out_layer_c也是一个线性层。

        # LFA
        # L --> L
        h_ls = s_l
        h_ls = self.trans_l_mem(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]

        # A --> L
        h_l_with_as = self.trans_l_with_a(s_l, s_a, s_a)
        h_as = h_l_with_as
        h_as = self.trans_a_mem(h_as)
        if type(h_as) == tuple:
            h_as = h_as[0]
        last_h_a = last_hs = h_as[-1]

        # V --> L
        h_l_with_vs = self.trans_l_with_v(s_l, s_v, s_v)
        h_vs = h_l_with_vs
        h_vs = self.trans_v_mem(h_vs)
        if type(h_vs) == tuple:
            h_vs = h_vs[0]
        last_h_v = last_hs = h_vs[-1]  #

        hs_proj_l_high = self.proj2_l_high(
            F.dropout(F.relu(self.proj1_l_high(last_h_l), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_l_high += last_h_l  # 残差连接
        logits_l_high = self.out_layer_l_high(hs_proj_l_high)

        hs_proj_v_high = self.proj2_v_high(
            F.dropout(F.relu(self.proj1_v_high(last_h_v), inplace=True), p=self.output_dropout, training=self.training))
        hs_proj_v_high += last_h_v
        logits_v_high = self.out_layer_v_high(hs_proj_v_high)

        hs_proj_a_high = self.proj2_a_high(
            F.dropout(F.relu(self.proj1_a_high(last_h_a), inplace=True), p=self.output_dropout,
                      training=self.training))
        hs_proj_a_high += last_h_a
        logits_a_high = self.out_layer_a_high(hs_proj_a_high)

        # fusion
        last_h_l = torch.sigmoid(self.projector_l(hs_proj_l_high))
        last_h_v = torch.sigmoid(self.projector_v(hs_proj_v_high))
        last_h_a = torch.sigmoid(self.projector_a(hs_proj_a_high))
        c_fusion = torch.sigmoid(self.projector_c(c_fusion))

        last_hs = torch.cat([last_h_l, last_h_v, last_h_a, c_fusion], dim=1)

        # prediction
        last_hs_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_hs), inplace=True), p=self.output_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)

        res = {
            'origin_l': proj_x_l,
            'origin_v': proj_x_v,
            'origin_a': proj_x_a,
            's_l': s_l,
            's_v': s_v,
            's_a': s_a,
            'c_l': c_l,
            'c_v': c_v,
            'c_a': c_a,
            's_l_r': s_l_r,  # 重构的特征再送入语言模态的特殊编码器中去得到的特征。
            's_v_r': s_v_r,
            's_a_r': s_a_r,
            'recon_l': recon_l,
            'recon_v': recon_v,
            'recon_a': recon_a,
            'c_l_sim': c_l_sim,
            'c_v_sim': c_v_sim,
            'c_a_sim': c_a_sim,
            'logits_l_hetero': logits_l_high,  # 特殊预测
            'logits_v_hetero': logits_v_high,  # 特殊预测
            'logits_a_hetero': logits_a_high,  # 特殊预测
            'logits_c': logits_c,  # 共享预测
            'output_logit': output  # 特殊和共享特征拼接起来再预测
        }
        return res
