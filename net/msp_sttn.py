import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
from util.Patching import patching_method
from util.Pos_embedding import LocPositionalEncoder


class Conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, residual=True, bn=True,
                 activation='LeakyReLU'):

        super().__init__()
        self.is_bn = bn
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        if activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()

    def forward(self, x):

        res = self.residual(x)

        x = self.conv(x)

        if self.is_bn:
            x = self.bn(x)

        x = x + res
        x = self.activation(x)

        return x


class Attention(nn.Module):

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m:
            scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        p_attn.detach().cpu().numpy()
        return p_val, p_attn


class Patch_Transformer(nn.Module):

    def __init__(self, length, encoding_w, encoding_h, encoding_dim, patch_size_w, patch_size_h, sub_embedding_dim,
                 is_mask=0, PATCH_METHOD='UNFOLD'):

        super().__init__()

        self.is_mask = is_mask
        self.length = length
        self.patch_method = PATCH_METHOD

        self.encoding_w = encoding_w  # 32
        self.encoding_h = encoding_h  # 32

        self.patch_size_w = patch_size_w  # 2
        self.patch_size_h = patch_size_h  # 16

        self.patch_num_w = self.encoding_w // self.patch_size_w  # 16
        self.patch_num_h = self.encoding_h // self.patch_size_h  # 2

        # 1D vector
        mid_dim = sub_embedding_dim * self.patch_size_w * self.patch_size_h  # 256*2*16

        self.embedding_Q = nn.Conv2d(in_channels=encoding_dim, out_channels=sub_embedding_dim, kernel_size=1)
        self.embedding_K = nn.Conv2d(in_channels=encoding_dim, out_channels=sub_embedding_dim, kernel_size=1)
        self.embedding_V = nn.Conv2d(in_channels=encoding_dim, out_channels=sub_embedding_dim, kernel_size=1)

        if is_mask:
            self.multihead_attn = Attention()
        else:
            self.multihead_attn = nn.MultiheadAttention(mid_dim, num_heads=1)

    def forward(self, c, q, mask):

        # [12, 768, 32, 32]
        B_T, C, W, H = c.shape
        T = self.length
        B = B_T // T

        encoding_w = self.encoding_w  # 32
        encoding_h = self.encoding_h  # 32

        Q = self.embedding_Q(q)
        K = self.embedding_K(c)  # (24, 32, 32, 32)
        V = self.embedding_V(c)

        # B,C//num
        C = Q.shape[1]
        #Q, K, V = patching_method(Q, K, V, B, C, self.patch_num_h, self.patch_num_w, self.patch_size_h,
                                  #self.patch_size_w, self.patch_method)
        Q, K, V = patching_method(Q, K, V, B, T, C, self.patch_num_h, self.patch_num_w, self.patch_size_h,
                                  self.patch_size_w, self.patch_method)

        if self.is_mask:
            attn_output, atten_output_weight = self.multihead_attn(Q, K, V, None)
            x = attn_output
        else:
            Q = Q.permute(1, 0, 2)
            K = K.permute(1, 0, 2)
            V = V.permute(1, 0, 2)
            attn_output, atten_output_weight = self.multihead_attn(Q, K, V)
            x = attn_output.permute(1, 0, 2)

        x = x.reshape(B_T, -1, encoding_h, encoding_w)

        return x,atten_output_weight


class Encoder(nn.Module):

    def __init__(self, input_channels, encoding_dim):
        super().__init__()

        CONV = Conv_block


        self.conv1 = CONV(input_channels, encoding_dim // 4, kernel_size=3, stride=1, dilation=1,
                                residual=False)
        self.conv2 = CONV(encoding_dim // 4, encoding_dim // 4, kernel_size=3, stride=1, dilation=1,
                                residual=True)
        self.conv3 = CONV(encoding_dim // 4, encoding_dim // 2, kernel_size=3, stride=1, dilation=1,
                                residual=True)
        self.conv4 = CONV(encoding_dim // 2, encoding_dim, kernel_size=3, stride=1, dilation=1,
                                residual=True)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c4,c3,c2,c1


class Decoder(nn.Module):

    def __init__(self,Length, output_channels, encoding_dim, Using_skip, Activation,Cat_style,only_conv6=0):

        super().__init__()

        self.Using_skip = Using_skip
        self.only_conv6 = only_conv6

        CONV = Conv_block

        if Cat_style == 'cat_trans':
            if only_conv6:
                self.conv5 = CONV(encoding_dim, encoding_dim // 2, kernel_size=3, stride=1, dilation=1)
                self.conv6 = CONV(encoding_dim // 2 if Using_skip else encoding_dim // 4,
                                        encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
                self.conv7 = CONV(encoding_dim // 2 if Using_skip else encoding_dim // 8,
                                        encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
                self.conv8 = CONV(encoding_dim // 4 if Using_skip else encoding_dim // 8,
                                        output_channels, kernel_size=3, stride=1, dilation=1,
                                        activation=Activation)


            else:
                self.conv5 = CONV(encoding_dim, encoding_dim // 2, kernel_size=3, stride=1, dilation=1)
                self.conv6 = CONV(encoding_dim // 1 if Using_skip else encoding_dim // 4,
                                        encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
                self.conv7 = CONV(encoding_dim // 2 if Using_skip else encoding_dim // 8,
                                        encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
                self.conv8 = CONV(encoding_dim // 2 if Using_skip else encoding_dim // 8,
                                        output_channels, kernel_size=3, stride=1, dilation=1,
                                        activation=Activation)
        elif Cat_style == 'cat':

            if only_conv6:
                self.conv5 = CONV(encoding_dim, encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
                self.conv6 = CONV(encoding_dim // 4 if Using_skip else encoding_dim // 4,
                                        encoding_dim // 8, kernel_size=3, stride=1, dilation=1)
                self.conv7 = CONV(encoding_dim // 4 if Using_skip else encoding_dim // 8,
                                        encoding_dim // 8, kernel_size=3, stride=1, dilation=1)
                self.conv8 = CONV(encoding_dim // 8 if Using_skip else encoding_dim // 8,
                                        output_channels, kernel_size=3, stride=1, dilation=1,
                                        activation=Activation)
            else:

                self.conv5 = CONV(encoding_dim, encoding_dim // 4, kernel_size=3, stride=1, dilation=1)
                self.conv6 = CONV(encoding_dim // 2 if Using_skip else encoding_dim // 4,
                                        encoding_dim // 8, kernel_size=3, stride=1, dilation=1)
                self.conv7 = CONV(encoding_dim // 4 if Using_skip else encoding_dim // 8,
                                        encoding_dim // 8, kernel_size=3, stride=1, dilation=1)
                self.conv8 = CONV(encoding_dim // 4 if Using_skip else encoding_dim // 8,
                                        output_channels, kernel_size=3, stride=1, dilation=1,
                                        activation=Activation)




    def forward(self, inp):

        c4, c3,c2,c1 = inp

        c5 = self.conv5(c4)

        if self.Using_skip:
            if self.only_conv6:
                pass
            else:
                c5 = torch.cat([c5, c3], dim=-3)

        c6 = self.conv6(c5)

        if self.Using_skip:
            c6 = torch.cat([c6, c2], dim=-3)

        c7 = self.conv7(c6)

        if self.Using_skip:
            if self.only_conv6:
                pass
            else:
                c7 = torch.cat([c7, c1], dim=-3)

        c8 = self.conv8(c7)

        return c8


class Multi_patch_transfomer(nn.Module):

    def __init__(self, Patch_list, length, cnn_encoding_w, cnn_encoding_h, cnn_encoding_dim, cnn_embedding_dim,
                 dropout, patch_method, residual=1, is_mask=0, norm_type='LN'):

        super().__init__()

        self.scale_num = len(Patch_list)
        self.patch_method = patch_method
        self.multi_patch_transformer = nn.ModuleList()

        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = lambda x: x

        if norm_type == 'BN':
            self.norm = nn.BatchNorm2d(cnn_embedding_dim)
        else:
            self.norm = nn.LayerNorm(cnn_encoding_h, cnn_encoding_w)

        self.dropout = nn.Dropout(dropout)

        sub_dim = cnn_embedding_dim // self.scale_num

        for i in range(self.scale_num):
            patch_size_w = Patch_list[i][0]
            patch_size_h = Patch_list[i][1]

            patch_transformer = Patch_Transformer(
                length=length,
                encoding_w=cnn_encoding_w,
                encoding_h=cnn_encoding_h,
                encoding_dim=cnn_encoding_dim,
                patch_size_w=patch_size_w,
                patch_size_h=patch_size_h,
                sub_embedding_dim=sub_dim,
                is_mask=is_mask,
                PATCH_METHOD=patch_method,
            )

            self.multi_patch_transformer.append(patch_transformer)

        self.ffn = Conv_block(cnn_embedding_dim, cnn_embedding_dim, kernel_size=3, stride=1, dilation=1, residual=True)



    def forward(self, c, q, mask):


        x = q
        v,att_list = self.multi_patch_forward(c, q, mask)
        att = self.residual(x) + self.dropout(v)
        att = self.norm(att)

        ff = self.residual(att) + self.dropout(self.ffn(att))
        ff = self.norm(ff)

        return ff,att_list

    def multi_patch_forward(self, c, q, mask):
        output = []
        att_list = []
        for i in range(self.scale_num):
            z,att_map = self.multi_patch_transformer[i](c, q, mask)
            output.append(z)
            att_list.append(att_map)
        output = torch.cat(output, 1)  # (6,256,50,50)

        return output,att_list


class Prediction_Model(nn.Module):

    def __init__(self, mcof, Length, Width, Height, Input_dim, Patch_list, Encoding_dim, Embedding_dim,
                 Dropout=0.2, Att_num=1, Cross_att_num=1, Using_skip=0,  Is_mask=0, residual=1,
                 Norm_type='LN',**arg):

        super().__init__()

        self.cross_att_num = Cross_att_num

        self.cat_style = arg['Cat_style']
        self.is_aux = arg['Is_aux']
        self.only_conv6 = arg['ONLY_CONV6']
        self.trans_residual = arg['TRANS_RESIDUAL']
        #self.one_hot = arg['ONE_HOT']

        self.mcof = mcof
        self.patch_method = mcof.patch_method

        self.input_channels = Input_dim
        self.output_channels = Input_dim


        encoding_w = Width
        encoding_h = Height

        encoding_dim = Encoding_dim  # 256
        embedding_dim = Embedding_dim  # 256

        if self.cat_style == 'cat':
            self.len_dim = 64
            self.spa_dim = (2*encoding_dim - self.len_dim)//2
        elif self.cat_style == 'cat_trans':
            self.len_dim = 64
            self.spa_dim = (encoding_dim - self.len_dim)//2

        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = lambda x: x

        self.loc_pos_enc = LocPositionalEncoder(self.len_dim, 0.3)  # (32,*,128)
        self.spa_pos_enc = LocPositionalEncoder(self.spa_dim, 0.3)

        self.norm_bn = nn.BatchNorm2d(Input_dim)
        self.norm_ln = nn.LayerNorm(encoding_h, encoding_w)


        self.encoder = Encoder(self.input_channels, encoding_dim)
        self.encoder_c = Encoder(self.input_channels, encoding_dim)
        if self.cross_att_num:
            self.encoder_q = Encoder(self.input_channels, encoding_dim)

        self.trans = Conv_block(encoding_dim*2, encoding_dim, kernel_size=1, stride=1, dilation=1, residual=self.trans_residual)

        if self.mcof.pos_en:
            if self.mcof.pos_en_mode == 'cat':
                if self.cat_style == 'cat':
                    tr_encoding_dim = encoding_dim * 2
                    tr_embedding_dim = embedding_dim * 2
                elif self.cat_style == 'cat_trans':
                    tr_encoding_dim = encoding_dim
                    tr_embedding_dim = embedding_dim
            else:
                tr_encoding_dim = encoding_dim
                tr_embedding_dim = embedding_dim
        else:
            tr_encoding_dim = encoding_dim
            tr_embedding_dim = embedding_dim


        if self.cross_att_num:
            self.tran0 = Conv_block(encoding_dim, tr_encoding_dim, kernel_size=1, stride=1,dilation=1, residual=self.trans_residual)

        self.decoder = Decoder(Length,self.output_channels, tr_encoding_dim, Using_skip, 'Tanh',self.cat_style,self.only_conv6)

        if 1:
            self.attention_c = nn.ModuleList()
            for a in range(Att_num):
                self.attention_c.append(
                    Multi_patch_transfomer(Patch_list, Length, encoding_w, encoding_h, tr_encoding_dim,
                                        tr_embedding_dim, Dropout, self.patch_method, residual,
                                        is_mask=Is_mask, norm_type=Norm_type))


        if self.cross_att_num:
            self.attention_cr = nn.ModuleList()
            for a in range(Cross_att_num):
                self.attention_cr.append(
                    Multi_patch_transfomer(Patch_list, Length, encoding_w, encoding_h, tr_encoding_dim,
                                        tr_embedding_dim, Dropout, self.patch_method, residual,
                                        is_mask=Is_mask, norm_type=Norm_type))


        self.dropout = nn.Dropout(p=Dropout)

        self.linear_tim = nn.Sequential(
            nn.Conv2d(embedding_dim, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.feedforward_tim = nn.Sequential(
            nn.Linear(Length * 2 * 32 * 32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout),
            nn.Linear(64, 48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout))

        self.linear_typ = nn.Sequential(
            nn.Conv2d(embedding_dim, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.feedforward_typ = nn.Sequential(
            nn.Linear(Length * 2 * 32 * 32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout),
            nn.Linear(64, 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout))

        self.linear_tim_aux = nn.Sequential(
            nn.Conv2d(tr_embedding_dim, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.feedforward_tim_aux = nn.Sequential(
            nn.Linear(Length * 2 * 32 * 32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout),
            nn.Linear(64, 48),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout))

        self.linear_typ_aux = nn.Sequential(
            nn.Conv2d(tr_embedding_dim, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.feedforward_typ_aux = nn.Sequential(
            nn.Linear(Length * 2 * 32 * 32, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout),
            nn.Linear(64, 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=Dropout))


        self.conv3d = nn.Conv3d(6,1,(1,1,1),stride=1,padding=(0,0,0))
        self.conv3d_act = nn.Tanh()

        self.lin_out = nn.Linear(6,1)

    def forward(self, avg, que, con):

        # B,T,C,H,W -> BT,C,H,W
        B, T, C, H, W = con.shape  # (6, 6, 2, 32, 32)

        if self.cross_att_num:
            x_q = que.reshape(-1, self.input_channels, H, W)  # (B*T, 2, 32, 32)


        a = avg.reshape(-1, self.input_channels, H, W)  # (B*T, 2, 32, 32)
        c = con.reshape(-1, self.input_channels, H, W)
        if self.cross_att_num:
            x_q = x_q.reshape(-1, self.input_channels, H, W)

        a = self.norm_bn(a)
        c = self.norm_bn(c)
        if self.cross_att_num:
            x_q = self.norm_bn(x_q)


        enc, c3, c2, c1 = self.encoder(a)  # (BT, 256, 32, 32)
        enc_c, c3_c, c2_c, c1_c = self.encoder_c(c)


        tim_cls_out = self.tim_class_pred(enc, avg)
        typ_cls_out = self.typ_class_pred(enc, avg)

        if self.cross_att_num:
            enc_q, c3_q, c2_q, c1_q = self.encoder_q(x_q)  # (BT, 256, 32, 32)


        if self.mcof.pos_en:
            seq_pos, spa_pos = self.pos_embedding(avg)

            att_c = torch.cat((enc_c, enc), dim=1)
            pos = torch.cat((spa_pos, seq_pos), dim=1)

            if self.cat_style == 'cat_trans':
                att_c = self.trans(att_c)

            att_c = att_c + pos

        else:
            att_c = enc + enc_c * 0.1

        att_map_layer_list = []
        if 1:
            att = att_c
            for att_layer in self.attention_c:
                att,att_map_list = att_layer(att, att, None)
                att = att + pos
                att_map_layer_list.append(att_map_list)

        if self.cross_att_num:
            if self.cat_style == 'cat':
                att_q = self.tran0(enc_q)
            else:
                att_q = enc_q
            #att_q = enc_q
            for att_layer in self.attention_cr:
                att,att_map_list = att_layer(att, att_q, None)
                att = att + pos
                att_map_layer_list.append(att_map_list)

        if self.is_aux:
            tim_cls_out = tim_cls_out + self.tim_class_pred_aux(att, avg)
            typ_cls_out = typ_cls_out + self.typ_class_pred_aux(att, avg)

        dec = self.decoder([att, c3,c2,c1])

        out = dec.reshape(B, -1, self.output_channels, H, W)


        out = out + avg

        return out, tim_cls_out, typ_cls_out,att_map_layer_list

    def pos_embedding(self, inp):

        B, T, C, H, W = inp.shape
        # (1,T,32) # [B, T, 32, 32, 32]
        pos_t = self.loc_pos_enc(T).permute(1, 2, 0).unsqueeze(-1).type_as(inp)
        #pos_t = pos_t.repeat(B, 1, 1, H, W).reshape(B * T, int(64//self.dim_factor), H, W)
        pos_t = pos_t.repeat(B, 1, 1, H, W).reshape(B * T, self.len_dim, H, W)

        # H位置
        # (1,H,112)->(112,H,1)  # [B, T, 112, 32, 32]
        spa_h = self.spa_pos_enc(H).permute(2, 1, 0).type_as(inp)
        #spa_h = spa_h.repeat(B, T, 1, 1, W).reshape(B * T, int(224//self.dim_factor), H, W)
        spa_h = spa_h.repeat(B, T, 1, 1, W).reshape(B * T, self.spa_dim, H, W)

        # W位置
        # (1,W,112)->(112,1,W)  # [B, T, 112, 32, 32]
        spa_w = self.spa_pos_enc(W).permute(2, 0, 1).type_as(inp)
        #spa_w = spa_w.repeat(B, T, 1, H, 1).reshape(B * T, int(224//self.dim_factor), H, W)
        spa_w = spa_w.repeat(B, T, 1, H, 1).reshape(B * T, self.spa_dim, H, W)

        spa = torch.cat([spa_h, spa_w], dim=1)

        return pos_t, spa

    def tim_class_pred(self, enc, inp):
        B, T, C, H, W = inp.shape


        enc = self.linear_tim(enc)
        enc = enc.reshape(B, T, C, H, W)

        enc = enc.reshape(B, -1)
        enc = self.feedforward_tim(enc)

        #cls_out = self.softmax_tim(enc)

        return enc

    def typ_class_pred(self, enc, inp):
        B, T, C, H, W = inp.shape

        enc = self.linear_typ(enc)
        enc = enc.reshape(B, T, C, H, W)

        enc = enc.reshape(B, -1)
        enc = self.feedforward_typ(enc)

        #typ_out = self.softmax_typ(enc)

        return enc

    def tim_class_pred_aux(self, enc, inp):
        B, T, C, H, W = inp.shape

        enc = self.linear_tim_aux(enc)
        enc = enc.reshape(B, T, C, H, W)

        enc = enc.reshape(B, -1)
        enc = self.feedforward_tim_aux(enc)

        #cls_out = self.softmax_tim(enc)

        return enc

    def typ_class_pred_aux(self, enc, inp):
        B, T, C, H, W = inp.shape

        enc = self.linear_typ_aux(enc)
        enc = enc.reshape(B, T, C, H, W)

        enc = enc.reshape(B, -1)
        enc = self.feedforward_typ_aux(enc)

        #typ_out = self.softmax_typ(enc)

        return enc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass in some training parameters')
    parser.add_argument('--mode', type=str, default='train', help='The processing phase of the model')
    parser.add_argument('--record', type=str, help='Recode ID')
    parser.add_argument('--task', type=str, default='B', help='Processing task type')
    parser.add_argument('--keep_train', type=int, default=0, help='Model keep training')
    parser.add_argument('--epoch_s', type=int, default=0, help='Continue training on the previous model')
    parser.add_argument('--inp_type', type=str, default='external',
                        choices=['external', 'accumulate', 'accumulate_avg', 'train', 'holiday', 'windspeed', 'weather',
                                 'temperature'])
    parser.add_argument('--patch_method', type=str, default='STTN', choices=['EINOPS', 'UNFOLD', 'STTN'])

    parser.add_argument('--debug', type=int, default=0, help='Model debug')
    parser.add_argument('--pos_en_mode', type=str, default='cat', help='positional encoding mode')
    mcof = parser.parse_args()

    PATCH_LIST = [[4,4]]
    net = Prediction_Model(
        mcof=mcof,
        Length=6,  # 8
        Width=32,  # 200
        Height=32,  # 200
        Input_dim=2,  # 1
        Patch_list=PATCH_LIST,  # 小片段的大小
        Att_num=2,  # 2
        Cross_att_num=0,  # 2
        Using_skip=1,  # 1
        Encoding_dim=256,  # 256
        Embedding_dim=256,  # 256
        Is_mask=1,  # 1
        TRANS_RESIDUAL=1,
        Norm_type='LN',
        Cat_style = 'cat',
        Is_aux=1,
        ONE_HOT=0,
        ONLY_CONV6=1,
    )


    from torchsummaryX import summary


    input_c = torch.randn(4, 6, 2, 32, 32)
    context_c = torch.randn(4, 6, 2, 32, 32)

    input_q = torch.randn(4, 6, 2, 32, 32)
    context_q = torch.randn(4, 6, 2, 32, 32)


    out, tim_cls_out, typ_cls_out,att_map = net(input_c, context_c, context_q)
    print('=============')
    print(out.shape, tim_cls_out.shape, typ_cls_out.shape)
