import math
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from tqdm import tqdm
from pTFGridNet import pTFGridNet_Small
from libs.utils import get_layer

class StreampTFGridNet_Small(nn.Module):
    def __init__(self,
                 n_fft=256,
                 n_layers=6,
                 gru_hidden_units=192,
                 attn_n_head=4,
                 attn_approx_qk_dim=516,
                 emb_dim=32,
                 emb_ks=8,
                 emb_hs=1,
                 activation="prelu",
                 eps=1.0e-5,
                 chunk_size=4,
                 left_context=60,
                 ):
        super().__init__()
        self.n_layers = n_layers
        assert n_fft % 2 == 0
        n_freqs = n_fft // 2 + 1

        # Speech Encoder
        t_ksize, f_ksize = 3, 3
        ks, padding = (t_ksize, f_ksize), (t_ksize // 2, f_ksize // 2)

        self.conv = nn.Sequential(nn.Conv2d(2, emb_dim, ks, padding=(0, f_ksize // 2)),
                                  nn.BatchNorm2d(emb_dim))
        self.t_pad = t_ksize - 1

        self.chunk_size = chunk_size

        # Speaker Encoder

        # Speaker Extractor
        self.fusion_blocks = nn.ModuleList([StreamFusionModule(emb_dim, chunk_size, left_context) for _ in range(n_layers)])
        self.separate_blocks = nn.ModuleList([StreamGridNetBlock(emb_dim,
                                                           emb_ks,
                                                           emb_hs,
                                                           n_freqs,
                                                           gru_hidden_units,
                                                           chunk_size,
                                                           left_context,
                                                           n_head=attn_n_head,
                                                           approx_qk_dim=attn_approx_qk_dim,
                                                           activation=activation,
                                                           eps=eps) for _ in range(n_layers)])

        # Speech Decoder
        self.deconv = nn.ConvTranspose2d(emb_dim, 2, ks, padding=(0, f_ksize // 2), bias=False)

    def forward(self,
                mix: torch.Tensor,
                aux: torch.Tensor,
                gru_h0,
                attn_cache_K,
                attn_cache_V,
                enc_conv_time_cache,
                dec_conv_add_cache,
                ):
        """Forward.
            Args:
                mix (torch.Tensor): batched audio tensor with N samples [B, 2, T, F]
                aux (torch.Tensor): batched emb tensor with N samples [B, emb_dim]
            Returns:
                enhanced (torch.Tensor): [B, 2, T, F] audio tensors with N samples.
        """
        # Speech Encoder
        esti_with_cache = torch.cat([enc_conv_time_cache, mix], dim=2)
        enc_conv_time_cache = mix[..., -self.t_pad:, :]
        esti = self.conv(esti_with_cache)  # [B, -1, T, F]

        # Speaker Extractor
        for i in range(self.n_layers):
            esti = self.fusion_blocks[i](aux, esti)
            esti, gru_h0[i], attn_cache_K[i], attn_cache_V[i] = self.separate_blocks[i](esti, gru_h0[i], attn_cache_K[i], attn_cache_V[i])  # [B, -1, T, F]

        # Speech Decoder
        esti = self.deconv(esti)  # [B, 2, T, F]
        cut_part_esti = esti[..., :self.chunk_size, :]
        cut_part_esti += dec_conv_add_cache
        dec_conv_add_cache[..., :self.t_pad, :] = esti[..., -self.t_pad:, :]
        esti = cut_part_esti.permute(0, 3, 2, 1)
        return esti, gru_h0, attn_cache_K, attn_cache_V, enc_conv_time_cache, dec_conv_add_cache

class StreamGridNetBlock(nn.Module):
    def __init__(self,
                 emb_dim,
                 emb_ks,
                 emb_hs,
                 n_freqs,
                 hidden_channels,
                 chunk_size,
                 left_context,
                 n_head=4,
                 approx_qk_dim=512,
                 activation='prelu',
                 eps=1e-5):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

        self.chunk_size = chunk_size
        self.left_context = left_context

        in_channels = emb_dim * emb_ks

        # Intra-Frame Full-Band Module
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_rnn = nn.GRU(in_channels,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)

        self.intra_linear = nn.ConvTranspose1d(hidden_channels * 2,
                                               emb_dim,
                                               kernel_size=emb_ks,
                                               stride=emb_hs)

        # Sub-Band Temporal Module
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_rnn = nn.GRU(emb_dim,
                                 hidden_channels,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False)
        self.inter_linear = nn.Conv1d(hidden_channels,
                                               emb_dim,
                                               kernel_size=1)

        # Cross-Frame Self-Attention Module
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)  # approx_qk_dim is only approximate
        assert emb_dim % n_head == 0

        for ii in range(n_head):
            self.add_module(f"attn_conv_Q_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_K_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module(f"attn_conv_V_{ii}",
                            nn.Sequential(nn.Conv2d(emb_dim, E, kernel_size=1),
                                          get_layer(activation)(),
                                          LayerNormalization4DCF((E, n_freqs), eps=eps)))
            self.add_module("attn_concat_proj",
                        nn.Sequential(nn.Conv2d(E*n_head, emb_dim, kernel_size=1),
                                      get_layer(activation)(),
                                      LayerNormalization4DCF((emb_dim, n_freqs), eps=eps)))


    def __getitem__(self, item):
        return getattr(self, item)

    def forward(self, x, gru_h0, attn_cache_K, attn_cache_V):
        B, C, old_T, old_F = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        F = math.ceil((old_F - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = nn.functional.pad(x, (0, F - old_F, 0, T - old_T))

        # Intra-Frame Full-Band Module
        input_ = x
        intra_rnn = self.intra_norm(input_)  # [B, C, T, F]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, F)  # [BT, C, F]
        intra_rnn = nn.functional.unfold(intra_rnn[..., None],
                                         (self.emb_ks, 1),
                                         stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, -1, C*emb_ks]
        intra_rnn = self.intra_rnn(intra_rnn)[0]  # [BT, -1, H]
        intra_rnn = intra_rnn.transpose(1, 2)  # [BT, H, -1]
        intra_rnn = self.intra_linear(intra_rnn)  # [BT, C, F]
        intra_rnn = intra_rnn.view([B, T, C, F])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()  # [B, C, T, F]
        intra_rnn = intra_rnn + input_  # [B, C, T, F]

        # # Sub-Band Temporal Module
        input_ = intra_rnn # [B, C, T, F]
        B_, C_, T_, F_ = input_.shape
        inter_rnn = self.inter_norm(input_) # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 2, 1)
        inter_rnn = inter_rnn.reshape(inter_rnn.shape[0] * inter_rnn.shape[1], inter_rnn.shape[2], inter_rnn.shape[3]) # [B * F, T, C]
        inter_rnn, gru_h0 = self.inter_rnn(inter_rnn, gru_h0) # [B * F, T, C]
        inter_rnn = inter_rnn.permute(0, 2, 1) # [B * F, C, T]
        inter_rnn = self.inter_linear(inter_rnn) # [B * F, C, T]
        inter_rnn = inter_rnn.permute(0, 2, 1) # [B * F, T, C]
        inter_rnn = inter_rnn.reshape(B_, F_, T_, C_) # (B,F,T,C)
        inter_rnn = inter_rnn.permute(0, 3, 2, 1)
        inter_rnn = inter_rnn + input_  # [B, C, T, F]

        # Cross-Frame Self-Attention Module
        inter_rnn = inter_rnn[..., :old_T, :old_F]
        batch = inter_rnn

        old_F = batch.size()[-1]

        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            all_Q.append(self[f"attn_conv_Q_{ii}"](batch))  # [B, C, T, F]
            all_K.append(self[f"attn_conv_K_{ii}"](batch))  # [B, C, T, F]
            all_V.append(self[f"attn_conv_V_{ii}"](batch))  # [B, C, T, F]

        Q = torch.cat(all_Q, dim=0)  # [B', C, T, F]
        K = torch.cat(all_K, dim=0)  # [B', C, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C, T, F]

        Q = Q.transpose(1, 2)
        old_shape = Q.shape
        Q = Q.flatten(start_dim=2)  # [B', T, C*F]
        K = K.transpose(1, 2)
        K = K.flatten(start_dim=2)  # [B', T, C*F]
        V = V.transpose(1, 2)  # [B', T, C, F]
        V = V.flatten(start_dim=2)  # [B', T, C*F]
        emb_dim = Q.shape[-1]

        K = K.transpose(1, 2)

        K_padded = torch.cat(([attn_cache_K, K]), dim=2)
        attn_cache_K = torch.cat((attn_cache_K[:, :, self.chunk_size:], K), dim=2)
        V_padded = torch.cat(([attn_cache_V, V]), dim=1)
        attn_cache_V = torch.cat((attn_cache_V[:, self.chunk_size:, :], V), dim=1)

        attn_mat = torch.matmul(Q, K_padded) / emb_dim ** 0.5  # [B', T, T]

        attn_mat = nn.functional.softmax(attn_mat, dim=2)  # [B', T, T]

        V = torch.matmul(attn_mat, V_padded)  # [B', T, C*F]

        V = V.reshape(old_shape)  # [B', T, C, F]
        V = V.transpose(1, 2)  # [B', C, T, F]
        emb_dim = V.shape[1]
        batch = V.view([self.n_head, B, emb_dim, -1, old_F])  # [H, B, C, T, F]
        batch = batch.transpose(0, 1)  # [B, H, C, T, F])
        batch = batch.contiguous().view([B, self.n_head * emb_dim, batch.shape[-2], -1])  # [B, C, T, F]
        batch = self["attn_concat_proj"](batch)  # [B, C, T, F]

        out = batch + inter_rnn

        return out, gru_h0, attn_cache_K, attn_cache_V

class StreamFusionModule(nn.Module):
    def __init__(self,
                 emb_dim, chunk_size, left_context,
                 nhead=4, dropout=0.1):
        super(StreamFusionModule, self).__init__()
        self.nhead = nhead
        self.dropout = dropout

        self.emb_dim = emb_dim
        self.attn = nn.MultiheadAttention(emb_dim,
                                          num_heads=nhead,
                                          dropout=dropout,
                                          batch_first=True)

        self.chunk_size = chunk_size
        self.left_context = left_context

        self.fusion = nn.Conv2d(emb_dim * 2, emb_dim, kernel_size=1)

    def forward(self,
                aux: torch.Tensor,
                esti: torch.Tensor) -> torch.Tensor:  # [B, C, T, F]
        B, C, T, F = esti.shape
        esti_flatten = esti.permute(0, 3, 2, 1).reshape(B * F, T, C)  # [B*F, T, C]

        aux = torch.repeat_interleave(aux, F, dim=0)  # [B*F, C]
        aux = aux.unsqueeze(1).repeat(1, T, 1)  # [B*F, T, C]
        aux_adapt = self.attn(aux, esti_flatten, esti_flatten, need_weights=False)[0]
        aux = aux + aux_adapt  # [B*F, T, C]

        aux = aux.reshape(B, F, T, C).permute(0, 3, 2, 1)
        esti = self.fusion(torch.cat((esti, aux), dim=1))  # [B, C, T, F]
        return esti


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        self.eps = eps

        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))

        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))

        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)  # [B,1,T,1]
        x_hat = ((x - mu_) / (std_ + self.eps)) * self.gamma + self.beta

        return x_hat



def _load_nnet(nnet, ckpt_path):
    cpt = torch.load(ckpt_path)
    nnet.load_state_dict(cpt["model"])
    return nnet


if __name__ == "__main__":
    device = "cpu"

    n_fft = 320
    n_layers = 2
    gru_size = 192
    attn_n_head = 8
    attn_approx_qk_dim = 1256
    chunk_size = 8
    left_context = 56

    freq = n_fft // 2 + 1
    hop_length = 160
    batch = 1

    model = pTFGridNet_Small(n_fft=n_fft,
                             n_layers=n_layers,
                             gru_hidden_units=gru_size,
                             attn_n_head=attn_n_head,
                             attn_approx_qk_dim=attn_approx_qk_dim,
                             emb_dim=64,
                             emb_ks=4,
                             emb_hs=2,
                             activation="prelu",
                             eps=1e-5,
                             chunk_size=chunk_size,
                             left_context=left_context).eval()

    stream_model = StreampTFGridNet_Small(n_fft=n_fft,
                                          n_layers=n_layers,
                                          gru_hidden_units=gru_size,
                                          attn_n_head=attn_n_head,
                                          attn_approx_qk_dim=attn_approx_qk_dim,
                                          emb_dim=64,
                                          emb_ks=4,
                                          emb_hs=2,
                                          activation="prelu",
                                          eps=1e-5,
                                          chunk_size=chunk_size,
                                          left_context=left_context).eval()

