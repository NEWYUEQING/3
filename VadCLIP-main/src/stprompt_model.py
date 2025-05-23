from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from clip import clip
from utils.layers import GraphConvolution, DistanceAdj

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = padding_mask.to(dtype=bool, device=x.device) if padding_mask is not None else None
        self.attn_mask = self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=padding_mask, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, padding_mask)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class TemporalAdapter(nn.Module):
    def __init__(self, width: int, sigma: float = 0.1):
        super().__init__()
        self.width = width
        self.sigma = sigma
        self.ln_1 = LayerNorm(width)
        self.ln_2 = LayerNorm(width)
        self.ffn = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width))
        ]))
        
    def build_distance_matrix(self, seq_len):
        distance_matrix = torch.zeros(seq_len, seq_len, device='cuda')
        for i in range(seq_len):
            for j in range(seq_len):
                distance_matrix[i, j] = torch.exp(-abs(i - j) / self.sigma)
        return distance_matrix
        
    def forward(self, x, seq_len):
        distance_matrix = self.build_distance_matrix(seq_len)
        attention_weights = F.softmax(distance_matrix, dim=-1)
        
        x_tm = self.ln_1(torch.matmul(attention_weights, x))
        
        x_ta = self.ln_2(self.ffn(x_tm) + x_tm)
        
        return x_ta

class STPrompt(nn.Module):
    def __init__(self,
                 num_class: int,
                 embed_dim: int,
                 visual_length: int,
                 visual_width: int,
                 visual_head: int,
                 visual_layers: int,
                 attn_window: int,
                 prompt_prefix: int,
                 prompt_postfix: int,
                 patch_size: int = 32,
                 stride: int = 32,
                 top_k: int = 16,
                 device: str = "cuda"):
        super().__init__()

        self.num_class = num_class
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        self.top_k = top_k

        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(self.attn_window)
        )
        
        self.temporal_adapter = TemporalAdapter(width=visual_width)

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        self.mlp1 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(visual_width, visual_width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(visual_width * 4, visual_width))
        ]))
        
        self.classifier = nn.Linear(visual_width, 1)

        self.clipmodel, _ = clip.load("ViT-B/16", device)
        for clip_param in self.clipmodel.parameters():
            clip_param.requires_grad = False

        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, self.embed_dim)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window):
        mask = torch.empty(self.visual_length, self.visual_length)
        mask.fill_(float('-inf'))
        for i in range(int(self.visual_length / attn_window)):
            if (i + 1) * attn_window < self.visual_length:
                mask[i * attn_window: (i + 1) * attn_window, i * attn_window: (i + 1) * attn_window] = 0
            else:
                mask[i * attn_window: self.visual_length, i * attn_window: self.visual_length] = 0

        return mask

    def adj4(self, x, seq_len):
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1)) # B*T*T
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # B*T*1
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2/(x_norm_x+1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                tmp = x2[i]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i] = adj2
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                adj2 = F.threshold(adj2, 0.7, 0)
                adj2 = soft(adj2)
                output[i, :seq_len[i], :seq_len[i]] = adj2

        return output

    def spatial_attention_aggregation(self, x_patch, seq_len):
        """空间注意力聚合机制(SA2)
        
        使用运动先验来关注潜在的空间异常位置
        
        Args:
            x_patch: 块级特征，形状为(batch_size, seq_len, H, W, D)
            seq_len: 序列长度
            
        Returns:
            x_as: 聚合后的空间特征，形状为(batch_size, seq_len, D)
        """
        batch_size, seq_len, H, W, D = x_patch.shape
        
        motion_magnitude = torch.zeros(batch_size, seq_len, H, W, device=self.device)
        
        for i in range(1, seq_len-1):
            diff = x_patch[:, i] * 2 - x_patch[:, i-1] - x_patch[:, i+1]
            diff_norm = torch.norm(diff, p=2, dim=-1)
            motion_magnitude[:, i] = diff_norm
        
        motion_magnitude[:, 0] = torch.norm(x_patch[:, 0] - x_patch[:, 1], p=2, dim=-1)
        motion_magnitude[:, -1] = torch.norm(x_patch[:, -1] - x_patch[:, -2], p=2, dim=-1)
        
        x_as = torch.zeros(batch_size, seq_len, D, device=self.device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                mo_flat = motion_magnitude[b, i].view(-1)
                topk_values, topk_indices = torch.topk(mo_flat, min(self.top_k, H*W), largest=True)
                
                attention_weights = F.softmax(topk_values, dim=0)
                
                selected_features = x_patch[b, i].view(H*W, D)[topk_indices]
                
                x_as[b, i] = torch.sum(selected_features * attention_weights.unsqueeze(1), dim=0)
        
        return x_as

    def extract_patch_features(self, images):
        """提取图像块特征
        
        使用滑动窗口方法提取图像块特征
        
        Args:
            images: 输入图像，形状为(batch_size, seq_len, C, H, W)
            
        Returns:
            patch_features: 块级特征，形状为(batch_size, seq_len, h, w, D)
        """
        
        batch_size, seq_len, D = images.shape
        
        h, w = 4, 4
        patch_features = torch.zeros(batch_size, seq_len, h, w, D, device=self.device)
        
        for b in range(batch_size):
            for i in range(seq_len):
                for hi in range(h):
                    for wi in range(w):
                        patch_features[b, i, hi, wi] = images[b, i]
        
        return patch_features

    def encode_video(self, images, padding_mask, lengths):
        images = images.to(torch.float)
        position_ids = torch.arange(self.visual_length, device=self.device)
        position_ids = position_ids.unsqueeze(0).expand(images.shape[0], -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        frame_position_embeddings = frame_position_embeddings.permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings

        x_clip, _ = self.temporal((images, None))
        x_clip = x_clip.permute(1, 0, 2)
        
        x_patch = self.extract_patch_features(x_clip)
        x_as = self.spatial_attention_aggregation(x_patch, lengths)
        
        batch_size = x_clip.shape[0]
        x_ta = torch.zeros_like(x_clip)
        
        for b in range(batch_size):
            seq_len = lengths[b] if lengths is not None else x_clip.shape[1]
            x_combined = x_clip[b, :seq_len] + x_as[b, :seq_len]
            x_ta[b, :seq_len] = self.temporal_adapter(x_combined, seq_len)
        
        adj = self.adj4(x_clip, lengths)
        disadj = self.disAdj(x_clip.shape[0], x_clip.shape[1])
        x1_h = self.gelu(self.gc1(x_clip, adj))
        x2_h = self.gelu(self.gc3(x_clip, disadj))

        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)
        x = self.linear(x)
        
        x = x + x_ta

        return x, x_patch

    def encode_textprompt(self, text):
        word_tokens = clip.tokenize(text).to(self.device)
        word_embedding = self.clipmodel.encode_token(word_tokens)
        text_embeddings = self.text_prompt_embeddings(torch.arange(77).to(self.device)).unsqueeze(0).repeat([len(text), 1, 1])
        text_tokens = torch.zeros(len(text), 77).to(self.device)

        for i in range(len(text)):
            ind = torch.argmax(word_tokens[i], -1)
            text_embeddings[i, 0] = word_embedding[i, 0]
            text_embeddings[i, self.prompt_prefix + 1: self.prompt_prefix + ind] = word_embedding[i, 1: ind]
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = word_embedding[i, ind]
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = word_tokens[i, ind]

        text_features = self.clipmodel.encode_text(text_embeddings, text_tokens)

        return text_features

    def spatial_anomaly_localization(self, x_patch, text_features, abnormal_texts, normal_texts):
        """基于LLM的文本提示的空间异常定位
        
        使用训练无关的方法进行空间异常定位
        
        Args:
            x_patch: 块级特征，形状为(batch_size, seq_len, h, w, D)
            text_features: 文本特征
            abnormal_texts: 异常描述文本列表
            normal_texts: 正常描述文本列表
            
        Returns:
            spatial_heatmap: 空间热图，形状为(batch_size, seq_len, h, w)
        """
        batch_size, seq_len, h, w, D = x_patch.shape
        
        normal_features = self.encode_textprompt(normal_texts)
        abnormal_features = self.encode_textprompt(abnormal_texts)
        
        spatial_heatmap = torch.zeros(batch_size, seq_len, h, w, device=self.device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                for i in range(h):
                    for j in range(w):
                        patch_feat = x_patch[b, t, i, j] / x_patch[b, t, i, j].norm(dim=-1, keepdim=True)
                        
                        abnormal_sim = 0
                        for af in abnormal_features:
                            af_norm = af / af.norm(dim=-1, keepdim=True)
                            sim = torch.exp((patch_feat @ af_norm) / 0.07)
                            abnormal_sim += sim
                        
                        total_sim = abnormal_sim
                        for nf in normal_features:
                            nf_norm = nf / nf.norm(dim=-1, keepdim=True)
                            sim = torch.exp((patch_feat @ nf_norm) / 0.07)
                            total_sim += sim
                        
                        spatial_heatmap[b, t, i, j] = abnormal_sim / total_sim
        
        return spatial_heatmap

    def forward(self, visual, padding_mask, text, lengths, abnormal_texts=None, normal_texts=None):
        visual_features, patch_features = self.encode_video(visual, padding_mask, lengths)
        
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)

        text_features = text_features_ori
        logits_attn = logits1.permute(0, 2, 1)
        visual_attn = logits_attn @ visual_features
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True)
        visual_attn = visual_attn.expand(visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2])
        text_features = text_features_ori.unsqueeze(0)
        text_features = text_features.expand(visual_attn.shape[0], text_features.shape[1], text_features.shape[2])
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)

        visual_features_norm = visual_features / visual_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features_norm.permute(0, 2, 1)
        logits2 = visual_features_norm @ text_features_norm.type(visual_features_norm.dtype) / 0.07
        
        spatial_heatmap = None
        if abnormal_texts is not None and normal_texts is not None:
            spatial_heatmap = self.spatial_anomaly_localization(patch_features, text_features_ori, abnormal_texts, normal_texts)

        return text_features_ori, logits1, logits2, spatial_heatmap
