from torch import nn
import torch
import torch.nn.functional as F
import os

class CrossAttentionFusion(nn.Module):
    def __init__(self, args, feature_dim, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.args = args
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads    
        self.feature_dim = feature_dim

        # PSSM特征投影
        self.pssm_proj = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # 交叉注意力机制
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2*d_model)
        
        # 注意力融合参数
        self.attn_drop = nn.Dropout(dropout)
        self.residual_factor = nn.Parameter(torch.tensor(0.1))
        
        # 输出融合
        self.fuse = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for module in [self.q_proj, self.kv_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x, pssm, attention_mask=None, valid_lens=None):
        """
        x: [B, L_q, D]
        pssm: [B, L_k, feat_dim]
        attention_mask: optional tensor [B, L_k] (1 for real tokens, 0 for pad)
        valid_lens: optional tensor/list of true lengths per sample
        """
        batch_size, seq_len, _ = x.size()
        pssm = pssm.float()
        pssm_feat = self.pssm_proj(pssm)  # [B, L, D]
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k, v = self.kv_proj(pssm_feat).chunk(2, dim=-1)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [B, H, Lq, Dh]
        k = k.permute(0, 2, 1, 3)  # [B, H, Lk, Dh]
        v = v.permute(0, 2, 1, 3)
        attn_score = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, H, Lq, Lk]

        # --- MASKING: do not allow attention onto padded key positions ---
        if attention_mask is not None:
            # attention_mask expected shape [B, Lk] with 1 for real tokens, 0 for pad
            # expand to [B, 1, 1, Lk] to match attn_score
            key_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # True where pad
            attn_score = attn_score.masked_fill(key_mask.to(attn_score.device), float('-1e9'))
        elif valid_lens is not None:
            # valid_lens: tensor/list shape [B]
            if not isinstance(valid_lens, torch.Tensor):
                valid_lens = torch.tensor(list(valid_lens), device=attn_score.device)
            B, H, Lq, Lk = attn_score.shape
            arange = torch.arange(Lk, device=attn_score.device).unsqueeze(0).expand(B, Lk)
            pad_mask = arange >= valid_lens.unsqueeze(1)  # [B, Lk]
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Lk]
            attn_score = attn_score.masked_fill(pad_mask, float('-1e9'))

        # Debugging: optionally print statistics once per process when DEBUG_ATTENTION=1
        if os.environ.get('DEBUG_ATTENTION') == '1' and not getattr(self, '_debugged', False):
            try:
                with torch.no_grad():
                    s = attn_score
                    print('\n[DEBUG_ATTENTION] attn_score stats: mean={:.6f} std={:.6f} min={:.6f} max={:.6f}'.format(
                          float(s.mean()), float(s.std()), float(s.min()), float(s.max())))
                    print('[DEBUG_ATTENTION] q stats: mean={:.6f} std={:.6f}'.format(float(q.mean()), float(q.std())))
                    print('[DEBUG_ATTENTION] k stats: mean={:.6f} std={:.6f}'.format(float(k.mean()), float(k.std())))
                    print('[DEBUG_ATTENTION] v stats: mean={:.6f} std={:.6f}'.format(float(v.mean()), float(v.std())))
                    print('[DEBUG_ATTENTION] pssm_feat stats: mean={:.6f} std={:.6f}'.format(float(pssm_feat.mean()), float(pssm_feat.std())))
                    if attention_mask is not None:
                        try:
                            mask = attention_mask
                            print('[DEBUG_ATTENTION] attention_mask sum per sample:', mask.sum(dim=1).cpu().tolist())
                        except Exception:
                            pass
                    # also compute softmax for a small slice to inspect distribution
                    tmp = F.softmax(attn_score, dim=-1)
                    print('[DEBUG_ATTENTION] attn_weights slice (batch0, head0, query0..2, key0..9):')
                    try:
                        slice_show = tmp[0, 0, :min(3, tmp.shape[2]), :min(10, tmp.shape[3])]
                        print(slice_show.cpu().numpy())
                    except Exception:
                        pass
                    # mark as debugged to avoid noisy repeated prints
                    self._debugged = True
            except Exception as e:
                print('[DEBUG_ATTENTION] debug print failed:', e)

        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        context = (attn_weights @ v).permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        fused = torch.cat([x, context], dim=-1)
        return self.fuse(fused) + self.residual_factor * x, attn_weights

class MS_CAM(nn.Module):
    '''
    单特征进行通道注意力加权,作用类似SE模块
    '''

    def __init__(self, channels=64, r=4):
        super(MS_CAM, self).__init__()
        inter_channels = int(channels // r)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        return x * wei

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, args, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        self.args = args
        self.linear = nn.Linear(20, 256).to(args.device)

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        residual = residual.float().to(self.args.device)
        residual = self.linear(residual)
        xa = x + residual # [batch, len, d_model]
        xa = torch.unsqueeze(xa, dim=1) # [batch, 1, len, d_model]
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        # print(xl.shape, xg.shape, xlg.shape)
        wei = torch.squeeze(self.sigmoid(xlg))
        xo = x * wei + residual * (1 - wei)
        return xo


class new_AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, args, channels=128, r=16):
        super(new_AFF, self).__init__()
        inter_channels = 16
        self.args = args
        self.linear = nn.Linear(20, 256).to(args.device)
        self.linear1 = nn.Linear(20, 18).to(args.device)
        self.linear2 = nn.Linear(256, 18).to(args.device)
        self.linear3 = nn.Linear(512, 256)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print(f"x shape {x.shape} pssm shape {residual.shape}")  # x shape torch.Size([16, 128, 256]) pssm shape torch.Size([16, 128, 20])
        residual = residual.float().to(self.args.device)
        residual = self.tanh1(self.linear(residual))
        x = self.tanh2(x)
        # x = self.linear2(x)
        xa = x + residual # [batch, len, d_model]
        # print(f'xa: {xa.shape}') # xa: torch.Size([16, 128, 256])
        # xa = torch.unsqueeze(xa, dim=1) # [batch, 1, len, d_model]
        # print(f"xa.shape {xa.shape}") # xa.shape torch.Size([16, 1, 128, 256])
        xl = self.local_att(xa)
        # exit(0)
        xg = self.global_att(xa)
        xlg = xl + xg
        # print(xl.shape, xg.shape, xlg.shape)
        wei = torch.squeeze(self.sigmoid(xlg))

        # wei = torch.full_like(wei, 0.9)
        # xo = x * wei + residual * (1 - wei)

        xo = torch.concat([x * wei, residual * (1 - wei)], dim=-1)
        xo = self.linear3(xo)

        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, args, channels=128, r=16):
        super(iAFF, self).__init__()
        inter_channels = 16
        self.args = args
        self.linear = nn.Linear(20, 256).to(args.device)
        self.linear1 = nn.Linear(20, 18).to(args.device)
        self.linear2 = nn.Linear(256, 18).to(args.device)
        self.ReLU1 = nn.ReLU().to(args.device)
        self.ReLU2 = nn.ReLU().to(args.device)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        residual = self.ReLU1(self.linear(residual.float()))
        x = self.ReLU2(x)
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo
