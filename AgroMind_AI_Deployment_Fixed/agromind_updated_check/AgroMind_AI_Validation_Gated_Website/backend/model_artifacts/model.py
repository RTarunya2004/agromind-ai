"""Reconstructed AgroMind hybrid architecture used for the supplied training run.
This is a TabNet-inspired learned feature-mask module, not the original pytorch-tabnet package.
"""
import torch
import torch.nn as nn

class MaskGenerator(nn.Module):
    def __init__(self, in_dim=80, steps=3, hidden=128):
        super().__init__(); self.steps=steps
        self.shared=nn.Sequential(nn.Linear(in_dim,hidden),nn.ReLU(),nn.LayerNorm(hidden),nn.Dropout(.1))
        self.mask_heads=nn.ModuleList([nn.Linear(hidden,in_dim) for _ in range(steps)])
        self.aux=nn.Linear(in_dim,57)
    def forward(self,x):
        h=self.shared(x); masks=torch.stack([torch.softmax(head(h),dim=-1) for head in self.mask_heads],dim=1)
        return masks,self.aux((masks*x.unsqueeze(1)).mean(dim=1))
class TransformerEncoderBlock(nn.Module):
    def __init__(self,embed_dim=64,num_heads=8,ff_dim=128,dropout_rate=.1):
        super().__init__(); self.layernorm1=nn.LayerNorm(embed_dim); self.attention=nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout_rate,batch_first=True); self.layernorm2=nn.LayerNorm(embed_dim); self.ffn=nn.Sequential(nn.Linear(embed_dim,ff_dim),nn.GELU(),nn.Dropout(dropout_rate),nn.Linear(ff_dim,embed_dim)); self.dropout=nn.Dropout(dropout_rate)
    def forward(self,x):
        z=self.layernorm1(x); a,_=self.attention(z,z,z,need_weights=False); x=x+self.dropout(a); return x+self.dropout(self.ffn(self.layernorm2(x)))
class TabNetTransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.initial_projection=nn.Linear(80,64); self.transformer_blocks=nn.ModuleList([TransformerEncoderBlock() for _ in range(2)])
    def forward(self,x):
        x=self.initial_projection(x)
        for b in self.transformer_blocks: x=b(x)
        return x
class Hybrid(nn.Module):
    def __init__(self):
        super().__init__(); self.mask_generator=MaskGenerator(); self.transformer_encoder=TabNetTransformerEncoder(); self.classification_head=nn.Linear(64,57)
    def forward(self,x):
        masks,aux=self.mask_generator(x); logits=self.classification_head(self.transformer_encoder(masks).mean(1)); return logits,masks,aux
