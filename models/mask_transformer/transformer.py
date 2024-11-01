import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.mask_transformer.pos_encoding as pos_encoding

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs


    

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, v_patch_nums=(1, 2, 4, 8, 16, 32, 49)):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.v_patch_nums = v_patch_nums
        # L = sum(self.v_patch_nums)
        # d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.v_patch_nums)]).view(1, L, 1)
        # dT = d.transpose(1, 2)
        # mask = torch.where(d >= dT, 1., 0.).reshape(1, 1, L, L)
        # self.register_buffer('mask', mask)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4, v_patch_nums=(1, 2, 4, 8, 16, 32, 49)):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate, v_patch_nums=v_patch_nums)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                v_patch_nums=(1, 2, 4, 8, 16, 32, 49)):
        super().__init__()
        self.tok_emb = nn.Embedding(num_vq+1, embed_dim)

        # self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate, v_patch_nums=v_patch_nums) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.lvl_emb = nn.Embedding(len(v_patch_nums), embed_dim)
        # nn.init.trunc_normal_(self.lvl_emb.weight.data, mean=0, std=0.02)
        
        self.pad_emb = nn.Embedding(3, embed_dim) # 0: padding

        self.v_patch_nums = v_patch_nums
        L = sum(self.v_patch_nums)
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.v_patch_nums)]).view(1, L, 1)
        dT = d.transpose(1, 2)
        self.register_buffer('lvl_1L', dT[:, 0].contiguous())

        self.mask_emb = nn.Embedding(2, embed_dim)
        self.pc_mask_emb = nn.Embedding(2, embed_dim)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, motion_tokens, text_emb, cont_emb, motion_mask, pc_mask):
        """
        Args:
            motion_tokens: (B, T) token sequence
            text_emb: (B, 1, C) text feature
            cont_emb: (B, T', C) contact feature
        """
        b, t = motion_tokens.size()
        
        text_emb_len = text_emb.size(1)
        cont_emb_len = cont_emb.size(1)

        token_embeddings = self.tok_emb(motion_tokens) # (bs, t, embed_dim)
        token_embeddings = torch.cat([text_emb, cont_emb, token_embeddings], dim=1) # (bs, t+1, embed_dim)
        
        x = self.pos_embed(token_embeddings)
        # add mask embedding
        motion_mask = motion_mask.int()
        motion_mask = self.mask_emb(motion_mask)
        x[:, text_emb_len+cont_emb_len:] = x[:, text_emb_len+cont_emb_len:] + motion_mask
        
        # add pc mask embedding
        pc_mask = pc_mask.int()
        pc_mask = self.pc_mask_emb(pc_mask)
        x[:, text_emb_len:text_emb_len+cont_emb_len] = x[:, text_emb_len:text_emb_len+cont_emb_len] + pc_mask
        
        # add lvl embedding
        lvl = self.lvl_emb(self.lvl_1L)
        x[:, text_emb_len+cont_emb_len:] = x[:, text_emb_len+cont_emb_len:] + lvl
        
        # add padding embedding on the three different types of padding
        token_seq_len = x.size(1)
        pad_emb = torch.zeros((b, token_seq_len), dtype=torch.long, device=x.device)
        pad_emb[:, :text_emb_len] = 0
        pad_emb[:, text_emb_len:text_emb_len+cont_emb_len] = 1
        pad_emb[:, text_emb_len+cont_emb_len:] = 2
        pad_emb = self.pad_emb(pad_emb)
        x = x + pad_emb
        
        for block in self.blocks:
            x = block(x)
            
        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                v_patch_nums=(1, 2, 4, 8, 16, 32, 49),
                use_diff_head=False):
        super().__init__()

        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate,v_patch_nums) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.use_diff_head = use_diff_head
        self.v_patch_nums = v_patch_nums
        if self.use_diff_head:
            self.head = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim, bias=False),
                    nn.ReLU(),
                    nn.Linear(embed_dim, num_vq+1, bias=False)
                ) for _ in range(len(v_patch_nums))])
        else:
            self.head = nn.Linear(embed_dim, num_vq+1, bias=False)
        
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        if self.use_diff_head:
            cur_L = 0
            logits_list = []
            for i, pn in enumerate(self.v_patch_nums):
                logits_list.append(self.head[i](x[:, cur_L:cur_L+pn]))
                cur_L += pn
            logits = torch.cat(logits_list, dim=1)
        else:
            logits = self.head(x)
        
        return logits