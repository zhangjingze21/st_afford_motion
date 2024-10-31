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

class MaskTransformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                v_patch_nums=(1, 2, 4, 8, 16, 32, 49),
                nb_iter=10,
                cond_drop_prob=0.1,
                use_diff_head=False):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, v_patch_nums)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, v_patch_nums, use_diff_head=use_diff_head)
        self.block_size = block_size
        self.num_vq = num_vq
        self.v_patch_nums = v_patch_nums
        
        self.cond_drop_prob = cond_drop_prob

        self.nb_iter = nb_iter
        self.mask_idx = num_vq

        self.noise_schedule = cosine_schedule

    def get_block_size(self):
        return self.block_size

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return (1. - mask) * cond
        else:
            return cond

    def forward(self, idxs, clip_feature, mask):
        if self.training:
            clip_feature = self.mask_cond(clip_feature)
        
        feat = self.trans_base(idxs, clip_feature, mask)
        logits = self.trans_head(feat[:, 1:])
        return logits

    def forward_with_cond_scale(self, idxs, clip_feature, mask, cond_scale=4):
        logits = self.forward(idxs, clip_feature, mask)
        if cond_scale == 0:
            return logits
        # null_clip_feature = torch.zeros_like(clip_feature)
        # null_clip_feature = torch.load('empty_text_feature.pth')
        null_clip_feature = torch.zeros_like(clip_feature)
        null_logits = self.forward(idxs, null_clip_feature, mask)
        if cond_scale > 0:
            # logits = (1 + cond_scale) * F.softmax(logits, dim=-1) - cond_scale * F.softmax(null_logits, dim=-1)
            logits = logits + cond_scale * (logits - null_logits)
            
            # logits = F.softmax(logits, dim=-1) + cond_scale * (F.softmax(logits, dim=-1) - F.softmax(null_logits, dim=-1))
        return logits

    @torch.no_grad()
    @eval_decorator
    def sample(self, clip_feature, mask, if_categorial=False):
        B = clip_feature.size(0)
        L = sum(self.v_patch_nums)
        
        # init random token sequence
        xs = torch.full((B, L), self.mask_idx, dtype=torch.long, device=clip_feature.device)
        scores = torch.zeros(B, L, device=clip_feature.device)
        

        for iter in torch.linspace(0, 1, self.nb_iter, device=clip_feature.device):
            rand_mask_prob = self.noise_schedule(iter)
            
            num_token_masked = torch.round(rand_mask_prob * L).clamp(min=1)  # (b, )
            
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            xs = torch.where(is_mask, self.mask_idx, xs)
            
            logits = self.forward_with_cond_scale(xs, clip_feature, mask, cond_scale=0) # (b, L, num_vq+1)
            probs = F.softmax(logits, dim=-1)
            
            # clean low prob tokens
            probs = top_k(probs, thres=0.9, dim=-1)
            
            
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                idx = idx.squeeze(-1)
                
            scores = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)
            
            xs = torch.where(is_mask, idx, xs)
            
        return xs
    
    @torch.no_grad()
    @eval_decorator
    def motion_control(self, idxs, clip_feature, mask, if_categorial=False, num_iter=10):
        """ Generate motion sequence with given body part token sequence
        Args:
            idxs: (B, L) token sequence, fixed body part token sequence
        """
        B = clip_feature.size(0)
        L = sum(self.v_patch_nums)
        
        # init random token sequence
        xs = torch.full((B, L), self.mask_idx, dtype=torch.long, device=clip_feature.device)
        xs = torch.where(idxs != self.mask_idx, idxs, xs)
        edit_mask = (idxs == self.mask_idx)
        
        scores = torch.zeros(B, L, device=clip_feature.device)
        scores = torch.where(idxs != self.mask_idx, torch.ones_like(scores,) * 1e5, scores)

        for iter in torch.linspace(0, 1, num_iter, device=clip_feature.device):
            rand_mask_prob = self.noise_schedule(iter)
            # rand_mask_prob = torch.tensor(0.08, device=clip_feature.device)
            
            num_token_masked = torch.round(rand_mask_prob * L).clamp(min=1)  # (b, )
            
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1)) & edit_mask
            xs = torch.where(is_mask, self.mask_idx, xs)
            
            logits = self.forward_with_cond_scale(xs, clip_feature, mask, cond_scale=0) # (b, L, num_vq+1)
            probs = F.softmax(logits, dim=-1)
            
            # clean low prob tokens
            probs = top_k(probs, thres=0.9, dim=-1)
            
            
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                idx = idx.squeeze(-1)
                
            scores = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)
            
            xs = torch.where(is_mask, idx, xs)
        
        return xs
    
    @torch.no_grad()
    @eval_decorator
    def motion_editing(self, idxs, clip_feature, mask, edit_mask=None, if_categorial=False, num_iter=10):
        B, L = idxs.size()
        
        if edit_mask is not None:
            raise NotImplementedError("edit_mask is not implemented yet")
    
        # init random token sequence
        xs = torch.full((B, L), self.mask_idx, dtype=torch.long, device=clip_feature.device)
        xs = torch.where(idxs != self.mask_idx, idxs, xs)

        # re-mask
        scores = torch.zeros(B, L, device=clip_feature.device)
        for iter in torch.linspace(0.9, 1, num_iter, device=clip_feature.device):
            rand_mask_prob = self.noise_schedule(iter)
            # rand_mask_prob = torch.tensor(0.02, device=clip_feature.device)
            
            num_token_masked = torch.round(rand_mask_prob * L).clamp(min=1)  # (b, )
            
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            xs = torch.where(is_mask, self.mask_idx, xs)
            
            logits = self.forward_with_cond_scale(xs, clip_feature, mask, cond_scale=0) # (b, L, num_vq+1)
            probs = F.softmax(logits, dim=-1)
            
            # clean low prob tokens
            probs = top_k(probs, thres=0.9, dim=-1)
            
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                idx = idx.squeeze(-1)
                
            scores = probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            # scores = scores.masked_fill(~is_mask, 1e5)
            
            xs = torch.where(is_mask, idx, xs)
        
        return xs
    

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

        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        # self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate, v_patch_nums=v_patch_nums) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.lvl_emb = nn.Embedding(len(v_patch_nums), embed_dim)
        # nn.init.trunc_normal_(self.lvl_emb.weight.data, mean=0, std=0.02)
        
        self.pad_emb = nn.Embedding(1, embed_dim) # 0: padding

        self.v_patch_nums = v_patch_nums
        L = sum(self.v_patch_nums)
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.v_patch_nums)]).view(1, L, 1)
        dT = d.transpose(1, 2)
        self.register_buffer('lvl_1L', dT[:, 0].contiguous())

        self.mask_emb = nn.Embedding(2, embed_dim)

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
    
    def forward(self, idx, clip_feature, mask):
        """
        Args:
            idx: (B, T) token sequence
        """
        b, t = idx.size()

        token_embeddings = self.tok_emb(idx) # (bs, t, embed_dim)
        token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1).expand(-1, 1, -1), token_embeddings], dim=1) # (bs, t+1, embed_dim)
        
        x = self.pos_embed(token_embeddings)
        
        # add mask embedding
        mask = mask.int()
        mask = self.mask_emb(mask)
        x[:, 1:] = x[:, 1:] + mask[:, :x.shape[1]]
        
        # add lvl embedding
        lvl = self.lvl_emb(self.lvl_1L)
        x[:, 1:] = x[:, 1:] + lvl[:, :x.shape[1]]
        
        # add padding embedding on the first token
        pad = self.pad_emb(torch.zeros(b, 1).long().to(idx.device))
        x[:, 0] = x[:, 0] + pad[:, 0]
        
        
        
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