import torch
import torch.nn as nn
import numpy as np
import os
import glob
import math
from natsort import natsorted
from torch.nn import functional as F
from omegaconf import DictConfig
from utils.training import load_ckpt
from torch.distributions import Categorical

from models.base import Model
from models.modules import SceneMapEncoderDecoder, SceneMapEncoder
from models.functions import (
    load_and_freeze_clip_model,
    encode_text_clip,
    load_and_freeze_bert_model,
    encode_text_bert,
    get_lang_feat_dim_type
)
    
from utils.misc import compute_repr_dimesion

from models.mask_transformer.transformer import CrossCondTransBase, CrossCondTransHead

from models.hvqvae.st_hvqvae import SpatialHumanVQVAE

from utils.losses import cal_loss, cal_performance

def compute_mask_from_length(m_length, max_motion_length=196, unit_length=4, v_patch_nums=(1,2,4,8,16,32,49)):
    """
    Args:
        m_length: motion length, (bs, )
    """
    bs = m_length.shape[0]
    mask = np.zeros((bs, max_motion_length,), dtype=bool)
    for i in range(bs):
        mask[i, m_length[i]:] = True

    down_t = int(np.sqrt(unit_length))
    mask = torch.from_numpy(mask).float().cuda().unsqueeze(0)# (1, 1, max_motion_length)
    for _ in range(down_t): # downsample the mask
        mask = F.interpolate(mask, size=(mask.shape[-1] // 2), mode='nearest')
    
    mask_list = []
    for si, pn in enumerate(reversed(v_patch_nums)):
        mask_list.insert(0, F.interpolate(mask, size=(pn), mode='nearest'))
    mask = torch.cat(mask_list, dim=-1).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
    return mask

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask

def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)

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
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

@Model.register()
class MASK_TRANS(nn.Module):
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        
        self.nb_code = cfg.sthvqvae.nb_code
        self.mask_idx = self.nb_code
        
        # load the pretrained hvqvae model and freeze it
        self.hvqvae = SpatialHumanVQVAE(cfg.sthvqvae).to(self.device)
        ckpt = natsorted(glob.glob(os.path.join(cfg.sthvqvae.resume_pth, 'ckpt', 'model*.pt')))
        assert len(ckpt) > 0, 'No checkpoint found.'
        load_ckpt(self.hvqvae, ckpt[-1])
        self.hvqvae.eval()
        for param in self.hvqvae.parameters():
            param.requires_grad = False
        
        self.v_patch_nums = cfg.sthvqvae.v_patch_nums
        self.unit_length = cfg.sthvqvae.down_t ** 2
        self.max_motion_length = cfg.sthvqvae.v_patch_nums[-1] * self.unit_length
        self.motion_seq_len = sum(self.v_patch_nums)
        
        
        self.motion_type = cfg.data_repr
        self.motion_dim = cfg.input_feats
        self.latent_dim = cfg.transformer.embed_dim
        
        self.arch = cfg.arch
        
        ## contact
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)
        self.planes = cfg.contact_model.planes
        if self.arch == 'trans_enc':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_dec':
            SceneMapModule = SceneMapEncoderDecoder
        else:
            raise NotImplementedError
        self.contact_encoder = SceneMapModule(
            point_feat_dim=self.contact_dim,
            planes=self.planes,
            blocks=cfg.contact_model.blocks,
            num_points=cfg.contact_model.num_points,
        )
        
        ## text
        self.text_model_name = cfg.text_model.version
        self.text_max_length = cfg.text_model.max_length
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)
        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError
        self.language_adapter = nn.Linear(self.text_feat_dim, self.latent_dim, bias=True)
        
        ## model architecture (mask modeling)
        self.trans_base = CrossCondTransBase(
            num_vq = cfg.sthvqvae.nb_code,
            embed_dim = cfg.transformer.embed_dim,
            clip_dim = cfg.transformer.clip_dim,
            block_size = cfg.transformer.block_size,
            num_layers = cfg.transformer.num_layers,
            n_head  = cfg.transformer.n_head,
            drop_out_rate = cfg.transformer.drop_out_rate,
            fc_rate = cfg.transformer.fc_rate,
            v_patch_nums = cfg.sthvqvae.v_patch_nums,
        )
        self.trans_head = CrossCondTransHead(
            num_vq = cfg.sthvqvae.nb_code,
            embed_dim = cfg.transformer.embed_dim,
            block_size = cfg.transformer.block_size,
            num_layers = cfg.transformer.num_layers,
            n_head = cfg.transformer.n_head,
            drop_out_rate = cfg.transformer.drop_out_rate,
            fc_rate = cfg.transformer.fc_rate,
            v_patch_nums = cfg.sthvqvae.v_patch_nums,
            use_diff_head = cfg.transformer.use_diff_head,
        )
        
    def forward(self, x, **kwargs):
        """ Forward pass of the model
        
        Args: 
            x: input motion, [bs, seq_len, motion_dim]
        
        Returns:
            pred_motion: predicted motion, [bs, seq_len, motion_dim]
            loss: cross entropy loss
        
        """
        
        ## text embedding
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float() # (bs, 1, embdim=512)
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device) # (bs, 1)
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool)) # 0 for valid, 1 for invalid
        else:
            raise NotImplementedError
        
        text_emb = self.language_adapter(text_emb) # (bs, 1, latent_dim)
        
        
        ## encode contact
        cont_emb = self.contact_encoder(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'): # trans_enc
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb) # [bs, num_groups, latent_dim], for trans_enc
            
        
        if 'x_token' in kwargs:
            motion_tokens = kwargs['x_token'] # (bs, ntokens=138)   
        else:
            motion_tokens = self.hvqvae.encode(x) # (bs, ntokens=138)
        # prepare mask
        bs = motion_tokens.shape[0]
        ntokens = motion_tokens.shape[1]
        rand_time = uniform((bs, ), device=self.device)
        rand_mask_probs = cosine_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
        batch_randperm = torch.rand((bs, ntokens), device=self.device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        
        # note this is our training target, not input
        target = torch.where(mask, motion_tokens, self.nb_code)
        x_idxs = motion_tokens.long().to(self.device).clone()

        # further apply bert masking scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        rand_id = torch.randint_like(x_idxs, self.nb_code)
        x_idxs = torch.where(mask_rid, rand_id, x_idxs)
        
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_idxs = torch.where(mask_mid, self.nb_code, x_idxs)
        
        motion_mask = compute_mask_from_length(m_length=self.max_motion_length-kwargs['x_mask'].sum(dim=-1).cpu().numpy(), max_motion_length=self.max_motion_length, unit_length=self.unit_length, v_patch_nums=self.v_patch_nums)
        motion_mask = torch.from_numpy(motion_mask).to(self.device)
        feat = self.trans_base(x_idxs, text_emb, cont_emb, motion_mask=motion_mask, pc_mask=cont_mask)
    
        feat = feat[:, -self.motion_seq_len:]
        
        logits = self.trans_head(feat)
        loss, pred_id, acc = cal_performance(logits.permute(0, 2, 1), target, ignore_index=self.nb_code)
        
        # pred_motion = self.hvqvae.forward_decoder(pred_motion_tokens)
        if self.training:
            return loss, acc
        else:
            return logits
    
    @torch.no_grad()
    def sample(self, x, if_categorial=False, num_iter=10, **kwargs):
        B = x.shape[0]
        L = sum(self.v_patch_nums)
        
        # init random token sequence
        xs = torch.full((B, L), self.mask_idx, dtype=torch.long, device=self.device)
        scores = torch.zeros(B, L, device=self.device)
        
        ## text embedding
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float() # (bs, 1, embdim=512)
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device) # (bs, 1)
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'], max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool)) # 0 for valid, 1 for invalid
        else:
            raise NotImplementedError
        
        text_emb = self.language_adapter(text_emb) # (bs, 1, latent_dim)
        
        ## encode contact
        cont_emb = self.contact_encoder(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'): # trans_enc
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb) # [bs, num_groups, latent_dim], for trans_enc
            
        motion_mask = compute_mask_from_length(m_length=self.max_motion_length-kwargs['x_mask'].sum(dim=-1).cpu().numpy(), max_motion_length=self.max_motion_length, unit_length=self.unit_length, v_patch_nums=self.v_patch_nums)
        motion_mask = torch.from_numpy(motion_mask).to(self.device)

        for iter in torch.linspace(0, 1, num_iter, device=self.device):
            rand_mask_prob = cosine_schedule(iter)
            
            num_token_masked = torch.round(rand_mask_prob * L).clamp(min=1)  # (b, )
            
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            xs = torch.where(is_mask, self.mask_idx, xs)
            
            feat = self.trans_base(xs, text_emb, cont_emb, motion_mask=motion_mask, pc_mask=cont_mask)
            feat = feat[:, -self.motion_seq_len:]
            logits = self.trans_head(feat)
            
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
            
        pred_motion = self.hvqvae.forward_decoder(xs)
        return pred_motion