import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hvqvae.encdec import Encoder, Decoder
from models.hvqvae.fsq import STFSQ
from utils.spatial import (
    humanise_feature_extractor,
    get_humanise_motion_feature_dim_list,
    get_humanise_level_idx
)
from utils.losses import MultiScaleReConsLossWithMask 

class SpatialHumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 v_patch_nums=(16, 24, 32, 40, 49)):
        
        super().__init__()
        self.code_dim = code_dim
        self.down_t = down_t
        self.quant = args.quantizer
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        
        self.humanise_level_idx = get_humanise_level_idx(len(v_patch_nums))
        self.motion_feat_dim = get_humanise_motion_feature_dim_list(len(v_patch_nums), self.humanise_level_idx)
        self.output_emb_width = output_emb_width
        
        self.num_code = [nb_code] * len(v_patch_nums)
        self.v_patch_nums = v_patch_nums
        self.num_level = len(v_patch_nums)
        assert len(self.motion_feat_dim) == self.num_level, "motion_feat_dim should have the same length as level"
        
        ## create encoder and decoder for different scales
        ## one encoder and multi-decoder for different scales
        self.encoder = nn.ModuleList(
            [Encoder(self.motion_feat_dim[l], output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm) for l in range(self.num_level)]
        )
        self.decoder = Decoder(self.motion_feat_dim[-1], output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        
        if self.quant == "hfsq":
            self.quantizer = nn.ModuleList(
                [STFSQ(vocab_size=self.num_code[l], dim=code_dim, patch_length=self.v_patch_nums[l], sequence_length=self.v_patch_nums[-1], rratio=args.rand_ratio) for l in range(self.num_level)]
            )
        else:
            raise ValueError('Invalid quantizer')
        
        if getattr(args, 'recons_loss', None) is not None:
            self.recons_loss = MultiScaleReConsLossWithMask(recons_loss=args.recons_loss, nb_joints=self.nb_joints)
        
    @torch.no_grad()
    def encode(self, x):
        """ Encode the input x to the code index in different spatial scales
        Args:
            x: (B, T, C)
        Return:
            code_idx: (B, L) where L is the total number of code index in different spatial scales
        """
        all_codes = []
        latent_motion = 0.
        for l in range(self.num_level):
            x_encoder = self.encoder[l](humanise_feature_extractor(x, level=l, humanise_level_idx=self.humanise_level_idx).permute(0, 2, 1)) # (B, C, L)
            x_encoder = x_encoder - latent_motion
            
            if self.quant == 'hfsq':
                code_idx = self.quantizer[l].fsquantize(x_encoder) # (B, L)
            else:
                raise ValueError('Invalid quantizer')
            
            latent_motion = latent_motion + self.quantizer[l].dequantize(code_idx)
            all_codes.append(code_idx)

        code_idx = torch.cat(all_codes, dim=1)
        return code_idx

    def forward(self, x, mask):
        ## create mask on temporal
        new_mask = mask.to(torch.float32).unsqueeze(1) # (B, 1, T)
        for _ in range(self.down_t): new_mask = F.interpolate(new_mask, size=(new_mask.shape[-1] // 2), mode='nearest')

        level_loss, vq_loss, all_perplexity = 0., 0., []
        latent_motion = 0.
        for l in range(self.num_level):            
            x_in = self.encoder[l](humanise_feature_extractor(x, level=l, humanise_level_idx=self.humanise_level_idx).permute(0, 2, 1)) # (B, C, L)
            x_in = x_in - latent_motion # residual learning

            x_quantized, loss, perplexity = self.quantizer[l](x_in, new_mask)
            latent_motion = latent_motion + x_quantized
            
            # regular loss on the raw space
            pred = self.decoder(latent_motion).permute(0, 2, 1) # (B, T, C)
            level_loss += self.recons_loss(
                humanise_feature_extractor(pred, level=l, humanise_level_idx=self.humanise_level_idx),
                humanise_feature_extractor(x, level=l, humanise_level_idx=self.humanise_level_idx),
                mask
            )
            vq_loss += loss
            all_perplexity.append(perplexity)

        pred_motion = self.decoder(latent_motion).permute(0, 2, 1)
        return pred_motion, level_loss, vq_loss, all_perplexity

    @torch.no_grad()
    def forward_decoder(self, x):
        """ Decode the code index to the output motion sequence
        Args:
            x: (B, L) where L is the total number of code index in different spatial scales
        Return:
            x_out: (B, T, C) where T is the length of the motion sequence
        """
        latent_motion = 0.
        cur_L = 0
        for l in range(self.num_level):
            x_d = x[:, cur_L:cur_L + self.v_patch_nums[l]]
            cur_L += self.v_patch_nums[l]

            if self.quant == "hfsq":
                x_d = self.quantizer[l].dequantize(x_d)
            else:
                raise ValueError('Invalid quantizer')
            
            latent_motion = latent_motion + x_d
        
        x_hat = self.decoder(latent_motion).permute(0, 2, 1) # (B, T, C）
        return x_hat