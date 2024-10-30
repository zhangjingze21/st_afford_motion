import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import List, Tuple

from models.hvqvae.resnet import Resnet1D

def vocab_size_to_levels(vocab_size: int) -> List[int]:
    """Converts a vocab size to a list of levels."""
    if vocab_size == 75:
        return [3, 5, 5]
    elif vocab_size == 125:
        return [5, 5, 5]
    elif vocab_size == 240:
        return [8, 6, 5]
    elif vocab_size == 320:
        return [8, 8, 5]
    elif vocab_size == 384:
        return [8, 8, 6]
    elif vocab_size == 500:
        return [4, 5, 5, 5]
    elif vocab_size == 625:
        return [5, 5, 5, 5]
    elif vocab_size == 1000:
        return [8, 5, 5, 5]
    else:
        raise ValueError('Invalid vocab size, still not implemented')

def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

class STFSQ(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        patch_length: int,
        sequence_length: int,
        **kwargs
    ):
        super().__init__()
        self.patch_length = patch_length # current patch length
        self.L = sequence_length # original sequence length

        levels = vocab_size_to_levels(vocab_size)
        self.levels = levels
        
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent = False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        self.dim = dim
        self.rratio = kwargs.get('rratio', 0.0)

        has_projections = self.dim != codebook_dim
        self.quant_resi = Phi(dim, 1)
        self.project_in = ProjectIn(dim, codebook_dim) if has_projections else nn.Identity()
        self.project_out = ProjectOut(dim, codebook_dim) if has_projections else nn.Identity()
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width
    
    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int64)
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, '... -> ... 1') # (..., 1)
        codes_non_centered = (indices // self._basis) % self._levels # (..., codebook_dim)
        return self._scale_and_shift_inverse(codes_non_centered)

    def indices_to_codecount(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the count of each code in the batch."""
        indices = indices.view(-1)

        code_onehot = torch.zeros(self.codebook_size, indices.shape[0], device=indices.device)  # (vocab_size, N * L)
        code_onehot.scatter_(0, indices.view(1, -1), 1) # (vocab_size, N * L)
        code_onehot = code_onehot * (1.0 - mask.view(1, -1))  # (vocab_size, N * L), mask invalid code
        
        code_count = code_onehot.sum(dim=-1) # (vocab_size,)
        return code_count

    def forward(self, f_BCL: torch.Tensor, m_B1L: torch.Tensor):
        new_mask = F.interpolate(m_B1L, size=(self.patch_length,), mode='nearest')
        
        f_nograd = f_BCL.detach()
        B, C, L = f_BCL.shape
        assert L == self.L, f"Input sequence length {L} does not match the model sequence length {self.L}"

        code_count = 0.
        mean_vq_loss = 0.
        
        ## interpolate to current patch length
        f_rest = F.interpolate(f_BCL, size=(self.patch_length,), mode='area').permute(0, 2, 1).contiguous() if L != self.patch_length else f_BCL.permute(0, 2, 1).contiguous()
        
        ## quantize
        z = self.project_in(f_rest) # (B, patch_length, codebook_dim)
        codes = self.quantize(z) # (B, patch_length, codebook_dim)
        indices = self.codes_to_indices(codes.detach()) # (B, patch_length)
        code_count += self.indices_to_codecount(indices, new_mask)
        
        # replace ratio r of the codes with random codes
        if self.training and self.rratio > 0:
            random_indices = torch.randint(0, self.codebook_size, (B, self.patch_length), device=f_BCL.device)
            random_codes = self.indices_to_codes(random_indices)
            random_mask = torch.rand(B, self.patch_length, device=f_BCL.device) < self.rratio
            codes = torch.where(random_mask.unsqueeze(-1), random_codes, codes)
        
        ## dequantize
        zhat = self.project_out(codes) # (B, patch_length, C)
        h_BCL = F.interpolate(zhat.permute(0, 2, 1).contiguous(), size=(L,), mode='linear') if L != self.patch_length else zhat.permute(0, 2, 1).contiguous() # (B, C, L)
        h_BCL = self.quant_resi(h_BCL) # (B, C, L)

        ## compute vq loss and perplexity
        mean_vq_loss += F.mse_loss(h_BCL.data, f_BCL) + F.mse_loss(h_BCL, f_nograd)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return h_BCL, mean_vq_loss, perplexity

    def dequantize(self, idxs_BL: torch.Tensor):
        codes = self.indices_to_codes(idxs_BL) # (B, patch_length, codebook_dim)
        zhat = self.project_out(codes) # (B, patch_length, C)
        h_BCL = F.interpolate(zhat.permute(0, 2, 1).contiguous(), size=(self.L,), mode='linear').contiguous() if self.L != self.patch_length else zhat.permute(0, 2, 1).contiguous()
        h_BCL = self.quant_resi(h_BCL)
        return h_BCL
    
    def fsquantize(self, f_BCL: torch.Tensor):
        B, C, L = f_BCL.shape
        f_BCL = F.interpolate(f_BCL, size=(self.patch_length,), mode='area').permute(0, 2, 1).contiguous() if L != self.patch_length else f_BCL.permute(0, 2, 1).contiguous()

        z = self.project_in(f_BCL) # (B, patch_length, codebook_dim)
        codes = self.quantize(z) # (B, patch_length, codebook_dim)
        indices = self.codes_to_indices(codes) # (B, patch_length)
        return indices


class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks//2)
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BChw):
        return h_BChw.mul(1-self.resi_ratio) + super().forward(h_BChw).mul_(self.resi_ratio)

class ProjectIn(nn.Module):
    def __init__(self, dim, codebook_dim):
        super().__init__()

        self.project_in = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            Resnet1D(dim, 2, 1, activation='relu', norm=None),
            nn.Conv1d(dim, dim // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(dim // 2, codebook_dim, kernel_size=1, stride=1)
        )
    
    def forward(self, z):
        return self.project_in(z.permute(0, 2, 1)).permute(0, 2, 1).contiguous()

class ProjectOut(nn.Module):
    def __init__(self, dim, codebook_dim):
        super().__init__()
        
        self.project_out = nn.Sequential(
            nn.Conv1d(codebook_dim, dim // 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(dim // 2, dim, kernel_size=1, stride=1),
            Resnet1D(dim, 2, 1, activation='relu', norm=None),
            nn.Conv1d(dim, dim, kernel_size=1, stride=1),
        )
    
    def forward(self, z):
        return self.project_out(z.permute(0, 2, 1)).permute(0, 2, 1).contiguous()