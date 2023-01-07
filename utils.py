import torch

def create_mask(src, tgt=None):
    # Mask out padded tokens in src
    src_mask = (src != src.tgt_pad).unsqueeze(-2)
    
    if tgt is not None:
        # Mask out padded tokens in tgt and future tokens in tgt
        tgt_mask = (tgt != tgt.tgt_pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    else:
        tgt_mask = None
        
    return src_mask, tgt_mask

def subsequent_mask(size):
    # Mask out future tokens in input sequence
    mask = torch.tril(torch.ones(size, size)).type(torch.uint8)
    return mask
