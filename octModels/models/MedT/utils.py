# from https://github.com/jeya-maria-jose/Medical-Transformer
import torch.nn as nn


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

