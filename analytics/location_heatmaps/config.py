from dataclasses import dataclass
from typing import List, Any
import numpy as np
TOPK = 1000
TOTAL_SIZE = 1024


@dataclass
class Config:
    dataset: List
    image: Any
    level_sample_size: int = 10000
    secagg_round_size: int = 10000
    threshold: float = 0
    collapse_threshold: float = None
    eps_func: Any  = lambda x, y: 1
    total_epsilon_budget: float = None
    top_k: int = TOPK
    partial:int  = 100
    max_levels: int  = 10
    threshold_func: Any = None
    collapse_func: Any = None
    total_size: int = TOTAL_SIZE
    min_dp_size: int = None
    dropout_rate: int = None
    output_flag: int = True
    quantize: bool = None
    noise_class: Any = None
    save_gif: bool = False
    positivity: bool = False
    start_with_level: int = 0
