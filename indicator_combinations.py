from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging
from itertools import combinations, product
import torch
from tqdm import tqdm
import aiohttp
import json
import os
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from . import ind_funcs
from .config import TRADING_PAIRS

class IndicatorCombinations:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ollama_url = "http://localhost:11434/api/generate"
        self.batch_size = 32
        self.num_workers = os.cpu_count()
        torch.set_num_threads(self.num_workers)
        torch.cuda.empty_cache()
        
        self.data_dir = os.path.join(Path(__file__).parent, 'data', 'historical')
        self.trading_pairs = TRADING_PAIRS

    # Rest of the class remains the same...
