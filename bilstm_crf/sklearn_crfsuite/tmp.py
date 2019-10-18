# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 11:05
# @Author  : RIO
# @desc: TODO:DESC
from tqdm import tqdm
import time
pbar = tqdm(["a", "b", "c", "d"])
for char in pbar:
    time.sleep(1)
    pbar.set_description("Processing %s" % char)