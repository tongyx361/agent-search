#!/usr/bin/env python
# coding: utf-8

# # Analyze HF dataset
#

from datasets import load_dataset
import pandas as pd


amath_dset = load_dataset("MARIO-Math-Reasoning/AlphaMath-Trainset", split="train")
amath_dset


amath_df = pd.DataFrame(amath_dset)
amath_df
