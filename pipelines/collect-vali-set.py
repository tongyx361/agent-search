#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


# # Collect validation set
#

import pandas as pd

from datasets import load_dataset
from agent_search.parse import extract_boxed


math_dset = load_dataset("hendrycks/competition_math", split="test")
math_dset


math_df = pd.DataFrame(math_dset)
math_df


math_df["level"] = math_df["level"].apply(lambda x: int(x.split(" ")[-1]))


math_df["ans_str"] = math_df["solution"].apply(extract_boxed)
math_df


math_df["is_ans_int"] = math_df["ans_str"].apply(lambda x: x.isnumeric())
math_df


int_ans_math_df = math_df[math_df["is_ans_int"]]
int_ans_math_df["ans"] = int_ans_math_df["ans_str"].apply(int)
int_ans_math_df


math_df["level"].value_counts().sort_index().plot(kind="bar")


int_ans_math_df["level"].value_counts().sort_index().plot(kind="bar")


non_neg_int_ans_math_df = int_ans_math_df[int_ans_math_df["ans"] >= 0]
non_neg_int_ans_math_df


hardest_non_neg_int_ans_math_df = non_neg_int_ans_math_df[int_ans_math_df["level"] >= 5]
hardest_non_neg_int_ans_math_df


hardest_non_neg_int_ans_math_df["type"].value_counts().plot(kind="bar")


min_type_size = min(hardest_non_neg_int_ans_math_df["type"].value_counts().values)
min_type_size


# ## I.I.D
#

iid_hardest_non_neg_int_ans_math_df = hardest_non_neg_int_ans_math_df.groupby(
    "type"
).sample(frac=0.23, random_state=42)
iid_hardest_non_neg_int_ans_math_df


iid_hardest_non_neg_int_ans_math_df = iid_hardest_non_neg_int_ans_math_df.rename(
    {
        "problem": "query",
        "ans": "ref_ans",
        "type": "domain",
        "solution": "ref_resp",
    },
    axis=1,
)
iid_hardest_non_neg_int_ans_math_df


iid_hardest_non_neg_int_ans_math_df[
    ["domain", "level", "ref_ans", "query", "ref_resp"]
].sample(frac=1, random_state=42).to_json(
    "../data/iid_hardest_non_neg_int_ans_math.json", orient="records", indent=2
)


# ## Balanced
#

balanced_hardest_non_neg_int_ans_math_df = hardest_non_neg_int_ans_math_df.groupby(
    "type"
).head(10)
balanced_hardest_non_neg_int_ans_math_df


balanced_hardest_non_neg_int_ans_math_df


balanced_hardest_non_neg_int_ans_math_df = (
    balanced_hardest_non_neg_int_ans_math_df.rename(
        {
            "problem": "query",
            "ans": "ref_ans",
            "type": "domain",
            "solution": "ref_resp",
        },
        axis=1,
    )
)
balanced_hardest_non_neg_int_ans_math_df


balanced_hardest_non_neg_int_ans_math_df[
    ["domain", "level", "ref_ans", "query", "ref_resp"]
].sample(frac=1, random_state=42).to_json(
    "../data/balanced_hardest_non_neg_int_ans_math.json", orient="records", indent=2
)
