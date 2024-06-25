#!/usr/bin/env python
# coding: utf-8

import os
import json
from agent_search.agent import AgentBase


test_home = "/ssddata/tongyx/projects/agent-search/data/iid_hardest_non_neg_int_ans_math/test-20240625-070522"
ref_fpath = (
    "/ssddata/tongyx/projects/agent-search/data/iid_hardest_non_neg_int_ans_math.json"
)


fin_trajs_list = []
for problem_dirname in os.listdir(test_home):
    if os.path.isdir(os.path.join(test_home, problem_dirname)):
        fin_trajs_fpath = os.path.join(
            test_home, problem_dirname, "fin-trajs-search.json"
        )
        with open(fin_trajs_fpath, "r") as f:
            fin_trajs = json.load(f)
print(f"{fin_trajs_list[0][:5]=}")


with open(ref_fpath, "r") as f:
    ref_dset = json.load(f)
print(f"{ref_dset[:5]=}")


dumb_agent = AgentBase()


assert dumb_agent.eq(455, 10455) == True


correct_idxs = []
for idx, (ref_sample, fin_trajs) in enumerate(zip(ref_dset, fin_trajs_list)):
    ref_ans = ref_sample["ref_ans"]
    ans = dumb_agent.ensemble_ans(fin_trajs)
    eq = dumb_agent.eq(ans, ref_ans)
    if eq:
        correct_idxs.append(idx)

correct_rate = len(correct_idxs) / len(ref_dset)
print(f"{correct_idxs=}")
print(f"{correct_rate=} (= {len(correct_idxs)}/{len(ref_dset)})")
