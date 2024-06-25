#!/usr/bin/env python
# coding: utf-8

# # Search answer for problems
#

import argparse
import torch
import json
import os
import csv

import time
from collections import defaultdict
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import set_seed

from agent_search.prompt import PromptTemplate
from agent_search.agent import VLLMPythonMathAgent, AgentSearchCfg


GPU_DTYPE = torch.bfloat16
LOCAL_GPU_MEM_GB = 80
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(
    description="Search for search strategies and hyper-parameters", allow_abbrev=False
)

parser.add_argument(
    "--model_id_or_path",
    type=str,
    default="deepseek-ai/deepseek-math-7b-rl",
    help="The HF model ID or path to the model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="The HF model ID or path to the model",
)
parser.add_argument(
    "--gpu_mem_util",
    type=float,
    default=0.9,
    help="The fraction of GPU memory to use",
)
parser.add_argument(
    "--gen_dset_fpath",
    type=str,
    help="The path to the dataset to generate the search space from",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.6,
    help="The temperature to use for sampling",
)

parser.add_argument(
    "--code_temperature",
    type=float,
    default=0.6,
    help="The temperature to use for sampling code",
)

parser.add_argument(
    "--init_temperature",
    type=float,
    default=1.0,
    help="The temperature to use for init sampling",
)

parser.add_argument(
    "--top_p",
    type=float,
    default=0.95,
    help="The top_p to use for sampling",
)

parser.add_argument(
    "--allow_timeout",
    action="store_true",
    help="Whether to allow timeout when executing code",
)

parser.add_argument(
    "--allow_err",
    action="store_true",
    help="Whether to allow errors when executing code",
)

parser.add_argument(
    "--allow_empty_output",
    action="store_true",
    help="Whether to allow empty outputs when executing code",
)

parser.add_argument(
    "--n_fin_ans",
    type=int,
    default=8,
    help="The number of answers to sample per problem",
)


parser.add_argument(
    "--n_max_out_tok",
    type=int,
    default=4096,
    help="The maximum number of tokens to generate. Few completion are longer than 1024 tokens",
)

parser.add_argument(
    "--n_max_tot_tok",
    type=int,
    default=4096,
    help="The maximum number of tokens to generate",
)

parser.add_argument(
    "--n_init_gen",
    type=int,
    default=1,
    help="The number of init generations",
)

parser.add_argument(
    "--n_init_sample",
    type=int,
    default=16,
    help="The number of samples when init sampling",
)

parser.add_argument(
    "--n_max_sample",
    type=int,
    default=1024,
    help="The maximum number of samples for trial per problem",
)

parser.add_argument(
    "--beam_width",
    type=int,
    default=1,
    help="The beam width to use for sampling",
)

parser.add_argument(
    "--n_max_call",
    type=int,
    default=None,
    help="DeepSeekMath-7B-RL usually can not correctly solve problems with more than 2 calls",
)

parser.add_argument(
    "--instr",
    type=str,
    default="The final answer must be a non-negative integer.",
    # default="""To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step.
    # Be clear so even an idiot can follow your instructions.
    # Write the entire script covering all the steps (use comments and document it well) and print the result.
    # The intermediary calculations may be free-form, but the final answer should always be an non-negative integer.
    # Your final answer should always be a non-negative integer, not an algebraic expression!
    # Your final answer should always be a non-negative integer, not an algebraic expression!
    # Your final answer should always be a non-negative integer, not an algebraic expression!
    # After solving the problem, output the final answer within \\boxed{}.""",
    help="The instruction to append to the problem.",
)

parser.add_argument(
    "--rag",
    action="store_true",
    help="Whether to use RAG model for generation",
)

args, _ = parser.parse_known_args()


set_seed(args.seed)


ICL_EG_QA_MAP = {
    """
Let $k, l > 0$ be parameters.
The parabola $y = kx^2 - 2kx + l$ intersects the line $y = 4$ at two points $A$ and $B$.
These points are distance 6 apart.
What is the sum of the squares of the distances from $A$ and $B$ to the origin?
    """: r'''
```python
from sympy import symbols, solve, sqrt

def sum_of_squares_of_distances():
    """Let $k, l > 0$ be parameters. The parabola $y = kx^2 - 2kx + l$ intersects the line $y = 4$ at two points $A$ and $B$. These points are distance 6 apart. What is the sum of the squares of the distances from $A$ and $B$ to the origin?
"""
    x, k, l = symbols('x k l')
    # Equation of the parabola
    parabola_eq = k*x**2 - 2*k*x + l - 4
    # Solve for x when y = 4
    x_values = solve(parabola_eq, x)
    # Distance from A and B to the origin
    distance_A = sqrt(x_values[0]**2 + 4**2)
    distance_B = sqrt(x_values[1]**2 + 4**2)
    # The sum of the squares of the distances
    sum_of_squares = distance_A**2 + distance_B**2
    # Substitute the condition that the distance between A and B is 6
    sum_of_squares = sum_of_squares.subs(sqrt((x_values[1] - x_values[0])**2), 6)

    return sum_of_squares

result = sum_of_squares_of_distances()
print(result)
```

```output
32 + (k - sqrt(k*(k - l + 4)))**2/k**2 + (k + sqrt(k*(k - l + 4)))**2/k**2
```

The code seem to fail to solve the equations directly. We need to further simplify the equation.

We know that the distance between $A$ and $B$ is $x_2-x_1=6$, and the Vieta's formulas give us that the sum of the roots of the equation $kx^2 - 2kx + l - 4 = 0$ is $x_1 +x_2 =-\frac{b}{a} = -\frac{-2k}{k} = 2$, and the product of the roots is $x_1x_2 = \frac{c}{a} = \frac{l - 4}{k}$. So we can use the identity $\left(x_1 - x_2\right)^2 = \left(x_1 + x_2\right)^2 - 4x_1x_2$ to find a relationship between $k$ and $l$.

By substitution, we have $\left(x_1 - x_2\right)^{2} = - 4\cdot\frac{l - 4}{k} = 6^2$, which simplifies to $4 - \frac{4l - 16}{k} = 36$, or $\frac{4l - 16}{k} = -32$. Multiplying both sides by $k$ gives $4l - 16 = -32k$, or $4l = 16 - 32k$. Dividing both sides by 4 gives $l = 4 - 8k$.

Now, we can substitute this expression for $l$ into the sum of squares of distances to get the final result.

```python
def sum_of_squares_of_distances_final():
    """Let $k, l > 0$ be parameters. The parabola $y = kx^2 - 2kx + l$ intersects the line $y = 4$ at two points $A$ and $B$. These points are distance 6 apart. What is the sum of the squares of the distances from $A$ and $B$ to the origin?
"""
    x, k = symbols('x k')
    # Equation of the parabola with l = 4 - 8k
    parabola_eq = k*x**2 - 2*k*x + 4 - 8*k - 4
    # Solve for x when y = 4
    x_values = solve(parabola_eq, x)
    # Distance from A and B to the origin
    distance_A = sqrt(x_values[0]**2 + 4**2)
    distance_B = sqrt(x_values[1]**2 + 4**2)
    # The sum of the squares of the distances
    sum_of_squares = distance_A**2 + distance_B**2
    # Substitute the condition that the distance between A and B is 6
    sum_of_squares = sum_of_squares.subs(sqrt((x_values[1] - x_values[0])**2), 6)

    return sum_of_squares

result = sum_of_squares_of_distances_final()
print(result)
```

```output
52
```

So, the sum of the squares of the distances from $A$ and $B$ to the origin is $52$. The answer is $\boxed{52}$.
'''
}

ICL_EG_QA_MAP = {
    k.replace("\n", " ").strip(): v.strip() for k, v in ICL_EG_QA_MAP.items()
}
for i, (q, a) in enumerate(ICL_EG_QA_MAP.items()):
    print(f"### Problem {i}")
    print(f"#### Problem")
    print(q)
    print(f"#### Solution")
    print(a)


dset = []
with open(args.gen_dset_fpath, "r") as f:
    for line in csv.reader(f):
        if line[0] == "id":
            continue
        dset.append({"query": line[1], "ref_ans": int(line[2])})


llm = LLM(
    model=args.model_id_or_path,
    tokenizer=args.model_id_or_path,
    tensor_parallel_size=torch.cuda.device_count(),
    dtype=GPU_DTYPE,
    seed=args.seed,
    trust_remote_code=True,
    gpu_memory_utilization=args.gpu_mem_util,
    swap_space=args.gpu_mem_util * LOCAL_GPU_MEM_GB,
)


sampling_params = SamplingParams(
    n=args.beam_width,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.n_max_out_tok,
    skip_special_tokens=True,
    # prompt_logprobs=1,
    # logprobs=0,
    seed=args.seed,
)
print(f"{sampling_params=}")
search_cfg = AgentSearchCfg(
    sampling_params=sampling_params,
    n_fin_ans=args.n_fin_ans,
    n_init_sample=args.n_init_sample,
    n_max_sample=args.n_max_sample,
    n_init_gen=args.n_init_gen,
    init_temperature=args.init_temperature,
    code_temperature=args.code_temperature,
    n_max_tot_tok=args.n_max_tot_tok,
    n_max_call=args.n_max_call,
    allow_timeout=args.allow_timeout,
    allow_err=args.allow_err,
    allow_empty_output=args.allow_empty_output,
)
print(f"{search_cfg.__dict__=}")


agent = VLLMPythonMathAgent(
    llm,
    prompt_template="deepseekmath-tool",
    search_cfg=search_cfg,
    code_cell_template="deepseekmath",
    rag_eg_qa_map=ICL_EG_QA_MAP if args.rag else None,
    eq=lambda x, y: x % 1000 == y % 1000,
)

agent.ctx_template.prompt_template.prompt_after_query = (
    " " + args.instr + agent.ctx_template.prompt_template.prompt_after_query
)
agent.ctx_template.code_cell_template.trunc_len = (100, 100)


search_home = os.path.splitext(args.gen_dset_fpath)[0]
search_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
search_dir = os.path.join(search_home, f"search-{search_timestamp}")
domain_correct_idxs = defaultdict(list)

problem_times = []

try:
    for idx, dp in enumerate(dset):
        search_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # search_dir = os.path.join(seed_dir, f"problem-{idx}-{search_timestamp}")
        search_dir = os.path.join(search_dir, f"problem-{idx}-{search_timestamp}")
        print(f"### {idx}")
        print("#### Problem")
        print(dp["query"])
        print("#### Solution")
        problem_start = time.time()
        ans = agent.search(
            init_input=dp["query"], ref_ans=dp["ref_ans"], output_dir=search_dir
        )
        problem_time = time.time() - problem_start
        problem_times.append(problem_time)
        mod_ref_ans = dp["ref_ans"] % 1000
        eq = (ans % 1000) == mod_ref_ans
        if eq and "domain" in dp:
            domain_correct_idxs[dp["domain"]].append(idx)
        print(
            f"Pred. ({ans} % 1000) {'=' if eq else '!'}= Ref. ({dp['ref_ans']} % 1000)"
        )
        print(f"Problem time: {problem_time:.2f}s")
        print(
            f"Avg. Time so far: {sum(problem_times)/len(problem_times):.2f}s (= {sum(problem_times):.2f} / {len(problem_times)})"
        )
        all_correct_idxs = sum(domain_correct_idxs.values(), [])
        print(
            f"Acc. so far: {len(all_correct_idxs) / (idx + 1):.2%} (= {len(all_correct_idxs)} / {idx + 1})"
        )
except BaseException as e:
    raise e
finally:
    all_correct_idxs = sum(domain_correct_idxs.values(), [])
    domain_correct_cnt = {k: len(v) for k, v in domain_correct_idxs.items()}

    metrics = {
        "search_cfg": {
            **{k: v for k, v in search_cfg.__dict__.items() if k != "sampling_params"},
            "sampling_params": search_cfg.sampling_params.__repr__(),
        },
        "acc": len(all_correct_idxs) / len(dset),
        "domain_correct_cnt": domain_correct_cnt,
        "avg_time": sum(problem_times) / len(problem_times),
        "domain_correct_idxs": domain_correct_idxs,
        "problem_times": problem_times,
    }

    with open(os.path.join(search_dir, f"metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
