#!/usr/bin/env python
# coding: utf-8

# # Search for search strategies and hyper-parameters
#

import argparse
import torch
import json
import os

import time
from collections import defaultdict
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import set_seed

from agent_search.prompt import PromptTemplate
from agent_search.agent import VLLMPythonMathAgent, AgentSearchCfg


GPU_DTYPE = torch.float16
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
    "--spec_neg",
    action="store_true",
    help="Whether to speculate negative answers",
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
    default=64,
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
    r"""
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
''',
    r"""
Let $ABCD$ be a unit square.
Let $P$ be the point on $AB$ such that $|AP| = 1/{20}$ and let $Q$ be the point on $AD$ such that $|AQ| = 1/{24}$. The lines $DP$ and $BQ$ divide the square into four regions.
Find the ratio between the areas of the largest region and the smallest region.
    """: r'''

```python
import numpy as np

def area_ratio():
    """Let $ABCD$ be a unit square. Let $P$ be the point on $AB$ such that $|AP| = 1/{20}$ and let $Q$ be the point on $AD$ such that $|AQ| = 1/{24}$. The lines $DP$ and $BQ$ divide the square into four regions. Find the ratio between the areas of the largest region and the smallest region. The final answer must be a non-negative integer."""
    # Define points of the square
    A = np.array([0, 0])
    B = np.array([1, 0])
    C = np.array([1, 1])
    D = np.array([0, 1])
    # Define points P and Q
    P = np.array([1 / 20, 0])
    Q = np.array([0, 1 / 24])
    # Calculate intersection of DP and BQ
    # Line DP: y = (1 - P[0]) / (P[1] - 1) * (x - P[0])
    # Line BQ: y = Q[1] / (Q[0] - 1) * (x - Q[0]) + Q[1]
    # Solving for x and y at intersection
    # Line equations in the form ax + by = c
    a1, b1, c1 = (
        (D[1] - P[1]),
        (P[0] - D[0]),
        (D[1] - P[1]) * D[0] + (P[0] - D[0]) * D[1],
    )
    a2, b2, c2 = (
        (Q[1] - B[1]),
        (B[0] - Q[0]),
        (Q[1] - B[1]) * B[0] + (B[0] - Q[0]) * B[1],
    )
    # Calculate determinant
    det = a1 * b2 - a2 * b1
    if det == 0:
        return "Lines are parallel"
    # Intersection point I
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    I = np.array([x, y])
    # Define the four regions
    regions = [[A, Q, I], [Q, D, I], [D, C, I], [C, B, I, P]]
    # Calculate areas of the regions
    def area(poly):
        x = poly[:, 0]
        y = poly[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    areas = [area(np.array(region)) for region in regions]
    smallest_area = min(areas)
    largest_area = max(areas)
    # Calculate the ratio
    ratio = largest_area / smallest_area

    return ratio

result = area_ratio()
print(result)
```

```output
480.00000000000006
```

The ratio between the areas of the largest region and the smallest region is $480$. The answer is $\boxed{480}$.
''',
    r"""
Let the ‘sparkle' operation on positive integer $n$ consist of calculating the sum of the digits of $n$ and taking its factorial, e.g. the sparkle of 13 is $4! = 24$.
A robot starts with a positive integer on a blackboard, then after each second for the rest of eternity, replaces the number on the board with its sparkle.
For some 'special' numbers, if they're the first number, then eventually every number that appears will be less than 6.
How many such special numbers are there with at most 36 digits?
""": r'''
```python
import math

def special_numbers():
    """Let the ‘sparkle' operation on positive integer $n$ consist of calculating the sum of the digits of $n$ and taking its factorial, e.g. the sparkle of 13 is $4! = 24$. A robot starts with a positive integer on a blackboard, then after each second for the rest of eternity, replaces the number on the board with its sparkle. For some 'special' numbers, if they're the first number, then eventually every number that appears will be less than 6. How many such special numbers are there with at most 36 digits?
"""
    # The only factorials less than 6 are that of 1 and 2.
    # Therefore, all special numbers must have digit sums of 1 or 2.
    # Also, no numbers with digit sums of 1 or 2 are factorials of any other number,
    # because they cannot be divisible by 3 (the digit sum of a number divisible by 3 is also divisible by 3).
    # Case 1: All 0s and one 1 (e.g., 10000)
    case_1 = math.comb(36, 1)
    # Case 2: All 0s and one 2 (e.g., 20000)
    case_2 = math.comb(36, 1)
    # Case 3: All 0s and two 1s (e.g., 10010)
    case_3 = math.comb(36, 2)
    # Total number of special numbers
    total_special_numbers = case_1 + case_2 + case_3

    return total_special_numbers

result = special_numbers()
print(result)
```

```output
702
```

So there are 702 special numbers with at most 36 digits. The answer is $\boxed{702}$.
''',
    r"""
What is the minimum value of $5x^2+5y^2-8xy$ when $x$ and $y$ range over all real numbers such that $|x-2y| + |y-2x| = 40$?
""": r'''
```python
def find_minimum_integer():
    """What is the minimum value of $5x^2+5y^2-8xy$ when $x$ and $y$ range over all integer solutions of the constraint equation $|x-2y| + |y-2x| = 40$?"""
    # The minimum value of $5x^2+5y^2-8xy$ when $x$ and $y$ range over all real numbers such that $|x-2y| + |y-2x| = 40$ is a non-negative integer.
    # So we can solve this problem by trying all integer solutions of the constraint equation and then finding the minimum value of $5x^2 + 5y^2 - 8xy$.
    min_value = float('inf')
    for x in range(-100, 101):
        for y in range(-100, 101):
            if abs(x - 2*y) + abs(y - 2*x) == 40:
                value = 5*x**2 + 5*y**2 - 8*x*y
                if value < min_value:
                    min_value = value
    return min_value

result = find_minimum_integer()
print(result)
```

```output
800
```

The minimum value of $5x^2+5y^2-8xy$ when $x$ and $y$ range over all real numbers such that $|x-2y| + |y-2x| = 40$ is a non-negative integer, and it is 800. So the answer is $\boxed{800}$.
''',
    r"""
There exists a unique increasing geometric sequence of five 2-digit positive integers. What is their sum?
""": r'''
```python
from sympy import symbols, solve, simplify

def sum_of_sequence():
    """There exists a unique increasing geometric sequence of five 2-digit positive integers. What is their sum?
"""
    a, r = symbols('a r')
    # The first term of the sequence is a, and the fifth term is a*r^4.
    # The sequence is increasing, so r > 1.
    # The sequence consists of 2-digit positive integers, so a and a*r^4 must be in the range 10 to 99.
    # We can solve this problem by trying all integers in the range under the constraint.
    for ar4 in range(99, 9, -1):
        for a in range(10, 100):
            r = (ar4 / a) ** (1 / 4)
            if r > 1:
                ar = a * r
                ar2 = ar * r
                ar3 = ar2 * r
                if ar.is_integer() and ar2.is_integer() and ar3.is_integer():
                    sum_of_sequence = sum([a, ar, ar2, ar3, ar4])

    return sum_of_sequence

result = sum_of_sequence()
print(result)
```

```output
211.0
```

So the sum of the unique increasing geometric sequence of five 2-digit positive integers is 211. The answer is $\boxed{211}$.
''',
    r"""
The points $\left(x, y\right)$ satisfying $((\vert x + y \vert - 10)^2 + ( \vert x - y \vert - 10)^2)((\vert x \vert - 8)^2 + ( \vert y \vert - 8)^2) = 0$ enclose a convex polygon.
What is the area of this convex polygon?
""": r'''
```python
from sympy import symbols, Eq, solve

def polygon_vertices():
    """The points $\left(x, y\right)$ satisfying $((\vert x + y \vert - 10)^2 + ( \vert x - y \vert - 10)^2)((\vert x \vert - 8)^2 + ( \vert y \vert - 8)^2) = 0$ enclose a convex polygon. What is the area of this convex polygon?"""
    # Define the variables
    x, y = symbols("x y")
    # Define the equations from the first factor
    eq1_cases = [Eq(x + y, 10), Eq(x + y, -10)]
    eq2_cases = [Eq(x - y, 10), Eq(x - y, -10)]
    # Define the equations from the second factor
    eq3_cases = [Eq(x, 8), Eq(x, -8)]
    eq4_cases = [Eq(y, 8), Eq(y, -8)]
    # Solve each combination of equations
    solutions = []
    # Solving combinations from the first factor
    for eq1 in eq1_cases:
        for eq2 in eq2_cases:
            sol = solve([eq1, eq2], (x, y))
            if sol:
                solutions.append(sol)
    # Solving combinations from the second factor
    for eq3 in eq3_cases:
        for eq4 in eq4_cases:
            sol = solve([eq3, eq4], (x, y))
            if sol:
                solutions.append(sol)
    # Extract unique solutions
    unique_solutions = {tuple(sol.items()) for sol in solutions}
    return unique_solutions

result = polygon_vertices()
print(result)
```

```output
{((x, 10), (y, 0)), ((x, 0), (y, -10)), ((x, -8), (y, 8)), ((x, 8), (y, 8)), ((x, -10), (y, 0)), ((x, -8), (y, -8)), ((x, 8), (y, -8)), ((x, 0), (y, 10))}
```

Now we have the coordinates of all the vertices. To find the area of this polygon, we can use the Shoelace formula (Gauss's area formula for polygons): $\text{Area} = \frac{1}{2} \left| \sum_{i=1}^{n} (x_i y_{i+1} - y_i x_{i+1}) \right|$

```python
def polygon_area():
    # Reorder the vertices in a logical sequence
    vertices_ordered = [(10, 0), (8, 8), (0, 10), (-8, 8), (-10, 0), (-8, -8), (0, -10), (8, -8)]
    # Repeat the first vertex at the end
    vertices = vertices_ordered + [vertices_ordered[0]]
    # Calculate the area using the Shoelace formula
    area = 0
    for i in range(len(vertices) - 1):
        area += (
            vertices[i][0] * vertices[i + 1][1] - vertices[i + 1][0] * vertices[i][1]
        )
    return abs(area) / 2

result = polygon_area()
print(result)
```

```output
320.0
```

Therefore, the area of the convex polygon is $320$. The answer is $\boxed{320}$.
''',
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


llm = LLM(
    model=args.model_id_or_path,
    tokenizer=args.model_id_or_path,
    tensor_parallel_size=torch.cuda.device_count(),
    dtype=GPU_DTYPE,
    seed=args.seed,
    trust_remote_code=True,
    gpu_memory_utilization=args.gpu_mem_util,
    # swap_space=args.gpu_mem_util * LOCAL_GPU_MEM_GB,
)


with open(args.gen_dset_fpath, "r") as f:
    dset = json.load(f)
print(f"Loaded dataset with {len(dset)} problems")


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
    spec_neg=args.spec_neg,
)
print(f"{search_cfg.__dict__=}")


agent = VLLMPythonMathAgent(
    llm,
    prompt_template="deepseekmath-tool",
    search_cfg=search_cfg,
    code_cell_template="deepseekmath",
    rag_eg_qa_map=ICL_EG_QA_MAP if args.rag else None,
)

agent.ctx_template.prompt_template.prompt_after_query = (
    " " + args.instr + agent.ctx_template.prompt_template.prompt_after_query
)


test_home = os.path.splitext(args.gen_dset_fpath)[0]
test_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
test_dir = os.path.join(test_home, f"test-{test_timestamp}")
domain_correct_idxs = defaultdict(list)

problem_times = []

try:
    for idx, dp in enumerate(dset[29:]):
        search_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # search_dir = os.path.join(seed_dir, f"problem-{idx}-{search_timestamp}")
        search_dir = os.path.join(test_dir, f"problem-{idx}-{search_timestamp}")
        print(f"### {idx}")
        print("#### Problem")
        print(dp["query"])
        print("#### Solution")
        problem_start = time.time()
        ans = agent.search(init_input=dp["query"], output_dir=search_dir)
        problem_time = time.time() - problem_start
        problem_times.append(problem_time)
        eq = agent.eq(ans, dp["ref_ans"])
        if eq and "domain" in dp:
            domain_correct_idxs[dp["domain"]].append(idx)
        print(f"Pred. ({ans}) {'=' if eq else '!'}= Ref. ({dp['ref_ans']})")
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

    with open(os.path.join(test_dir, f"metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
