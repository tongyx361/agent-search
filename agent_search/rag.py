from typing import Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

EG_QAS = [
    (
        r"""
Let $k, l > 0$ be parameters.
The parabola $y = kx^2 - 2kx + l$ intersects the line $y = 4$ at two points $A$ and $B$.
These points are distance 6 apart.
What is the sum of the squares of the distances from $A$ and $B$ to the origin?
""",
        r'''
```python
from sympy import symbols, solve, sqrt

def sum_of_squares_of_distances():
    """Let $k, l > 0$ be parameters. The parabola $y = kx^2 - 2kx + l$ intersects the line $y = 4$ at two points $A$ and $B$. These points are distance 6 apart. What is the sum of the squares of the distances from $A$ and $B$ to the origin?
"""
    # To find the sum of the squares of the distances from $A$ and $B$ to the origin, where $A$ and $B$ are points of intersection of the parabola $y = kx^2 - 2kx + l$ and the line $y = 4$, we first need to determine the x-coordinates of $A$ and $B$ by solving $kx^2 - 2kx + l = 4$, which simplifies to: $kx^2 - 2kx + (l - 4) = 0$.
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
    ),
    (
        r"""
Let the ‘sparkle' operation on positive integer $n$ consist of calculating the sum of the digits of $n$ and taking its factorial,
e.g. the sparkle of 13 is $4! = 24$. A robot starts with a positive integer on a blackboard,
then after each second for the rest of eternity, replaces the number on the board with its sparkle.
For some 'special' numbers, if they're the first number, then eventually every number that appears will be less than 6. How many such special numbers are there with at most 36 digits?
""",
        r'''
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
    ),
    (
        r"""
What is the minimum value of $5x^2+5y^2-8xy$ when $x$ and $y$ range over all real numbers such that $|x-2y| + |y-2x| = 40$?
""",
        r'''

```python
def find_minimum_integer():
    """What is the minimum value of $5x^2+5y^2-8xy$ when $x$ and $y$ range over all integer solutions of the constraint equation $|x-2y| + |y-2x| = 40$?
"""
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
    ),
    (
        r"""
There exists a unique increasing geometric sequence of five 2-digit positive integers. What is their sum?
""",
        r'''
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
    ),
    (
        r"""For how many positive integers $m$ does the equation $$\vert \vert x-1 \vert -2 \vert=\frac{m}{100}$$ have $4$ distinct solutions?""",
        r'''
```python
def number_of_solutions():
    """For how many positive integers $m$ does the equation \[\vert \vert x-1 \vert -2 \vert=\frac{m}{100}\] have $4$ distinct solutions? The final answer must be a non-negative integer.
"""
    # The original equaition simplifies to $\vert x-1 \vert = \pm \frac{m}{100} + 2 > 0$, and further simplifies to x1 = m/100 + 3, x2 = -m/100 + 3, x3 = m/100 + 1, x4 = -m/100 + 1.
    # To have 4 distinct solutions, m/100 should be less than 2, we only need to check values of m from 1 to 199.
    num_solutions = 0
    for m_value in range(1, 200):
        n = m/100
        xs = n + 3, -n + 3, n + 1, -n + 1
        if len(set(xs)) == 4:
          num_solutions += 1

    return num_solutions

result = number_of_solutions()
print(result)
```

```output
199
```

So, the number of positive integers $m$ for which the equation has 4 distinct solutions is 199. The correct answer is $\boxed{199}$.
''',
    ),
    (
        r"""
Suppose that we roll four 6-sided fair dice with faces numbered 1 to~6. Let $a/b$ be the probability that the highest roll is a 5, where $a$ and $b$ are relatively prime positive integers. Find $a + b$.
""",
        r'''

```python
from sympy import Rational, binomial, simplify

def a_plus_b_in_probability_highest_five():
    """Suppose that we roll four 6-sided fair dice with faces numbered 1 to~6. Let $a/b$ be the probability that the highest roll is a 5, where $a$ and $b$ are relatively prime positive integers. Find $a + b$. The final answer must be a non-negative integer."""
    # Total number of outcomes when rolling four dice
    total_outcomes = 6**4
    # Number of outcomes where the highest roll is a 5
    # This is equivalent to the number of outcomes where 1/2/3/4 dies shows 5 and the other 3/2/1/0 dice show values from 1 to 4
    favorable_outcomes = sum(binomial(4, n) * 4 ** (4 - n) for n in [1, 2, 3, 4])
    # Calculate the probability
    probability = Rational(favorable_outcomes, total_outcomes)
    # Simplify the probability
    simplified_probability = simplify(probability)
    # Extracting the numerator and denominator and summing them
    numerator = simplified_probability.numerator
    denominator = simplified_probability.denominator
    # Summing the numerator and denominator
    a_plus_b = numerator + denominator
    return a_plus_b

result = a_plus_b_in_probability_highest_five()
print(result)
```

```output
185
```

So $a + b = 185$. The answer is $\boxed{185}$.
''',
    ),
    (
        r"""
The points $\left(x, y\right)$ satisfying $((\vert x + y \vert - 10)^2 + ( \vert x - y \vert - 10)^2)((\vert x \vert - 8)^2 + ( \vert y \vert - 8)^2) = 0$ enclose a convex polygon.
What is the area of this convex polygon?
    """,
        r'''
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
    ),
    (
        r"""Let $ABCD$ be a unit square. Let $P$ be the point on $AB$ such that $|AP| = 1/{20}$ and let $Q$ be the point on $AD$ such that $|AQ| = 1/{24}$. The lines $DP$ and $BQ$ divide the square into four regions. Find the ratio between the areas of the largest region and the smallest region.""",
        r'''
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
    ),
]

EG_QAS = [
    (q.replace("\n", " ").replace(" <blank> ", "\n\n").strip(), a.strip())
    for q, a in EG_QAS
]


class RAG:
    def __init__(
        self,
        model: str = "Alibaba-NLP/gte-large-en-v1.5",
        qas: list[tuple[str, str]] = None,
        device: str = "cuda",
        torch_dtype=torch.float16,
        top_k: int = 1,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(
            model,
            trust_remote_code=True,
            # unpad_inputs=True,
            # use_memory_efficient_attention=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)
        self.model.eval()

        self.max_len = self.tokenizer.model_max_length
        self.qas = qas if qas else EG_QAS
        self.qa_embeds = self.embed([q + "\n\n" + a for q, a in self.qas])
        self.top_k = top_k

        print(self.__dict__)

    def embed(self, text: Union[str, list[str]], norm: bool = True) -> torch.Tensor:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        input_dict = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        # with torch.autocast(
        #     device_type=self.device.type, dtype=torch.float16
        # ):
        #     with torch.inference_mode():
        with torch.no_grad():
            outputs = self.model(**input_dict.to(self.device))

        embeddings = outputs.last_hidden_state[:, 0]
        if norm:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def retrieve(self, query: str) -> list[tuple[str, str]]:
        query_embed = self.embed(query)
        sim_scores = F.cosine_similarity(query_embed, self.qa_embeds, dim=1)
        print(f"{sim_scores=}")
        top_results = sim_scores.argsort(descending=True)[: self.top_k]
        top_scores = sim_scores[top_results]
        print(f"{top_scores=}")
        top_qas = [self.qas[i] for i in top_results]
        return top_qas
