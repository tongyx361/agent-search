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
    (
        r"""
Every morning, Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards.
When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop.
When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop.
Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her,including the $t$ minutes spent in the coffee shop.
""",
        r'''
```python
from sympy import symbols, solve, sqrt

def number_of_minutes_the_walk_takes():
    """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
"""
    # Define the variables
    s, t = symbols('s t')
    # Equation 1: When Aya walks at s km/h and the total time is 4 hours
    eq1 = Eq(9/s + t/60, 4)
    # Equation 2: When Aya walks at s+2 km/h and the total time is 2 hours 24 minutes
    eq2 = Eq(9/(s+2) + t/60, 2 + 24/60)
    # Solve the system of equations
    solution = solve((eq1, eq2), (s, t))
    # This problem has multi solutions
    # Get the solution that is bigger then zero
    for s_t in solution:
        s_value, t_value = s_t
        if s_value > 0 and t_value > 0:
            break
    # Calculate the time taken when walking at s+1/2 km/h
    speed_new = s_value + 1/2
    time_new_walk = 9 / speed_new  # in hours
    time_total_new = time_new_walk + t_value / 60  # in hours
    # Convert the total time to minutes
    time_total_new_minutes = time_total_new * 60
    return time_total_new_minutes

result = number_of_minutes_the_walk_takes()
print(result)
```

```output
204.000000000000
```

When Aya walks at $s + \frac{1}{2}$ kilometers per hour, the walk, including the time spent at the coffee shop, takes her 204 minutes. The answer is $\boxed{204}$.
''',
    ),
    (
        r"""
Real numbers $x$ and $y$ with $x,y>1$ satisfy $\log_x(y^x)=\log_y(x^{4y})=10.$ What is the value of $xy$?
""",
        r'''

```python
from sympy import symbols, log, solve

def value_of_xy():
    """Real numbers $x$ and $y$ with $x,y>1$ satisfy $\log_x(y^x)=\log_y(x^{4y})=10.$ What is the value of $xy$?
"""
    # Define the variables
    x, y = symbols('x y', positive=True, real=True)
    # Given equations
    eq1 = log(y**x, x) - 10
    eq2 = log(x**(4*y), y) - 10
    # Solve the system of equations
    solution = solve((eq1, eq2), (x, y))
    return solution

result = value_of_xy()
print(result)
```

```output
Runtime errors: Cell In[17], line 11
      8 eq2 = 10 * log(y) - 4...y*log(-10*LambertW(-log(y)/10)/log(y)) + 10*log(y)
```

Given that solving the system directly with SymPy's `solve` function is running into difficulties, let's try to simplify the problem manually and see if we can solve it analytically. Given: $$10 \log(x) = x \log(y)$$ $$10 \log(y) = 4y \log(x)$$ Let's solve these equations step-by-step. First, express one variable in terms of another. From the first equation: $$10 \log(x) = x \log(y) \implies \log(y) = \frac{10 \log(x)}{x}$$ Substitute $\log(y)$ into the second equation: $$10 \log\left(\frac{10 \log(x)}{x}\right) = 4y \log(x)$$ . Second, simplify the expressions. Notice from both equations: $$10 \log(x) = x \log(y)$$ $$\log(y) = \frac{10 \log(x)}{x}$$ . From: $$10 \log(y) = 4y \log(x)$$ , substitute $\log(y)$ : $$10 \left(\frac{10 \log(x)}{x}\right) = 4y \log(x)$$ $$\frac{100 \log(x)}{x} = 4y \log(x)$$ $$\frac{100}{x} = 4y$$ $$y = \frac{25}{x}$$ . Third, substitute $y$ back into one of the original equations. Using $$10 \log(x) = x \log(y)$$ $$10 \log(x) = x \log\left(\frac{25}{x}\right)$$ $$10 \log(x) = x (\log(25) - \log(x))$$ $$10 \log(x) = x \log(25) - x \log(x)$$ $$10 \log(x) + x \log(x) = x \log(25)$$ $$\log(x) (10 + x) = x \log(25)$$ $$\log(x) = \frac{x \log(25)}{10 + x}$$ . Finally, solve for x. Given that $x$ and $y$ are both positive real numbers greater than 1, solving this equation will provide us with the value of $x$ and subsequently $y$ . Solve this manually and verify. We can use numerical methods to solve for $x$ .

```python
import sympy as sp

def value_of_xy():
    """Real numbers $x$ and $y$ with $x,y>1$ satisfy $\log_x(y^x)=10.$ and $\log_x(y^x)=10.$ What is the value of $xy$?
"""
    # Define the variables
    x = sp.symbols('x', positive=True, real=True)
    # Equation in simplified form
    eq = sp.Eq(sp.log(x), (x * sp.log(25)) / (10 + x))
    # Solve the equation numerically
    x_value = sp.nsolve(eq, x, 2)
    # Calculate the value of y
    y_value = 25 / x_value
    # Calculate the value of xy
    xy_value = x_value * y_value
    return xy_value.evalf()

result = value_of_xy()
print(result)
```

```output
25.0000000000000
```

The value of $xy$ is $25$ . Thus, the answer is $\boxed{25}$.
''',
    ),
    (
        r"""
Alice and Bob play the following game. A stack of $n$ tokens lies before them.
The players take turns with Alice going first.
On each turn, the player removes $1$ token or $4$ tokens from the stack.
The player who removes the last token wins.
Find the number of positive integers $n$ less than or equal to $2024$ such that there is a strategy that guarantees that Bob wins,
regardless of Alice’s moves.
""",
        r'''
```python
def number_of_positive_integers_guarantee_bob_wins():
    """Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes $1$ token or $4$ tokens from the stack. The player who removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ such that there is a strategy that guarantees that Bob wins, regardless of Alice’s moves.
"""
    # Create a list to store the winning status for each number of tokens
    # True means the starting player (Alice) wins, False means Bob wins
    winning_status = [False] * (2024 + 1)
    # Initialize base cases
    # Alice wins if n = 1, since she can take the last token
    winning_status[1] = True
    # Alice loses if n = 2, since she can only take one token and leave the last token to Bob
    winning_status[2] = False
    # Alice wins if n = 3, since she can only take one token and leave the last 2 token to Bob
    winning_status[3] = True
    # Alice wins if n = 4, since she can take all four tokens in one time
    winning_status[4] = True
    # Determine the winning status for each number of tokens
    for n in range(5, 2024 + 1):
        # Alice wins if she can leave a losing position for Bob
        if not winning_status[n - 1] or not winning_status[n - 4]:
            winning_status[n] = True
        else:
            winning_status[n] = False
    # Count the number of losing positions for Alice, which means Bob wins
    bob_wins = sum(1 for win in winning_status[1:] if not win)

    return bob_wins

result = number_of_positive_integers_guarantee_bob_wins()
print(result)
```

```output
809
```

Therefore, the number of positive integers $n$ less than or equal to 2024 such that there is a strategy that guarantees Bob wins, regardless of Alice’s moves, is $809$.​ The answer is $\boxed{809}$.
''',
    ),
    (
        r"""
Jen enters a lottery by picking $4$ distinct numbers from $S=\{1,2,3,\cdots,9,10\}.$
$4$ numbers are randomly chosen from $S.$
She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers,
and wins the grand prize if all four of her numbers were the randomly chosen numbers.
The probability of her winning the grand prize given that she won a prize is $\tfrac{m}{n}$
where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
""",
        r'''
```python
import math
from fractions import Fraction

def find_m_n_sum():
    """were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
"""
    # Total number of ways to choose 4 numbers from 10
    total_combinations = math.comb(10, 4)
    # Number of ways Jen can pick 4 numbers
    jen_combinations = math.comb(10, 4)
    # Number of ways to choose exactly 4 numbers out of 4 chosen by Jen
    grand_prize_combinations = 1
    # Number of ways to choose 2, 3, or 4 numbers out of 4 chosen by Jen
    at_least_two_combinations = 0
    # Exactly 2 numbers from Jen's 4 and 2 from the other 6
    at_least_two_combinations += math.comb(4, 2) * math.comb(6, 2)
    # Exactly 3 numbers from Jen's 4 and 1 from the other 6
    at_least_two_combinations += math.comb(4, 3) * math.comb(6, 1)
    # Exactly 4 numbers from Jen's 4 (grand prize)
    at_least_two_combinations += grand_prize_combinations
    # Probability of winning the grand prize given that she won a prize
    prob_grand_prize_given_winning = Fraction(grand_prize_combinations, at_least_two_combinations)
    # Get the numerator and denominator
    m, n = prob_grand_prize_given_winning.numerator, prob_grand_prize_given_winning.denominator
    # Sum of m and n
    m_n_sum = m + n

    return m_n_sum


result = find_m_plus_n()
print(result)
```

```output
116
```

The value of $m + n$ for the given probability problem is $116$. The answer is $\boxed{116}$.
''',
    ),
    (
        r"""
Rectangles $ABCD$ and $EFGH$ are drawn such that $D,E,C,F$ are collinear. Also, $A,D,H,G$ all lie on a circle. If $BC=16,$ $AB=107,$ $FG=17,$ and $EF=184,$ what is the length of $CE$?
<blank>
[asy] import graph; unitsize(0.1cm); pair A = (0,0);pair B = (70,0);pair C = (70,16);pair D = (0,16);pair E = (3,16);pair F = (90,16);pair G = (90,33);pair H = (3,33); dot(A^^B^^C^^D^^E^^F^^G^^H); label("$A$", A, S);label("$B$", B, S);label("$C$", C, N);label("$D$", D, N);label("$E$", E, S);label("$F$", F, S);label("$G$", G, N);label("$H$", H, N); draw(E--D--A--B--C--E--H--G--F--C); [/asy]
""",
        r'''
```python
from sympy import symbols, Eq, solve

def length_of_CE():
    """Rectangles $ABCD$ and $EFGH$ are drawn such that $D,E,C,F$ are collinear. Also, $A,D,H,G$ all lie on a circle. If $BC=16,$ $AB=107,$ $FG=17,$ and $EF=184,$ what is the length of $CE$?
    [asy] import graph; unitsize(0.1cm);  pair A = (0,0);pair B = (70,0);pair C = (70,16);pair D = (0,16);pair E = (3,16);pair F = (90,16);pair G = (90,33);pair H = (3,33); dot(A^^B^^C^^D^^E^^F^^G^^H); label("$A$", A, S);label("$B$", B, S);label("$C$", C, N);label("$D$", D, N);label("$E$", E, S);label("$F$", F, S);label("$G$", G, N);label("$H$", H, N); draw(E--D--A--B--C--E--H--G--F--C); [/asy]
"""
    # To solve this problem, considering how to use the condition of circle
    # Extend $AD$ and $GH$ until they meet at $P$.
    # From the Power of a Point Theorem, we have $(PH)(PG)=(PD)(PA)$.
    # In triangle $PDEH$, $PD=EH,$ $PH=DE$
    # Considering $CE=CD-DE=AB-DE=107-DE,$ suppose $DE=x$, then $CE=107-x$
    # Then we get $PH=DE=x,$
    # $PA=PD+DA=EH+DA=FG+BC=33,$
    # $PG=PH+GH=DE+DA=x + 184$
    # Substituting in these values, we get $(x)(x + 184)=(17)(17 + 16)=561$
    # Define the variable $x$
    x = symbols("x")
    # Define the equation
    eq = Eq(x*(x + 184), 17 * (17 + 16))
    # Solve the equation
    solutions = solve(eq, x)
    # The high-ordered equation has muti solutions, get the $x$ that is larger than 0
    x_value = [solution for solution in solutions if solution > 0][0]
    # Then we get $CE=107-x$
    ce_length = 107 - x_value

    return ce_length

result = length_of_CE()
print(result)
```

```output
104
```

Therefore, the length of $CE$ is $104$. The answer is $\boxed{104}$.
''',
    ),
    (
        r"""
Consider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\times 8$ grid. Find the number of such paths that change direction exactly four times, like in the examples shown below.
<blank>
[asy] size(7.5cm); usepackage("tikz");label("\begin{tikzpicture}[scale=.4]\draw(0,0)grid(8,8);\draw[line width=2,red](0,0)--(2,0)--(2,3)--(5,3)--(5,8)--(8,8);\end{tikzpicture}",origin); label("\begin{tikzpicture}[scale=.4]\draw(0,0)grid(8,8);\draw[line width=2,red](0,0)--(0,3)--(3,3)--(3,5)--(8,5)--(8,8);\end{tikzpicture}",E); [/asy]
""",
        r'''
```python
from math import comb

def number_of_paths():
    """Consider the paths of length $16$ that follow the lines from the lower left corner to the upper right corner on an $8\times 8$ grid. Find the number of such paths that change direction exactly four times, like in the examples shown below.
    [asy] size(7.5cm); usepackage("tikz");label("\begin{tikzpicture}[scale=.4]\draw(0,0)grid(8,8);\draw[line width=2,red](0,0)--(2,0)--(2,3)--(5,3)--(5,8)--(8,8);\end{tikzpicture}",origin); label("\begin{tikzpicture}[scale=.4]\draw(0,0)grid(8,8);\draw[line width=2,red](0,0)--(0,3)--(3,3)--(3,5)--(8,5)--(8,8);\end{tikzpicture}",E); [/asy]
"""
    # To accurately solve this problem, we need to ensure that the paths change direction exactly four times. This means that there are precisely four points where the direction changes from horizontal to vertical or from vertical to horizontal.
    # To achieve this, the path direction is either vertical,horizontal,vertical,horizontal,vertical or horizontal,vertical,horizontal,vertical,horizontal.
    # For the former one, it means split the 8 horizontal moves into 3 segments, and split the 8 vertical moves into 2 segments.
    # The latter one is the same as the former one.
    # We'll compute the number of ways to split the 8 horizontal moves into 3 segments and the 8 vertical moves into 3 segments using combinations, and then multiply the two results.
    # Calculate paths with 3 segments of right moves and 2 segments of up moves
    right_segments_3 = combinations(8 - 1, 3 - 1)
    up_segments_2 = combinations(8 - 1, 2 - 1)
    # Calculate paths with 2 segments of right moves and 3 segments of up moves
    right_segments_2 = combinations(8 - 1, 2 - 1)
    up_segments_3 = combinations(8 - 1, 3 - 1)
    # Total paths is the sum of both cases
    total_paths = right_segments_3 * up_segments_2 + right_segments_2 * up_segments_3
    return total_paths

result = number_of_paths()
print(result)
```

```output
294
```

The number of paths of length 16 on an $8 \times 8$ grid that change direction exactly four times is $294$. The answer is $\boxed{294}$.
''',
    ),
    (
        r"""
Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$.
Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$.
""",
        r'''
```python
from sympy import isprime

def find_least_m():
    """Let $p$ be the least prime number for which there exists a positive integer $n$ such that $n^{4}+1$ is divisible by $p^{2}$. Find the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$.
"""
    # Least p, m satisfying $m^{4}+1$ is divisible by $p^{2}$ must satisfy m < p**2
    # Search Least p with Least solution m that 0 < m < p**2 in 1000 that satisfies $m^{4}+1$ is divisible by $p^{2}$
    for p in range(2, 1000):
        if isprime(p):
            for m in range(1, p**2):
                if (m**4 + 1) % (p**2) == 0:
                    return m

    return None

result = find_least_m()
print(result)
```

```output
110
```

Therefore, the least positive integer $m$ such that $m^{4}+1$ is divisible by $p^{2}$ is $110.$ The answer is $\boxed{110}$.
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
            unpad_inputs=True,
            use_memory_efficient_attention=True,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

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
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16
        ):  # or bfloat16
            with torch.inference_mode():
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
