# %%
from itertools import product

p = 4

all_combinations = []
all_combinations.append((0,))  # Include the (0,) combination
for i in range(1, p+1):
    for comb in product([0, 1], repeat=i):
        if comb[-1] == 1:  # Only include combinations ending with 1
            all_combinations.append(comb)


"""
expected output: [
(0), 
(1), 
(0, 1), 
(1, 1), 
(0, 0, 1), 
(1, 0, 1), 
(0, 1, 1), 
(1, 1, 1), 
(0, 0, 0, 1), 
(1, 0, 0, 1), 
(0, 1, 0, 1), 
(1, 1, 0, 1), 
(0, 0, 1, 1), 
(1, 0, 1, 1), 
(0, 1, 1, 1), 
(1, 1, 1, 1)
]
"""
print(all_combinations)
# %%
