"""
4x4 Mini-Sudoku Solver using Thermodynamic Sampling

This example demonstrates using THRML to solve a 4x4 Sudoku puzzle.
The puzzle uses values 0-3 (representing 1-4 in traditional Sudoku).

Constraints:
- Each row must contain unique values
- Each column must contain unique values
- Each 2x2 box must contain unique values
"""

import jax
import jax.numpy as jnp
import numpy as np
from thrml import (
    CategoricalNode,
    Block,
    BlockGibbsSpec,
    SamplingSchedule,
    sample_states,
)
from thrml.factor import FactorSamplingProgram
from thrml.models.discrete_ebm import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
)


def create_sudoku_grid():
    """Create a 4x4 grid of categorical nodes."""
    nodes = [[CategoricalNode() for _ in range(4)] for _ in range(4)]
    return nodes


def create_uniqueness_factor(cells, n_categories=4, penalty=-500.0):
    """
    Create a factor that penalizes duplicate values in a group of cells.

    For each pair of cells in the group, we create weights that give
    high energy (low probability) when both cells have the same value.

    Args:
        cells: List of CategoricalNode objects that must be unique
        n_categories: Number of possible values (4 for mini-Sudoku)
        penalty: Negative value to penalize same values (lower = stronger penalty)

    Returns:
        CategoricalEBMFactor that enforces uniqueness
    """
    # Create pairwise factors for all pairs in the group
    factors = []

    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            # Create weight matrix for this pair
            # Diagonal elements (same value) get penalty, off-diagonal get reward
            weights = np.zeros((1, n_categories, n_categories))
            for k in range(n_categories):
                weights[0, k, k] = penalty  # Strong penalty when both cells have same value
            # Small reward for different values
            for k1 in range(n_categories):
                for k2 in range(n_categories):
                    if k1 != k2:
                        weights[0, k1, k2] = 1.0  # Reward different values

            # Create factor for this pair
            node_groups = [Block([cells[i]]), Block([cells[j]])]
            factors.append(CategoricalEBMFactor(node_groups, jnp.array(weights)))

    return factors


def setup_sudoku_constraints(grid):
    """
    Set up all Sudoku constraints (row, column, and box uniqueness).

    Args:
        grid: 4x4 list of CategoricalNode objects

    Returns:
        List of CategoricalEBMFactor objects
    """
    all_factors = []

    # Row constraints
    for row in range(4):
        row_cells = [grid[row][col] for col in range(4)]
        all_factors.extend(create_uniqueness_factor(row_cells))

    # Column constraints
    for col in range(4):
        col_cells = [grid[row][col] for row in range(4)]
        all_factors.extend(create_uniqueness_factor(col_cells))

    # Box constraints (2x2 boxes)
    for box_row in range(2):
        for box_col in range(2):
            box_cells = []
            for r in range(2):
                for c in range(2):
                    box_cells.append(grid[box_row * 2 + r][box_col * 2 + c])
            all_factors.extend(create_uniqueness_factor(box_cells))

    return all_factors


def create_sampling_program(grid, givens):
    """
    Create a BlockSamplingProgram for solving Sudoku.

    Args:
        grid: 4x4 list of CategoricalNode objects
        givens: List of (row, col, value) tuples for initial clues

    Returns:
        Tuple of (program, init_free_state, init_clamped_state)
    """
    # Flatten grid for easier manipulation
    all_nodes = [node for row in grid for node in row]

    # Track which nodes are given (clamped) vs. free
    given_indices = set()
    given_values = {}
    for row, col, value in givens:
        idx = row * 4 + col
        given_indices.add(idx)
        given_values[idx] = value

    # Separate into free and clamped blocks
    free_nodes = []
    clamped_nodes = []
    clamped_values = []

    for idx, node in enumerate(all_nodes):
        if idx in given_indices:
            clamped_nodes.append(node)
            clamped_values.append(given_values[idx])
        else:
            free_nodes.append(node)

    # Create blocks
    free_blocks = [Block([node]) for node in free_nodes]
    clamped_blocks = [Block([node]) for node in clamped_nodes]

    # Set up node shape/dtype
    node_sd = {CategoricalNode: jax.ShapeDtypeStruct(shape=(), dtype=jnp.uint8)}

    # Create BlockGibbsSpec
    spec = BlockGibbsSpec(free_blocks, clamped_blocks, node_sd)

    # Get all constraint factors
    factors = setup_sudoku_constraints(grid)

    # Create conditional samplers (one per free block)
    samplers = [CategoricalGibbsConditional(n_categories=4) for _ in free_blocks]

    # Create sampling program
    program = FactorSamplingProgram(spec, samplers, factors, [])

    # Initialize states
    key = jax.random.key(42)
    init_free_state = [
        jax.random.randint(key, (1,), minval=0, maxval=4, dtype=jnp.uint8)
        for _ in free_blocks
    ]
    init_clamped_state = [
        jnp.array([val], dtype=jnp.uint8) for val in clamped_values
    ]

    return program, init_free_state, init_clamped_state, free_nodes, clamped_nodes


def count_violations(grid):
    """Count the number of constraint violations in a solution."""
    violations = 0

    # Check rows
    for row in grid:
        if len(set(row)) != 4:
            violations += (4 - len(set(row)))

    # Check columns
    for col in range(4):
        col_vals = [grid[row][col] for row in range(4)]
        if len(set(col_vals)) != 4:
            violations += (4 - len(set(col_vals)))

    # Check 2x2 boxes
    for box_row in range(2):
        for box_col in range(2):
            box_vals = []
            for r in range(2):
                for c in range(2):
                    box_vals.append(grid[box_row * 2 + r][box_col * 2 + c])
            if len(set(box_vals)) != 4:
                violations += (4 - len(set(box_vals)))

    return violations


def solve_sudoku(grid, givens, n_warmup=5000, n_samples=500, n_restarts=5, seed=42):
    """
    Solve a 4x4 Sudoku puzzle using thermodynamic sampling with multiple restarts.

    Args:
        grid: 4x4 list of CategoricalNode objects
        givens: List of (row, col, value) tuples for initial clues
        n_warmup: Number of warmup iterations
        n_samples: Number of samples to collect
        n_restarts: Number of random restarts to try
        seed: Random seed

    Returns:
        Best solution found (4x4 numpy array)
    """
    best_solution = None
    best_violations = float('inf')

    for restart in range(n_restarts):
        # Create fresh grid for each restart
        grid = create_sudoku_grid()

        # Create sampling program
        program, init_free, init_clamped, free_nodes, clamped_nodes = \
            create_sampling_program(grid, givens)

        # Set up sampling schedule
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=20
        )

        # Sample with different seed for each restart
        key = jax.random.key(seed + restart * 1000)
        all_nodes = free_nodes + clamped_nodes
        samples = sample_states(
            key,
            program,
            schedule,
            init_free,
            init_clamped,
            [Block(all_nodes)]
        )

        # Extract solution from last sample
        solution_flat = samples[0][-1]

        # Reconstruct full grid
        all_nodes_list = [node for row in grid for node in row]
        solution_dict = {}

        for i, node in enumerate(free_nodes):
            node_idx = all_nodes_list.index(node)
            solution_dict[node_idx] = int(solution_flat[i])

        for i, node in enumerate(clamped_nodes):
            node_idx = all_nodes_list.index(node)
            solution_dict[node_idx] = int(init_clamped[i][0])

        # Create 4x4 grid
        solution = np.zeros((4, 4), dtype=int)
        for idx in range(16):
            row = idx // 4
            col = idx % 4
            solution[row, col] = solution_dict[idx]

        # Check if this is the best solution so far
        violations = count_violations(solution)
        print(f"  Restart {restart + 1}/{n_restarts}: {violations} violations")

        if violations < best_violations:
            best_violations = violations
            best_solution = solution.copy()

        if violations == 0:
            break  # Found valid solution, stop early

    return best_solution


def print_sudoku(grid, title="Sudoku Grid"):
    """Pretty print a 4x4 Sudoku grid."""
    print(f"\n{title}")
    print("─" * 13)
    for i, row in enumerate(grid):
        row_str = "│"
        for j, val in enumerate(row):
            # Convert 0-3 to 1-4 for display, or show . for empty
            display_val = val + 1 if val >= 0 else "."
            row_str += f" {display_val}"
            if j == 1:
                row_str += " │"
        row_str += " │"
        print(row_str)
        if i == 1:
            print("├─────┼─────┤")
    print("─" * 13)


def is_valid_solution(grid):
    """Check if a 4x4 Sudoku solution is valid."""
    # Check rows
    for row in grid:
        if len(set(row)) != 4 or set(row) != {0, 1, 2, 3}:
            return False

    # Check columns
    for col in range(4):
        col_vals = [grid[row][col] for row in range(4)]
        if len(set(col_vals)) != 4 or set(col_vals) != {0, 1, 2, 3}:
            return False

    # Check 2x2 boxes
    for box_row in range(2):
        for box_col in range(2):
            box_vals = []
            for r in range(2):
                for c in range(2):
                    box_vals.append(grid[box_row * 2 + r][box_col * 2 + c])
            if len(set(box_vals)) != 4 or set(box_vals) != {0, 1, 2, 3}:
                return False

    return True


if __name__ == "__main__":
    # Example puzzle with more givens (easier to solve)
    # Values are 0-3 representing 1-4
    # Using -1 to represent empty cells in the initial display
    puzzle = [
        [0, -1, 2, -1],
        [-1, 2, -1, 0],
        [2, -1, 0, -1],
        [-1, 0, -1, 2]
    ]

    # Define givens as (row, col, value) tuples
    # Solution should be:
    # 1 4 3 2
    # 3 2 1 4  => 2 3 4 1
    # 3 1 2 4
    # 4 2 3 1  => 1 4 2 3
    givens = [
        (0, 0, 0),  # 1
        (0, 2, 2),  # 3
        (1, 1, 2),  # 3
        (1, 3, 0),  # 1
        (2, 0, 2),  # 3
        (2, 2, 0),  # 1
        (3, 1, 0),  # 1
        (3, 3, 2),  # 3
    ]

    print("=" * 50)
    print("4x4 Mini-Sudoku Solver with Thermodynamic Sampling")
    print("=" * 50)

    print_sudoku(puzzle, "Initial Puzzle")

    # Create grid and solve
    print("\nSolving with multiple restarts...")
    grid = create_sudoku_grid()
    solution = solve_sudoku(grid, givens, n_warmup=5000, n_samples=500, n_restarts=10, seed=42)

    print_sudoku(solution, "\nBest Solution Found")

    # Validate
    if is_valid_solution(solution):
        print("\n✓ Valid solution found!")
    else:
        violations = count_violations(solution)
        print(f"\n✗ Solution has {violations} constraint violations.")
        print("Note: Thermodynamic sampling is probabilistic and may not always")
        print("find perfect solutions for hard constraint problems like Sudoku.")
        print("Try increasing n_warmup, n_restarts, or the penalty parameter.")
