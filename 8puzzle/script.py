import importlib
import random
import time

# Number of tests to run
NUMBER_OF_TESTS = 1


def main():
    # Run multiple tests
    multiple_tests(NUMBER_OF_TESTS)


def multiple_tests(num_tests: int):
    """
    Run multiple tests and print the results
    Args:
        num_tests: int
    """
    times: list[tuple[float, float]] = []
    for _ in range(num_tests):
        times.append(solve())

    print("Average time for tile switches: " + str(sum([t[0] for t in times]) / num_tests))
    print("Average time for manhatten: " + str(sum([t[1] for t in times]) / num_tests))


def solve() -> (float, float):
    """
    Solve the puzzle and return the time it took to solve
    Returns:
        tuple: (float, float)
    """

    # Generate a random puzzle
    initial_puzzle: str = generate_puzzle()
    print("Initial puzzle: \n" + initial_puzzle)

    # Solve the puzzle using manhatten_distance
    print("Manhatten \n")
    start_time2 = time.time()
    solve_manhatten(initial_puzzle)
    temps_exec2 = time.time() - start_time2
    print("L'Heuristique de manhatten à pris " + str(temps_exec2) + " secondes\n")

    # Solve the puzzle using tile switches remaining
    print("Tile switch \n")
    start_time = time.time()
    solve_tile_switches(initial_puzzle)
    temps_exec = time.time() - start_time
    print("L'Heuristique de Tile switch à pris " + str(temps_exec) + " secondes\n")

    # Compare the two heuristics and print the results
    print("La différence de ces 2 heuristiques est de " + str(abs(temps_exec2 - temps_exec)) + " secondes\n\n")
    return temps_exec, temps_exec2


# Solve the puzzle using tile switches remaining
def solve_tile_switches(initial_puzzle: str, goal: str = "1 2 3 4 5 6 7 8 0"):
    """
    Solve the puzzle using tile switches remaining
    Args:
        initial_puzzle: initial puzzle state
        goal: goal state
    """
    puzzle = importlib.import_module("8puzzle")
    goal = puzzle.EightPuzzle(goal)
    initial = puzzle.EightPuzzle(initial_puzzle)
    h = puzzle.EightPuzzle.tile_switches_remaining
    output = puzzle.EightPuzzle.state_transition
    print(initial.a_star(goal, h, output))


# Solve the puzzle using manhatten_distance
def solve_manhatten(initial_puzzle: str, goal: str = "1 2 3 4 5 6 7 8 0"):
    """
    Solve the puzzle using manhatten_distance
    Args:
        initial_puzzle: initial puzzle state
        goal: goal state
    """
    puzzle = importlib.import_module("8puzzle")
    goal = puzzle.EightPuzzle(goal)
    initial = puzzle.EightPuzzle(initial_puzzle)
    h = puzzle.EightPuzzle.manhatten_distance
    output = puzzle.EightPuzzle.state_transition
    print(initial.a_star(goal, h, output))


# Generate int array of size 9
def generate_array() -> list:
    """
    Generate int array of size 9
    Returns: list of ints from 0 to 8

    """
    return [i for i in range(9)]


# Shuffle array
def shuffle_array(array: list):
    """
    Shuffle array
    Args:
        array: the array to shuffle
    """
    solvable: bool = False
    while not solvable:
        random.shuffle(array)
        solvable = is_solvable(array)


# Check if puzzle is solvable
def is_solvable(array: list) -> bool:
    """
    Check if puzzle is solvable
    Args:
        array: the array to check
    Returns:
        bool: True if solvable, False otherwise
    """
    inversions = 0
    for i in range(8):
        for j in range(i + 1, 9):
            if array[i] > array[j]:
                inversions += 1
    return inversions % 2 == 0


# Generate a random puzzle
def generate_puzzle() -> str:
    """
    Generate a random puzzle
    Returns: The puzzle as a string
    """
    array = generate_array()

    shuffle_array(array)
    return array_to_string(array)


# Convert array to string
def array_to_string(array: list) -> str:
    """
    Convert array to string
    Args:
        array: the array to convert
    Returns:
        str: the array as a string

    """
    res = ""
    for i in range(8):
        res += str(array[i]) + " "
    return res + str(array[8])


main()
