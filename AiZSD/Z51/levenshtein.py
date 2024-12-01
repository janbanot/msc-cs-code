def initialize_2d_array(m, n):
    array = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        array[i][0] = i
    for j in range(n + 1):
        array[0][j] = j

    return array


def edit_distance_dynamic(word_a, word_b):
    """
    Calculate the edit distance between two strings using dynamic programming.

    The edit distance (Levenshtein distance) is the minimum number of single-character
    operations (insertions, deletions, or substitutions) required to transform word_a
    into word_b.

    The algorithm builds a matrix where cell [i][j] represents the minimum edit distance
    between the first i characters of word_a and the first j characters of word_b.
    """
    m = len(word_a)
    n = len(word_b)

    array = initialize_2d_array(m, n)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            replacement_cost = 0 if word_a[i - 1] == word_b[j - 1] else 1

            replace = array[i - 1][j - 1] + replacement_cost
            delete = array[i - 1][j] + 1
            insert = array[i][j - 1] + 1

            result = min(replace, delete, insert)
            array[i][j] = result

    return array


def edit_distance_recursive(s1, s2):
    """
    Calculate edit distance recursively based on the given formula:
    d(ε,ε) = 0
    d(s,ε) = |s|
    d(ε,s) = |s|
    d(s₁z₁,s₂z₂) = min(
        d(s₁,s₂) + χ(z₁ ≠ z₂),
        d(s₁z₁,s₂) + 1,
        d(s₁,s₂z₂) + 1
    )
    """

    # Base cases
    if len(s1) == 0:  # d(ε,s) = |s|
        return len(s2)
    if len(s2) == 0:  # d(s,ε) = |s|
        return len(s1)

    # Get prefixes (s₁, s₂) and last characters (z₁, z₂)
    s1_prefix = s1[:-1]
    s2_prefix = s2[:-1]
    z1 = s1[-1]
    z2 = s2[-1]

    # Calculate χ(z₁ ≠ z₂)
    replacement_cost = 0 if z1 == z2 else 1

    # Implement the recursive formula
    return min(
        # d(s₁,s₂) + χ(z₁ ≠ z₂) - replacement
        edit_distance_recursive(s1_prefix, s2_prefix) + replacement_cost,
        # d(s₁z₁,s₂) + 1 - deletion
        edit_distance_recursive(s1, s2_prefix) + 1,
        # d(s₁,s₂z₂) + 1 - insertion
        edit_distance_recursive(s1_prefix, s2) + 1,
    )


def edit_distance_memoized(s1, s2, memo=None):
    if memo is None:
        memo = {}

    # Create key for memoization, lengths uiquely identify subproblems
    key = (len(s1), len(s2))

    # If result already calculated, return it
    if key in memo:
        return memo[key]

    # Base cases
    if len(s1) == 0 and len(s2) == 0:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    # Get prefixes and last characters
    s1_prefix = s1[:-1]
    s2_prefix = s2[:-1]
    z1 = s1[-1]
    z2 = s2[-1]

    # Calculate χ(z₁ ≠ z₂)
    replacement_cost = 0 if z1 == z2 else 1

    # Calculate result
    result = min(
        edit_distance_memoized(s1_prefix, s2_prefix, memo) + replacement_cost,
        edit_distance_memoized(s1, s2_prefix, memo) + 1,
        edit_distance_memoized(s1_prefix, s2, memo) + 1,
    )

    # Store result in memo
    memo[key] = result
    return result


if __name__ == "__main__":
    word_a = "kot"
    word_b = "młot"

    array = edit_distance_dynamic(word_a, word_b)
    for row in array:
        print(row)

    distance_recursive = edit_distance_recursive(word_a, word_b)
    distance_memoized = edit_distance_memoized(word_a, word_b)

    # Time complexity: O(n²) - two nested loops iterating over lengths of both strings
    # Space complexity: O(n²) - table of size (|s1|+1) × (|s2|+1)
    print(f"Levensthein Distance dynamic: {array[-1][-1]}")

    # Time complexity: O(3ⁿ) - because of three recursive calls at each step
    # - For each character we make 3 recursive calls, the recursion tree has depth max(|s1|, |s2|) = n
    # - At each level, we branch into 3 new calls -> a tree with 3ⁿ nodes in total
    # Space complexity: O(n) - due to recursion stack (max(|s1|, |s2|))
    print(f"Levensthein Distance recursive: {distance_recursive}")

    # Time complexity: O(n²) - each subproblem solved only once
    # - Each subproblem is defined by two parameters: length of s1 prefix and length of s2 prefix
    # - Total number of possible subproblems is |s1| × |s2|, each subproblem is computed once
    # Space complexity: O(n²) - for memoization table, it's size is |s1| × |s2|
    print(f"Levensthein Distance memoized: {distance_memoized}")
