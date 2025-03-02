# Partitioning a Natural Number - Parallel Implementation

## Problem Description
Implement a parallel version of the following (sequential) program, which calculates the number of ways to partition a given natural number \( n \), i.e., to write it as a sum of natural numbers no greater than \( n \).

### Example 1
For \( n = 5, m = 5 \), 5 can be partitioned in 7 different ways:

```
5
4 + 1
3 + 2
3 + 1 + 1
2 + 2 + 1
2 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1
```

### Example 2
If we limit the components to \( \leq 3 \), we get 5 ways (\( n = 5, m = 3 \)):

```
3 + 2
3 + 1 + 1
2 + 2 + 1
2 + 1 + 1 + 1
1 + 1 + 1 + 1 + 1
```

## Requirements
The parallel version of the method `PartitionsCounter.countPartitions()` should be placed in the `Task` class implementing the `RecursiveTask<Long>` interface.

## Hint
To achieve good speed-ups and efficient task execution, tasks should not be too small. The size of a task should be understood as the product of the parameters \( n \) and \( m \). For small tasks, they should be executed sequentially. Otherwise, the task should be split into subtasks that can be executed in parallel.

## Example Output
For \( n = 110, m = 110 \), the following output is obtained when executed on an 8-core CPU:

```
The answer is: 607163746
Computation time: 3.8 sec.
The answer is: 607163746
Computation time: 0.5 sec.
```
