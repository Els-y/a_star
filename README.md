# A* Algorithm

Two applications of the A* algorithm:
* N-Puzzle Game
* Traveling Salesman Problem, TSP

## N-Puzzle Game
The code in `puzzle.py` has realized four heuristics:
1. Manhattan distance
2. The number of misplaced tiles
3. X-Y heuristics
4. Linear Conflict

The code can be used as follows, and you can replace the third argument of the class `Puzzle` instantiation to use different heuristics to estimate cost.

The four alternative values are `[Puzzle.MANHATTAN, Puzzle.MISPLACED, Puzzle.XY_HEURISTICS, Puzzle.LINEAR_CONFLICT]`
```python
begin_node = PuzzleNode([5, 1, 5, 2, 7, 0, 4, 6, 3, 8])
end_node = PuzzleNode([9, 1, 2, 3, 4, 5, 6, 7, 8, 0])

print("Manhattan:")
p_manhattan = Puzzle(begin_node, end_node, Puzzle.MANHATTAN)
p_manhattan.a_star()
p_manhattan.print_result()
```

The *list* argument used in `PuzzleNode` instantiation represents the node state, in which the first element is the index of the empty tile. For the above node `begin_node`, it can be represents as follows, 

```
1 5 2
7   4
6 3 8
```

**Note:** If you use X-Y heuristics, it may take a long time to get the results because of calling two bfs search to estimate cost. However, you can rewrite this code in C/C++, which will greatly reduce the execution time.

## Traveling Salesman Problem, TSP
The heuristic function is derived from [CSE 471/598 Project I](http://www.public.asu.edu/~huanliu/AI04S/project1.htm)

* Initial State: Agent in the start city and has not visited any other city
* Goal State: Agent has visited all the cities and reached the start city again
* Successor Function: Generates all cities that have not yet visited
* Edge-cost: distance between the cities represented by the nodes, use this cost to calculate g(n).
* h(n): distance to the nearest unvisited city from the current city + estimated distance to travel all the unvisited cities (MST heuristic used here) + nearest distance from an unvisited city to the start city.

The usage of the code `tsp.py` looks like this:
```python
tsp = TSP()
problem = tsp.problem_generator(5)
graph = tsp.generate_graph(problem)

finish_state = tsp.a_star(problem, 0)

path = finish_state.get_path()
dist = finish_state.calc_path(path, graph)

print("Problem:")
print(problem)

print("\nDistance Graph:")
print_graph(graph)

print()
print("answer: " + str(path) + " " + str(dist))
```
