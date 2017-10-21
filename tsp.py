# time: 2017-10-09 19:18:43
# author: Els-y

import random
import queue
import copy

class TSP:
    cities = ['Arad', 'Bucharest', 'Craiova', 'Drobeta', 'Eforie', 'Fagaras', 'Giurgiu',
              'Hirsova', 'Iasi', 'Lugoj', 'Mehadia', 'Neamt', 'Oradea', 'Pitesti',
              'Rimnicu Vilcea', 'Sibiu', 'Timisoara', 'Urziceni', 'Vaslui', 'Zerind']

    class StateNode:
        def __init__(self, current, record, h_cost, g_cost):
            self.current = current
            self.record = record
            self.h_cost = h_cost
            self.g_cost = g_cost
            self.f_cost = h_cost + g_cost

        def is_completed(self):
            for node in self.record:
                if not node.visited:
                    return False
            return True

        def get_visited(self):
            visited = []
            for node in self.record:
                if node.visited:
                    visited.append(node.nid)
            return visited

        def get_unvisited(self):
            visited = []
            for node in self.record:
                if not node.visited:
                    visited.append(node.nid)
            return visited

        def get_path(self):
            path = []

            node = self.record[self.current]
            while True:
                path.insert(0, node.nid)
                if node.prev_nid is None:
                    break
                else:
                    node = self.record[node.prev_nid]

            path.append(node.nid)
            return path

        @staticmethod
        def calc_path(path, graph):
            begin = path[0]
            distance = graph[path[-1]][begin]

            for nid in path[1:]:
                distance += graph[begin][nid]
                begin = nid

            return distance

        def __lt__(self, other):
            return self.f_cost < other.f_cost

        def __le__(self, other):
            return self.f_cost <= other.f_cost

        def __gt__(self, other):
            return self.f_cost > other.f_cost

        def __ge__(self, other):
            return self.f_cost >= other.f_cost

        def __repr__(self):
            return "{" + str(self.current) + ", " + str(self.h_cost) + ", " + str(self.record) + "}"

    class CityNode:
        def __init__(self, nid, name, location, prev_nid=None, visited=False):
            self.nid = nid
            self.name = name
            self.location = location
            self.prev_nid = prev_nid
            self.visited = visited

        def __repr__(self):
            return '<' + str(self.nid) + ', ' + str(self.name) + ', ' + str(self.prev_nid) + '>'

    def problem_generator(self, citySize):
        if citySize > len(TSP.cities):
            citySize = len(TSP.cities)

        selected_cities = random.sample(TSP.cities, citySize)
        locations = []

        for i in range(citySize):
            locations.append((random.random(), random.random()))

        return list(zip(selected_cities, locations))

    def generate_graph(self, problem):
        size = len(problem)
        graph = [[0] * size for i in range(size)]

        for i in range(size):
            for j in range(i + 1, size):
                graph[i][j] = graph[j][i] = self.calc_distance(problem[i][1], problem[j][1])

        return graph

    def calc_distance(self, loc_a, loc_b):
        return ((loc_a[0] - loc_b[0]) ** 2 + (loc_a[1] - loc_b[1]) ** 2) ** 0.5

    def a_star(self, problem, start_city=0):
        graph = self.generate_graph(problem)

        record = self.init_record(problem, start_city)
        h_cost = self.estimate_h_cost(start_city, start_city, [start_city], graph)
        start_state = TSP.StateNode(start_city, record, h_cost, 0)

        pri_queue = queue.PriorityQueue()
        pri_queue.put(start_state)

        while not pri_queue.empty():
            state_node = pri_queue.get()

            if state_node.is_completed():
                break

            visited = state_node.get_visited().copy()
            unvisited = set(range(len(problem))) - set(visited)

            for nid in unvisited:
                next_record = copy.deepcopy(state_node.record)
                next_record[nid].prev_nid = state_node.current
                next_record[nid].visited = True

                h_cost = self.estimate_h_cost(nid, state_node.current, visited, graph)
                g_cost = state_node.g_cost + graph[state_node.current][nid]

                next_state_node = TSP.StateNode(nid, next_record, h_cost, g_cost)

                pri_queue.put(next_state_node)

        return state_node

    def init_record(self, problem, start):
        record = []

        for nid, city in enumerate(problem):
            if nid == start:
                record.append(TSP.CityNode(nid, city[0], city[1], None, True))
            else:
                record.append(TSP.CityNode(nid, city[0], city[1], None))

        return record

    def nearest_unvisited_dist(self, current, visited, graph):
        nearest_dist = 0

        unvisited = set(range(len(graph))) - set(visited)

        for i in unvisited:
            if nearest_dist == -1 or graph[current][i] < nearest_dist:
                nearest_dist = graph[current][i]

        return nearest_dist

    def estimate_h_cost(self, start, current, visited, graph):
        mst = MST(graph, visited)
        mst_record = mst.solve(current)
        mst_distance = mst.calc_cost(mst_record)

        nearest_distance = self.nearest_unvisited_dist(current, visited, graph)
        nearest_distance_from_start = self.nearest_unvisited_dist(start, visited, graph)

        return mst_distance + nearest_distance + nearest_distance_from_start

    def estimate_g_cost(self, record, graph):
        g_cost = 0

        for node in record:
            if node.prev_nid is not None:
                g_cost += graph[node.nid][node.prev_nid]

        return g_cost

class MST:
    class Node:
        def __init__(self, nid, prev=None, dist=-1, visited=False):
            self.nid = nid
            self.prev = prev
            self.dist = dist
            self.visited = visited

        def __repr__(self):
            return '<nid: ' + str(self.nid) + ', prev: ' + str(self.prev) + ', dist: ' + str(self.dist) + '>'

    def __init__(self, graph, visited):
        self.graph = graph
        self.visited = set(visited)
        self.unvisited = set(range(len(graph))) - self.visited

    def solve(self, start):
        record = self.init_record(start)

        while True:
            nearest_node = self.get_nearest_unvisited(record)

            if nearest_node is None:
                break

            nearest_node.visited = True

            for node in record:
                if not node.visited and node.dist > self.graph[nearest_node.nid][node.nid]:
                    node.prev = nearest_node.nid
                    node.dist = self.graph[nearest_node.nid][node.nid]

        return record

    def init_record(self, start):
        init_record = []

        for nid in self.unvisited:
            if nid == start:
                init_record.append(MST.Node(nid, None, self.graph[start][nid]))
            else:
                init_record.append(MST.Node(nid, start, self.graph[start][nid]))

        return init_record

    def get_nearest_unvisited(self, record):
        nearest_node = None

        for node in record:
            if not node.visited and (nearest_node is None or node.dist < nearest_node.dist):
                nearest_node = node

        return nearest_node

    def calc_cost(self, record):
        total_dist = 0
        for node in record:
            total_dist += node.dist

        return total_dist

    def print_graph(self):
        for line in self.graph:
            for dist in line:
                print(dist, end=' ')
            print()

def print_graph(graph):
    for line in graph:
        for dist in line:
            print(dist, end=' ')
        print()

def test_correct_rate(size, times, start):
    tsp = TSP()
    correct_count = 0

    for i in range(times):
        problem = tsp.problem_generator(size)
        finish_state = tsp.a_star(problem, start)
        graph = tsp.generate_graph(problem)

        a_path = finish_state.get_path()
        a_dist = finish_state.calc_path(a_path, graph)

        min_dist, min_path = get_actual_ans(finish_state, graph, start)

        if min_dist >= a_dist:
            correct_count += 1

    return correct_count / times

def get_actual_ans(finish_state, graph, start):
    min_path = []
    min_dist = None

    nid_arr = list(range(len(graph)))
    nid_arr.remove(start)

    import itertools
    for p in itertools.permutations(nid_arr, len(nid_arr)):
        path = [start]
        path.extend(p)

        dist = finish_state.calc_path(path, graph)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            min_path = path

    return (min_dist, min_path)

def main():
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

if __name__ == "__main__":
    main()