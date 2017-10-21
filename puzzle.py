import random
import copy

class PuzzleNode:
    def __init__(self, node_state, parent=None, depth=0):
        self.node_state = node_state
        self.parent = parent
        self.depth = depth
        self.dimension = int((len(node_state) - 1)**0.5)
        self.value = 0
        if self.dimension ** 2 + 1 != len(node_state):
            raise ValueError(
                "Puzzle must be square and "
                "node_state[0] is the index of the empty tile.")

    def copy(self):
        return PuzzleNode(self.node_state.copy(), self.parent, self.depth)

    def equal(self, other):
        for i in range(len(self.node_state)):
            if self.node_state[i] != other.node_state[i]:
                return False
        return True

    def can_move(self):
        # up, down, left, right
        movable = [False] * 4

        if self.node_state[0] > self.dimension:
            movable[0] = True
        if self.node_state[0] <= self.dimension * (self.dimension - 1):
            movable[1] = True
        if self.node_state[0] % self.dimension != 1:
            movable[2] = True
        if self.node_state[0] % self.dimension != 0:
            movable[3] = True

        return movable

    def move(self, direction):
        if direction == 0:
            return self.move_up()
        elif direction == 1:
            return self.move_down()
        elif direction == 2:
            return self.move_left()
        elif direction == 3:
            return self.move_right()
        else:
            return False

    def move_up(self):
        empty_pos = self.node_state[0]
        empty_value = self.node_state[empty_pos]

        if self.node_state[0] > self.dimension:
            self.parent = self.copy()
            self.depth += 1
            self.node_state[empty_pos] = self.node_state[empty_pos - self.dimension]
            self.node_state[empty_pos - self.dimension] = empty_value
            self.node_state[0] = empty_pos - self.dimension

            return True
        else:
            return False

    def move_down(self):
        empty_pos = self.node_state[0]
        empty_value = self.node_state[empty_pos]

        if self.node_state[0] <= self.dimension * (self.dimension - 1):
            self.parent = self.copy()
            self.depth += 1
            self.node_state[empty_pos] = self.node_state[empty_pos + self.dimension]
            self.node_state[empty_pos + self.dimension] = empty_value
            self.node_state[0] = empty_pos + self.dimension

            return True
        else:
            return False


    def move_left(self):
        empty_pos = self.node_state[0]
        empty_value = self.node_state[empty_pos]

        if self.node_state[0] % self.dimension != 1:
            self.parent = self.copy()
            self.depth += 1
            self.node_state[empty_pos] = self.node_state[empty_pos - 1]
            self.node_state[empty_pos - 1] = empty_value
            self.node_state[0] = empty_pos - 1

            return True
        else:
            return False

    def move_right(self):
        empty_pos = self.node_state[0]
        empty_value = self.node_state[empty_pos]

        if self.node_state[0] % self.dimension != 0:
            self.parent = self.copy()
            self.depth += 1
            self.node_state[empty_pos] = self.node_state[empty_pos + 1]
            self.node_state[empty_pos + 1] = empty_value
            self.node_state[0] = empty_pos + 1

            return True
        else:
            return False

    def scatter(self, step=20):
        node = self.copy()
        last_dirc = None

        for i in range(step):
            movable_dirc = []
            for dirc, can in enumerate(node.can_move()):
                if can:
                    movable_dirc.append(dirc)

            next_dirc = movable_dirc[random.randint(0, len(movable_dirc) - 1)]
            while last_dirc is not None and (last_dirc + next_dirc == 1 or last_dirc + next_dirc == 5):
                next_dirc = movable_dirc[random.randint(0, len(movable_dirc) - 1)]

            node.move(next_dirc)
            last_dirc = next_dirc

        node.parent = None
        node.depth = 0

        return node

    def __repr__(self):
        return "{" + str(self.depth) + ", " + str(self.node_state) + "}"

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

class Puzzle:
    MANHATTAN = 0
    MISPLACED = 1
    XY_HEURISTICS = 2
    LINEAR_CONFLICT = 3

    def __init__(self, begin_node, end_node, h_func=0):
        self.begin_node = begin_node
        self.end_node = end_node
        self.current_node = begin_node
        self.open_list = []
        self.close_list = []
        self.is_completed = False
        self.search_node_num = 0
        self.h_func = h_func

    def find_follow_nodes(self):
        follow_nodes = []

        for i in range(4):
            node = self.current_node.copy()
            if node.move(i) and not self.contains(node):
                follow_nodes.append(node)

        return follow_nodes

    def contains(self, target_node):
        for node in self.open_list:
            if node.equal(target_node):
                return True

        for node in self.close_list:
            if node.equal(target_node):
                return True

        return False

    def a_star(self):
        self.open_list.append(self.begin_node)

        while len(self.open_list) != 0:
            self.current_node = self.open_list.pop(0)

            if self.current_node.equal(self.end_node):
                self.is_completed = True
                break

            self.close_list.append(self.current_node)
            self.search_node_num += 1

            follow_nodes = self.find_follow_nodes()
            for node in follow_nodes:
                self.insert_to_open(node)

    def insert_to_open(self, target_node):
        self.estimate_value(target_node)
        for index, node in enumerate(self.open_list):
            if target_node.value < node.value:
                self.open_list.insert(index, target_node)
                return

        self.open_list.append(target_node)

    def estimate_value(self, node):
        g_cost = node.depth
        h_cost = 0

        if self.h_func == Puzzle.MANHATTAN:
            # Manhattan
            h_cost = self.get_manhattan(node)
        elif self.h_func == Puzzle.MISPLACED:
            # Misplaced
            h_cost = self.get_misplaced(node)
        elif self.h_func == Puzzle.XY_HEURISTICS:
            # XY_HEURISTICS
            h_cost = self.get_xy_heuristics(node)
        elif self.h_func == Puzzle.LINEAR_CONFLICT:
            # Linear Conflict
            h_cost = self.get_linear_conflict(node)

        node.value = h_cost + g_cost

    def get_manhattan(self, node):
        value = 0

        for index in range(1, node.dimension * node.dimension + 1):
            if index != node.node_state[0]:
                node_row = (index - 1) / node.dimension
                node_col = (index - 1) % node.dimension

                dst_index = self.find_val_pos(node.node_state[index], self.end_node)
                dst_row = (dst_index - 1) / node.dimension
                dst_col = (dst_index - 1) % node.dimension

                value += int(abs(node_row - dst_row)) + int(abs(node_col - dst_col))

        return value

    def get_misplaced(self, node):
        value = 0

        for index in range(1, node.dimension * node.dimension + 1):
            if index != node.node_state[0] and node.node_state[index] != self.end_node.node_state[index]:
                value += 1

        return value

    def get_xy_heuristics(self, node):
        # X-Y heuristic
        # https://heuristicswiki.wikispaces.com/X-Y
        x_queue = [XY_Node(node.node_state)]
        y_queue = [XY_Node(node.node_state)]
        target_node = XY_Node(self.end_node.node_state)

        # X
        while len(x_queue) != 0:
            row_node = x_queue.pop(0)

            if row_node.row_equal(target_node):
                break

            follow_nodes = row_node.find_row_adjacent()
            for node in follow_nodes:
                x_queue.append(node)

        # Y
        while len(y_queue) != 0:
            col_node = y_queue.pop(0)

            if col_node.col_equal(target_node):
                break

            follow_nodes = col_node.find_col_adjacent()
            for node in follow_nodes:
                y_queue.append(node)

        return row_node.depth + col_node.depth

    def get_linear_conflict(self, node):
        # Linear Conflict
        # https://heuristicswiki.wikispaces.com/Linear+Conflict
        conflict = 0

        # row-conflict
        for row in range(node.dimension):
            for i in range(1, node.dimension):
                cur_i_pos = row * node.dimension + i
                end_i_pos = self.find_val_pos(node.node_state[cur_i_pos], self.end_node)
                if (end_i_pos - 1) // node.dimension == row:
                    for j in range(i + 1, node.dimension + 1):
                        cur_j_pos = row * node.dimension + j
                        end_j_pos = self.find_val_pos(node.node_state[cur_j_pos], self.end_node)
                        if (end_j_pos - 1) // node.dimension == row and end_j_pos < end_i_pos:
                            conflict += 1

        # col-conflict
        for col in range(node.dimension):
            for i in range(node.dimension - 1):
                cur_i_pos = col + 1 + i * node.dimension
                end_i_pos = self.find_val_pos(node.node_state[cur_i_pos], self.end_node)
                if (end_i_pos - 1) % node.dimension == col:
                    for j in range(i + 1, node.dimension):
                        cur_j_pos = col + 1 + j * node.dimension
                        end_j_pos = self.find_val_pos(node.node_state[cur_j_pos], self.end_node)
                        if (end_j_pos - 1) % node.dimension == col and end_j_pos < end_i_pos:
                            conflict += 1

        return 2 * conflict + self.get_manhattan(node)

    def find_val_pos(self, target_val, node):
        for index in range(1, len(node.node_state)):
            if node.node_state[index] == target_val:
                return index

        return None

    def get_solution_path(self):
        path = []

        node = self.current_node
        while node is not None:
            path.insert(0, node)
            node = node.parent

        return path

    def solution_path_str(self):
        if self.is_completed:
            path_str = "Begin->"
            path = self.get_solution_path()
            for node in path:
                path_str += str(node) + "->"
            path_str += "End"
        else:
            path_str = "Not Completed"

        return path_str

    def print_solution_matrix(self):
        count = 0
        if self.is_completed:
            path = self.get_solution_path()
            for node in path:
                for index, val in enumerate(node.node_state):
                    if index == 0:
                        continue
                    count += 1
                    if index != node.node_state[0]:
                        print("%2d" % val, end=' ')
                    else:
                        print('  ', end=' ')
                    if count % node.dimension == 0:
                        print()
                print()

    def print_result(self):
        print("Search Node Number: " + str(self.search_node_num))
        print(self.solution_path_str())
        # self.print_solution_matrix()

class XY_Node:

    def __init__(self, node_state, depth=0):
        self.node_state = node_state
        self.depth = depth
        self.dimension = 3
        self.row_adjacent = self.get_row_adjacent()
        self.row_empty_pos = (self.node_state[0] - 1) // 3
        self.col_adjacent = self.get_col_adjacent()
        self.col_empty_pos = (self.node_state[0] - 1) % 3
        self.last_row_val = None
        self.last_col_val = None

    def copy(self):
        node = XY_Node(self.node_state)
        node.depth = self.depth
        node.row_adjacent = copy.deepcopy(self.row_adjacent)
        node.row_empty_pos = self.row_empty_pos
        node.col_adjacent = copy.deepcopy(self.col_adjacent)
        node.col_empty_pos = self.col_empty_pos

        return node

    def get_row_adjacent(self):
        row_adjacent = []

        for index in range(0, self.dimension):
            row_adjacent.append(set(self.node_state[index * 3 + 1: index * 3 + self.dimension + 1]))

        return row_adjacent

    def get_col_adjacent(self):
        col_adjacent = [set(), set(), set()]

        for index in range(1, len(self.node_state)):
            if index % 3 == 1:
                col_adjacent[0].add(self.node_state[index])
            elif index % 3 == 2:
                col_adjacent[1].add(self.node_state[index])
            else:
                col_adjacent[2].add(self.node_state[index])

        return col_adjacent

    def find_row_adjacent(self):
        if self.row_empty_pos == 1:
            row_index = [0, 2]
        else:
            row_index = [1]

        row_nodes = []
        for index in row_index:
            for val in self.row_adjacent[index]:
                if val != self.last_row_val:
                    row_nodes.append(self.row_move(index, val))

        return row_nodes

    def row_move(self, index, val):
        next_node = self.copy()
        row_adjacent = next_node.row_adjacent
        row_adjacent[next_node.row_empty_pos].remove(0)
        row_adjacent[next_node.row_empty_pos].add(val)
        row_adjacent[index].remove(val)
        row_adjacent[index].add(0)
        next_node.row_empty_pos = index
        next_node.depth = self.depth + 1
        next_node.last_row_val = val

        return next_node

    def find_col_adjacent(self):
        if self.col_empty_pos == 1:
            col_index = [0, 2]
        else:
            col_index = [1]

        col_nodes = []
        for index in col_index:
            for val in self.col_adjacent[index]:
                if val != self.last_col_val:
                    col_nodes.append(self.col_move(index, val))

        return col_nodes

    def col_move(self, index, val):
        next_node = self.copy()
        col_adjacent = next_node.col_adjacent
        col_adjacent[next_node.col_empty_pos].remove(0)
        col_adjacent[next_node.col_empty_pos].add(val)
        col_adjacent[index].remove(val)
        col_adjacent[index].add(0)
        next_node.col_empty_pos = index
        next_node.depth = self.depth + 1
        next_node.last_col_val = val

        return next_node

    def row_equal(self, other):
        for i in range(len(self.row_adjacent)):
            if self.row_adjacent[i] != other.row_adjacent[i]:
                return False

        return True

    def col_equal(self, other):
        for i in range(len(self.col_adjacent)):
            if self.col_adjacent[i] != other.col_adjacent[i]:
                return False

        return True

    def __repr__(self):
        return "<" + str(self.row_adjacent) + ", " + str(self.col_adjacent) + ">"

def test_average(end_node, scatter_step, times, h_func):
    # Manhattan 15step 200times ~= 100.675
    #           15step 1000times ~= 557.705
    #           20step 100times ~= 414.43
    # Misplaced 15step 200times ~= 304.875
    # X-Y heuristic 15step 1000times ~= 46.791
    total_search = 0

    for i in range(times):
        begin_node = end_node.scatter(scatter_step)
        p = Puzzle(begin_node, end_node, h_func)
        p.a_star()

        total_search += p.search_node_num

    return total_search / times

def main():
    begin_node = PuzzleNode([5, 1, 5, 2, 7, 0, 4, 6, 3, 8])
    end_node = PuzzleNode([9, 1, 2, 3, 4, 5, 6, 7, 8, 0])

    print("Manhattan:")
    p_manhattan = Puzzle(begin_node, end_node, Puzzle.MANHATTAN)
    p_manhattan.a_star()
    p_manhattan.print_result()

    print("Misplaced:")
    p_misplaced = Puzzle(begin_node, end_node, Puzzle.MISPLACED)
    p_misplaced.a_star()
    p_misplaced.print_result()

    print("X-Y Heuristic:")
    p_xy_heuristics = Puzzle(begin_node, end_node, Puzzle.XY_HEURISTICS)
    p_xy_heuristics.a_star()
    p_xy_heuristics.print_result()

    print("Linear Conflict:")
    p_linear_conflict = Puzzle(begin_node, end_node, Puzzle.LINEAR_CONFLICT)
    p_linear_conflict.a_star()
    p_linear_conflict.print_result()

if __name__ == "__main__":
    main()
