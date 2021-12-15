# Course: CS261 - Data Structures
# Author: Matt Sanders
# Assignment: 6 - Part 2
# Description: Implementation of a directed graph with weighted edges,
#              represented as an adjacency matrix by using lists.

import heapq  # Import functionality for a heap/priority queue
from collections import deque  # Import queue and deque functionality


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Adds a new vertex to the graph and updates the vertex count.
        Returns the vertex count.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'
        self.v_count += 1  # Increment the vertex count

        if not graph:  # If graph is empty, add vertex to graph and update count
            graph.append([0])  # Initialize vertex with list of zero
        else:
            new_vert = []  # Initialize a new empty vertex

            for vertex in graph:  # Add another zero to each vertex in the graph
                vertex.append(0)

            for index in range(self.v_count):  # Populate the new vertex with zeros
                new_vert.append(0)

            graph.append(new_vert)  # Add new vertex to the graph

        return self.v_count  # Vertex count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Adds a new edge to the graph connecting from 'src' to 'dst' with
        given weight. If 'weight' is negative, neither 'src' nor 'dst' is
        in the graph, or 'src' = 'dst', then the method does nothing. If
        the edge already exists, the weight is updated to the new weight.
        Returns nothing.
        """

        graph = self.adj_matrix  # Assigns graph to 'graph'

        if src != dst and weight > 0:  # Check if 'src' and 'dst' are equal and that 'weight' is positive

            if src in range(self.v_count) and dst in range(self.v_count):  # Checks if 'src' and 'dst' are acceptable indices
                graph[src][dst] = weight  # Updates the weight of the edge

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes an edge between the given vertex indices. If
        either vertex does not exist in the graph or no edge
        exists between them, it does nothing. Returns nothing.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'

        if src in range(self.v_count) and dst in range(self.v_count):  # Check if both indices are in the graph

            if graph[src][dst] > 0:  # Check for an existing edge between the vertices
                graph[src][dst] = 0  # Remove the edge

    def get_vertices(self) -> []:
        """
        Returns a list of the vertices in the graph in no
        particular order.
        """

        return list(range(self.v_count))

    def get_edges(self) -> []:
        """
        Returns a list of edges in the graph in no particular
        order. Each edge is returned as a tuple of source vertex,
        destination vertex, and edge weight.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'
        output = []  # Initialize output as empty list

        for index in range(self.v_count):  # Loop through graph vertices

            for second_index in range(self.v_count):  # Loop through the list of edges for each vertex

                # If the edge weight is greater than zero, add a tuple of source, destination, and weight to output
                if graph[index][second_index] > 0:
                    output.append((index, second_index, graph[index][second_index]))

        return output

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list of vertices and returns true if the list represents a valid
        path. Returns false otherwise. If given an empty path, method returns true.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'
        index = 0  # Initialize loop counter

        while index < len(path) - 1:
            curr = graph[path[index]]  # Assign first vertex from path to curr
            next_v = path[index + 1]  # Assign the next vertex from path to next_v

            if curr[next_v] == 0:
                return False  # If there is no path to the next vertex, return false

            index += 1

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Performs a depth-first search in the current graph starting from
        the given v_start and ending at the given v_end; if no end is given,
        the whole graph is searched. Returns a list of visited vertices in
        the order they were visited. If the starting vertex is not in the
        graph, an empty list is returned; if the end vertex is not in the
        graph, the whole graph is searched.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'
        visited = []  # Initialize list of visited vertices as empty
        stack = [v_start]  # Initialize stack with starting vertex

        while stack:
            curr = stack.pop()  # Pop top of stack and assign to curr

            if curr in range(len(graph)):  # Checks if current vertex in the graph
                adjacent_list = graph[curr]  # Assign list of directed edges from current vertex to adjacent_list

                if curr not in visited:
                    visited.append(curr)  # Add curr to visited list if it's not already there

                    if v_end in visited:  # If v_end has been reached, return the list of visited vertices
                        return visited
                    else:

                        for index in range(-1, -len(adjacent_list) - 1, -1):  # Cycle backward through vertex's directed edge list
                            corrected_index = len(adjacent_list) + index  # Calculates the positive index of current position

                            if adjacent_list[corrected_index] > 0:
                                stack.append(corrected_index)  # Adds current index to stack if it is an edge from current vertex

        return visited

    def bfs(self, v_start, v_end=None) -> []:
        """
        Performs a breadth-first search in the current graph starting from
        the given v_start and ending at the given v_end; if no end is given,
        the whole graph is searched. Returns a list of visited vertices in
        the order they were visited. If the starting vertex is not in the
        graph, an empty list is returned; if the end vertex is not in the
        graph, the whole graph is searched.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'
        visited = []  # Initialize visited vertices list as empty
        queue = deque([v_start])  # Initialize queue with the starting vertex

        while queue:
            curr = queue.popleft()  # Dequeue the first value from the queue and assign it to curr

            if curr in range(len(graph)):  # Checks if the current vertex in graph
                adjacent_list = graph[curr]  # Assign list of directed edges from current vertex to adjacent_list

                if curr not in visited:
                    visited.append(curr)  # Adds current vertex to list of visited vertices if it's not already there

                    if v_end in visited:  # If v_end was just added to the visited list, return the list
                        return visited
                    else:

                        for index in range(len(adjacent_list)):  # Cycle forward through the vertex's directed edge list

                            if adjacent_list[index] > 0 and index not in visited:  # Check if there is an edge and if that vertex has been visited
                                queue.append(index)  # Add the vertex to the queue

        return visited

    def has_cycle(self) -> bool:
        """
        Returns true if there is at least one directed cycle in the graph.
        Returns false otherwise.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'

        for index in range(len(graph)):  # Loop through every vertex in graph
            queue = deque([])  # Initialize queue as empty
            visited = []  # Initializes a list of visited vertices as empty
            vertex = graph[index]  # Assign list of edges for current vertex to 'vertex'

            for adjacent_index in range(len(vertex)):  # Add all vertex edge values from edge list to queue if they are greater than zero

                if vertex[adjacent_index] > 0:
                    queue.append(adjacent_index)  # Add vertex value to queue

            while queue:  # Loop while the queue is not empty
                new_vertex = queue.popleft()  # Dequeue the first value from queue and assign it to new_vertex
                visited.append(new_vertex)

                if graph[new_vertex][index] > 0:  # Check if new_vertex completes a cycle
                    return True
                else:

                    for new_index in range(len(graph[new_vertex])):  # Loop through new vertex's edge list

                        if graph[new_vertex][new_index] > 0 and new_index not in visited:
                            queue.append(new_index)  # Add next vertex to queue if edge value is greater than zero

        return False

    def dijkstra(self, src: int) -> []:
        """
        Implements dijkstra's algorithm to compute the shorted distance
        from given source vertex to all other vertices in the graph.
        Returns a list with one value per vertex of shortest path to
        that vertex from source vertex; if a vertex isn't reachable,
        it's value is infinity.
        """

        graph = self.adj_matrix  # Assign graph to 'graph'
        output = []  # Initialize output as an empty list
        visited = dict()  # Initialize visited as an empty dictionary
        pqueue = [(src, 0)]  # Initialize 'pqueue' as a list with a tuple of source vertex and distance to itself
        heapq.heapify(pqueue)  # Converts the 'pqueue' from a list to a priority queue

        while pqueue:  # Loop while priority queue is not empty
            vertex = heapq.heappop(pqueue)  # Pop lowest priority value from queue and assign it to vertex
            distance = vertex[1]  # Assign the second value from 'vertex' tuple to distance

            if vertex[0] not in visited or vertex[1] < visited[vertex[0]]:
                visited[vertex[0]] = vertex[1]  # Add vertex to visited with key = vertex name, value = min distance

                for index in range(len(graph[vertex[0]])):  # Loop through the vertex's edge list

                    if graph[vertex[0]][index] > 0:  # Check if a directed edge exists
                        v_distance = distance + graph[vertex[0]][index]  # Calculate total distance to vertex
                        heapq.heappush(pqueue, (index, v_distance))  # Add vertex to heap as a tuple, (vertex, distance)

        for element in range(len(graph)):

            if element not in visited:
                output.append(float("inf"))  # Add infinity to output if vertex was unreachable
            else:
                output.append(visited[element])  # Add the vertex distance to output

        return output


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)

    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')
    
    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0, 99)]
    for src, dst, *weight in edges_to_add:
        g.add_edge(src, dst, *weight)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)

    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
