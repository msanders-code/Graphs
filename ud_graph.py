# Course: CS 261
# Author: Matt Sanders
# Assignment: 6 - Part 1
# Description: Implementation of an undirected graph utilizing
#              an adjacency list to hold the edges.

from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Add new vertex to the graph without establishing any
        edges. The given vertex can be any string. If the vertex
        is already part of the graph, the method does nothing.
        Returns nothing.
        """

        if v not in self.adj_list:
            self.adj_list[v] = []  # Adds the vertex to the graph if not already in graph
        
    def add_edge(self, u: str, v: str) -> None:
        """
        Adds an edge to the graph that connects from 'u' to 'v' and
        'v' to 'u'. If neither vertex exists in the graph, they are
        created and then connected with an edge; if one exists and
        not the other, the one that doesn't exist is added and then
        they're connected with an edge. If an edge already exists or
        if 'u' = 'v', the method does nothing. Returns nothing.
        """

        if u != v:
            graph = self.adj_list  # Assign graph to current graph

            if u not in graph and v in graph:
                graph[u] = [v]  # Add 'u' to graph and add 'v' to u's value list
                graph[v].append(u)  # Add 'u' to v's value list
            elif v not in graph and u in graph:
                graph[v] = [u]  # Add 'v' to graph and 'u' to v's value list
                graph[u].append(v)  # Add 'v' to u's value list
            elif u not in graph and v not in graph:
                graph[u] = [v]  # Add 'u' to graph and 'v' to u's value list
                graph[v] = [u]  # Add 'v' to graph and 'u' to v's value list
            else:

                if v not in graph[u]:
                    graph[u].append(v)  # Add 'v' to u's value list
                    graph[v].append(u)  # Add 'u' to v's value list

    def remove_edge(self, v: str, u: str) -> None:
        """
        Removes an edge between the given vertices. if either or
        both of the given vertices is not in the graph, the method
        does nothing. If there is no edge between the given vertices,
        the method does nothing. Returns nothing.
        """

        graph = self.adj_list  # Assign graph to the current graph

        if u in graph and v in graph:  # Check if 'u' and 'v' are in the graph

            if v in graph[u]:  # Check for an edge between 'u' and 'v'
                graph[v].remove(u)  # Remove 'u' from v's list
                graph[u].remove(v)  # Remove 'v' from u's list

    def remove_vertex(self, v: str) -> None:
        """
        Removes a vertex and all its connected edges. If the given vertex
        is not in the graph, the method does nothing. Returns nothing.
        """

        graph = self.adj_list  # Assign graph to the current graph

        if v in graph:  # Check if given vertex in graph

            for edge in graph[v]:  # Remove 'v' from the edge set of every vertex in v's edge set
                graph[edge].remove(v)

            graph.pop(v)  # Remove 'v' from the graph

    def get_vertices(self) -> []:
        """
        Returns a list of vertices in the graph, in no particular order.
        """

        return list(self.adj_list)

    def get_edges(self) -> []:
        """
        Returns a list of edges, tuples of incident vertices, in the graph,
        in no particular order.
        """

        graph = self.get_vertices()  # Define a list of vertices
        output = []  # Initialize an empty output list

        for vertex in graph:  # Loop through the list of vertices

            for key in self.adj_list:  # Loop through the vertices in th graph

                if vertex in self.adj_list[key]:  # Check each vertex's list for the vertex from the vertex list

                    if (vertex, key) not in output and (key, vertex) not in output:
                        output.append((vertex, key))  # Add a tuple of vertices to the output list

        return output

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list of vertices and returns true if the list
        is a valid path, false otherwise. An empty list returns
        true and a single vertex returns true.
        """

        graph = self.adj_list  # Assign the graph to 'graph'

        if len(path) == 1:

            if path[0] not in graph:  # If the vertex in path is not in the graph, return False
                return False

        elif len(path) > 1:
            index = 0  # Initialize loop counter

            while index < len(path) - 1:

                if path[index + 1] in graph[path[index]]:  # Check if the next vertex in path is adjacent to current vertex
                    index += 1  # Increment loop counter
                else:
                    return False

        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during DFS search, in the order
        they were visited. If the optional 'v_end' parameter is used,
        the search stops when this vertex is reached. If no end is given
        or the end is not in the graph, it visits every vertex in the
        graph. If the starting vertex is not in the graph, an empty list
        is returned.
        """

        graph = self.adj_list  # Assign vertex dictionary to 'graph'
        v_visited = []  # Initialize empty list of visited vertices

        if v_start in graph:  # Check if starting vertex in graph
            stack = [v_start]  # Initialize a stack with the starting vertex

            while stack:
                curr = stack.pop()  # Pop the last value in the stack
                adjacent_v = sorted(graph[curr], reverse=True)  # Assign curr's reverse sorted list of adjacent vertices to 'adjacent_v'

                if curr not in v_visited: # Check if current vertex not in visited yet
                    v_visited.append(curr)  # Add curr to visited vertex list

                    if curr == v_end:  # Check if we're at the end of the search
                        return v_visited
                    else:
                        stack.extend(adjacent_v)  # Add curr's adjacent vertices to the stack

        return v_visited

    def bfs(self, v_start, v_end=None) -> []:
        """
        Return list of vertices visited during BFS search, in the order
        they were visited. If the optional 'v_end' parameter is used,
        the search stops when this vertex is reached. If no end is given
        or the end is not in the graph, it visits every vertex in the
        graph. If the starting vertex is not in the graph, an empty list
        is returned.
        """

        graph = self.adj_list  # Assign the graph to 'graph'
        v_visited = []  # Initialize an empty list to hold the visited vertices

        if v_start in graph:  # Checks if starting vertex in graph
            queue = deque([v_start])  # Initialize a queue with the starting vertex using imported method

            while queue:
                curr = queue.popleft()  # Remove the first value in the queue
                adjacent_v = sorted(graph[curr])  # Create a new sorted list of curr's adjacent vertices

                if curr not in v_visited:  # Check if current vertex not in visited yet
                    v_visited.append(curr)  # If curr has not been visited add it to the visited vertices

                    if curr == v_end:  # Check if we have reached the given end vertex
                        return v_visited
                    else:

                        for vertex in adjacent_v:  # Loop through curr's adjacent vertices

                            if vertex not in v_visited:  # If the adjacent vertex has not been visited, add it to the queue
                                queue.append(vertex)

        return v_visited

    def count_connected_components(self):
        """
        Returns the number of connected components in the graph.
        """

        graph = self.adj_list  # Assign the graph to 'graph'
        count = 0  # Initialize connected count
        vertices = deque(self.get_vertices())  # Initializes queue of vertices from the graph
        queue = deque([])  # Initialize another queue for traversing the vertices
        visited = []  # Initialize a list to hold visited vertices

        while vertices:
            queue.append(vertices.popleft())  # Remove first graph vertex from the vertex queue

            while queue:  # Process queue until empty
                vertex = queue.popleft()  # Remove first value from queue
                visited.append(vertex)  # Add value removed from queue to the visited list

                if vertex in vertices:
                    vertices.remove(vertex)  # Remove current vertex from the vertex queue

                for adjacent in graph[vertex]:  # Process the vertex's adjacent vertices

                    if adjacent not in queue and adjacent not in visited:
                        queue.append(adjacent)  # Add the vertex to the queue is not already in queue or visited list

            count += 1  # Increment the connected count

        return count

    def has_cycle(self):
        """
        Return True if the graph contains at least one cycle, False otherwise.
        """

        graph = self.adj_list

        for vertex in graph:  # Loops through every vertex in the graph
            index = 0  # Initializes loop counter to zero
            curr = graph[vertex]  # Initializes curr to the current vertex edge list

            while index < len(curr) - 1:
                if curr[index + 1] in graph[curr[index]]:  # Checks if the current vertex in list of next vertex in list
                    return True

                index += 1

            visited = []  # Initializes visited as an empty list
            trail = deque([])  # Initializes trail as an empty queue
            trail.append(vertex)  # Adds the current vertex to queue

            while trail:  # Cycles until the queue is empty
                current_v = trail.popleft()  # Removes the first value from queue and assigns it to current_v

                if not visited:  # If visited is empty, add all values from current vertex's edge list to queue
                    trail.extend(graph[current_v])
                else:

                    for element in graph[current_v]:

                        if element not in visited:  # If the edge vertex is not in visited, add it to the queue
                            trail.append(element)

                if current_v not in visited:  # If the current vertex is not in visited, add it to visited
                    visited.append(current_v)
                else:
                    return True

        return False


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)
    
    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')
    print(g)
    g.remove_vertex('D')
    print(g)
    
    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())

