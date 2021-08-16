import json
import copy
from collections import deque


class Graph(object):

    def __init__(self, from_dict=None):
        if from_dict:
            self._from_dict(from_dict)
        else:
            self._graph = {}

    def __str__(self):
        return json.dumps(self._graph)

    def _reset(self):
        self._graph = {}

    def _from_dict(self, from_dict):
        self._reset()

        if not from_dict:
            raise ValueError('Missed input dict')

        for k in from_dict:
            self.add_node(k)
            for v in from_dict[k]:
                self.add_node(v)
                self.add_edge(k, v)

    def _to_dict(self):
        _dict = {}

        for n in self._graph:
            _dict[n] = copy.deepcopy(list(self._graph[n]))

        return _dict

    def _exist_node(self, node):
        return node in self._graph

    def _exist_edge(self, node, dependent):
        return (node in self._graph) and (dependent in self._graph[node])

    def _validate(self):
        return set(self._graph.keys()) - set(self.dependents)

    def add_node(self, node):
        if not self._exist_node(node):
            self._graph[node] = []

    def add_edge(self, node, dependent):
        if self._exist_edge(node, dependent):
            return

        if node not in self._graph:
            self.add_node(node)
        if dependent not in self._graph:
            self.add_node(dependent)

        self._graph[node].append(dependent)

        if not self._validate():
            raise ValueError('Invalid graph')

    def remove_edge(self, node, dependent):
        if node not in self._graph:
            return

        # remove all same dependent
        while dependent in self._graph[node]:
            self._graph[node].remove(dependent)

    @staticmethod
    def _dedup_cycle(lst):

        def _is_same_cycle(x, y):
            return (len(x) == len(y)) and (set(x).issubset(y))

        dedup_lst = []
        lst.sort(key=len)
        for index, val in enumerate(lst):
            dup = False
            for _item in lst[index + 1:]:
                if _is_same_cycle(val, _item):
                    dup = True
                    break
            if not dup:
                dedup_lst.append(val)

        return dedup_lst

    def find_node_cycles(self, node):
        visited = []
        trace = []
        cycles = []

        def _dfs(node_index):
            nonlocal cycles
            if node_index in trace:
                _cycle = []
                trace_index = trace.index(node_index)
                for i in range(trace_index, len(trace)):
                    _cycle.append(trace[i])
                cycles.append(_cycle)
                return

            visited.append(node_index)
            trace.append(node_index)

            if node_index != '' and node_index in self._graph:
                children = self._graph[node_index]
                for child in children:
                    _dfs(child)
            trace.pop()

        _dfs(node)

        # dedup cycle
        dedup_cycles = self._dedup_cycle(cycles)

        return dedup_cycles

    def find_all_cycles(self):
        cycles = []
        for node in self.nodes:
            cycles.extend(self.find_node_cycles(node))

        # dedup cycle
        dedup_cycles = self._dedup_cycle(cycles)

        return dedup_cycles

    # change self to a dag graph
    def _self_to_dag(self):
        for _v in self.nodes:
            visited = []
            trace = []

            def _dfs(node_index):
                if node_index in trace:
                    # found cycle, remove cycle and stop dfs
                    # print(node_index, trace, visited)
                    # print('remove', trace[-1], node_index)
                    self.remove_edge(trace[-1], node_index)
                    return

                visited.append(node_index)
                trace.append(node_index)

                if node_index != '' and node_index in self._graph:
                    children = self._graph[node_index]
                    for child in children:
                        _dfs(child)
                trace.pop()

            _dfs(_v)

        return self

    # return a new dag graph but not change self
    def get_dag(self):
        dag_graph = Graph(self._to_dict())
        return dag_graph._self_to_dag()

    def get_reversed_graph(self):
        reversed_graph = Graph()

        for node, dependents in self._graph.items():
            for d in dependents:
                reversed_graph.add_edge(d, node)

        return reversed_graph

    def all_downstreams(self, node):
        if (node is None) or (not self._exist_node(node)):
            raise ValueError(f"Invalid node {node}")

        visited = []
        trace = []
        downstreams = []

        def _dfs(node_index):
            if node_index in trace:
                raise ValueError('Has cycle in downstream')

            visited.append(node_index)
            trace.append(node_index)

            if node_index != '' and node_index in self._graph:
                children = self._graph[node_index]
                if not children:
                    downstreams.append(copy.deepcopy(trace))
                else:
                    for child in children:
                        _dfs(child)

            trace.pop()

        _dfs(node)

        return downstreams

    def topological_sort(self):
        in_degree = {}
        for u in self._graph:
            in_degree[u] = 0

        for u in self._graph:
            for v in self._graph[u]:
                in_degree[v] += 1

        queue = deque()
        for u in in_degree:
            if in_degree[u] == 0:
                queue.appendleft(u)

        l = []
        while queue:
            u = queue.pop()
            l.append(u)
            for v in self._graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.appendleft(v)

        if len(l) == len(self._graph):
            return l
        else:
            raise ValueError('graph has cycle')

    def is_dag(self):
        if not self.vertexes:
            return False, 'No vertexes'

        try:
            self.topological_sort()
        except Exception as err:
            return False, str(err)

        return True

    # nodes (all nodes)
    @property
    def nodes(self):
        return self._graph.keys()

    # vertexes  (nodes without upstream)
    @property
    def vertexes(self):
        return list(set(self._graph.keys()) - set(self.dependents))

    # dependents (nodes with upstream)
    @property
    def dependents(self):
        _dependents = set()
        for v in self._graph.values():
            _dependents = _dependents.union(set(v))

        return list(_dependents)

    # leaves (nodes without downstream )
    @property
    def leaves(self):
        _leaves = []

        for node, value in self._graph.items():
            if len(value) == 0:
                _leaves.append(node)

        return _leaves

    @staticmethod
    def dependents_count(steams):
        v = [j for i in steams for j in i]
        return len(set(v).union(v)) - 1


if __name__ == '__main__':

    test_graph = {
        "a": ["b"],
        "b": ["c", "f"],
        "c": ["d"],
        "d": ["e", "b"],
        "e": ["c", "d"],
        "g": ["b"]
    }

    test_graph1 = {
        'A': ['B', 'C', 'D'],
        'B': ['E'],
        'C': ['D', 'F'],
        'D': ['B', 'E', 'G'],
        'E': [],
        'F': ['D', 'G'],
        'G': ['E']
    }

    g = Graph(from_dict=test_graph)
    print(g.is_dag())
    g1 = Graph(from_dict=test_graph1)
    print(g1.is_dag())
    # reversed_graph = g.get_reversed_graph()
    # print(g.vertexes, g.dependents, g.leaves)
    # print(g.find_all_cycles())
    # print(g.all_downstreams('A'))
    # print(g.topological_sort())

    # dag_g = g.get_dag()
    # print(dag_g)
    #
    # for _v in dag_g.nodes:
    #     downstreams = dag_g.all_downstreams(_v)
    #     print(f"{_v}: {Graph.dependents_count(downstreams)}, {downstreams}")

    # print(g)
    # print('Cycle:', g.find_all_cycles())
    # print('DAG:', g.get_dag())
    # print(g.get_dag().all_downstreams('b'))

    # print(reversed_graph)
    # print('Cycle:', reversed_graph.find_all_cycles())
    # print('DAG:', reversed_graph.get_dag())
    # print(reversed_graph.get_dag().all_downstreams('b'))
