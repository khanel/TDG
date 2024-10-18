class TSP:
    def __init__(self, graph, path):
        self.cities_graph = graph
        self.path = path



    def calculate_path_distance(self, path):
        distance = 0
        if path[0] != path[-1]:
            path.append(path[0])

        for i in range(len(path) - 1):
            distance += self.cities_graph.weights[path[i]][path[i + 1]]
        return distance
    
    def add_city(self, cities_distance_weights):
        self.graph.add_edge(cities_distance_weights)

class Graph:
    def __init__(self, weights):
        self.weights = weights
        self.vertices = [i+1 for i in range(len(weights))]
        self.edges = []
        self.calculate_edges(weights)

    def calculate_edges(self, weights, first_index=0):
        for i in range(first_index, len(self.vertices)):
            for j in range(len(self.vertices)):
                if i != j and weights[i][j] != 0:
                    self.edges.append((i+1, j+1, weights[i][j]))
        return self.edges

    def add_edge(self, weight):
        self.vertices.append(len(self.vertices) + 1)
        for i, j in enumerate(weight):
                if weight[i] != 0:
                    self.edges.append((i+1, len(self.vertices), weight[i]))
                    self.edges.append((len(self.vertices), i+1, weight[i]))
        self.edges

