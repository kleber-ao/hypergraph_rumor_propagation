# Headers
import numpy as np
import scipy.sparse as sp

# The following class implements random number generation via static arrays to avoid single calls to numpy's default RNG
class Random_number_generator():
    pointer = 0
    counter = 0
    def __init__(self, seed, size):
        self.rng = np.random.default_rng(seed)
        self.barn = self.rng.random(size) # BARN stands for Big Array of Random Numbers
        self.barn_size = size
    def _regenerate(self):
        self.barn = self.rng.random(self.barn_size)
        self.pointer = 0
        self.counter += 1 # counts how many times the BARN was regenerated
        print('Array of random numbers already regenerated for ' + str(self.counter) + ' times.')
    def get(self):
        if self.pointer < len(self.barn):
            random_float = self.barn[self.pointer]
            self.pointer += 1 # whenever a random number is used, move the pointer to the next number
            return random_float
        else:
            self._regenerate()
            random_float = self.barn[self.pointer]
            self.pointer += 1
            return random_float

# Inverse transform sampling from uniformly random numbers to exponentially distributed random numbers
def unif_to_exp(x,rate):
    return (-(1/rate)*np.log(x))

# Incidence matrices are iterated through using two dictionaries rather than multidimensional arrays
def store_incidence_matrix_into_two_dicts(I):
    edges_incident_to_node = {}
    nodes_which_belong_to_hyperedge = {}
    pair_of_row_and_col = list(zip(I.tocoo().row, I.tocoo().col))

    for v,e in pair_of_row_and_col:
        if v not in edges_incident_to_node:
            edges_incident_to_node[v] = [e]
        elif e not in edges_incident_to_node[v]:
            edges_incident_to_node[v].append(e)

        if e not in nodes_which_belong_to_hyperedge:
            nodes_which_belong_to_hyperedge[e] = [v]
        elif v not in nodes_which_belong_to_hyperedge[e]:
            nodes_which_belong_to_hyperedge[e].append(v)
        
    return edges_incident_to_node, nodes_which_belong_to_hyperedge