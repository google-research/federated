import numpy as np
import random


class CountMinSketch(object):
    """
    A non GPU implementation of the count min sketch algorithm.
    """
    def __init__(self, depth, width):
        """ Create a new count-min sketch.

        Args:
            w: width of the sketch
            d: depth of the sketch (number of hash functions)
        """
        self._memomask = dict()
        hash_functions = [self.hash_function(i) for i in range(depth)]
        self.depth = depth
        self.width = width
        self.hash_functions = hash_functions
        self.M = np.zeros([self.depth, self.width], dtype=np.float64)

    def add(self, x, delta=1):
        for i in range(self.depth):
            self.M[i][self.hash_functions[i](x) % self.depth] += delta

    def query(self, x):
        return min([self.M[i][self.hash_functions[i](x) % self.width] for i in range(self.depth)])

    def get_matrix(self):
        return self.M

    def hash_function(self, n):
        """Generate a hash function.

        Args:
            n: the index of the hash function

        Returns:
            A generated hash function
        """
        random.seed(n)
        mask = self._memomask.setdefault(n, random.getrandbits(32))

        def my_hash(x):
            return hash(str(x) + str(n)) ^ mask

        return my_hash
