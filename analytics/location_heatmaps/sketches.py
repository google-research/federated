import numpy as np
import random
import hashlib

class CountMinSketch(object):
    """
    A non GPU implementation of the count min sketch algorithm.
    """
    def __init__(self, d, w, hash_functions, M=None):
        self.d = d
        self.w = w
        self.hash_functions = hash_functions
        if len(hash_functions) != d:
            raise ValueError("The number of hash functions must match match the depth. (%s, %s)" % (d, len(hash_functions)))
        if M is None:
            self.M = np.zeros([d, w], dtype=np.float64)
        else:
            self.M = M

    def add(self, x, delta=1):
        for i in range(self.d):
            self.M[i][self.hash_functions[i](x) % self.w] += delta

    def batch_add(self, lst):
        pass

    def query(self, x):
        return min([self.M[i][self.hash_functions[i](x) % self.w] for i in range(self.d)])

    def get_matrix(self):
        return self.M



_memomask = {}


def hash_function(n):
    """
    :param n: the index of the hash function
    :return: a generated hash function
    """
    mask = _memomask.get(n)

    if mask is None:
        random.seed(n)
        mask = _memomask[n] = random.getrandbits(32)

    def my_hash(x):
        return hash(str(x) + str(n)) ^ mask

    return my_hash


def gpu_hash_function(j, rand):
    """
    This is a python duplicate of the string hash function used
    in gpu_countminsketch.py
    :param j: the index of the hash function
    :param rand: a list of generated random numbers, must be at least as long as j
    :return: a generated hash function
    """
    def my_hash(s):
        value = rand[j]

        for c in s:
            value = (((value << 5) + value) + ord(c)) % 2**32

        return value

    return my_hash


def get_count_min_sketch(depth=20, width=2000):
    """
    Return count-min sketch.

    Args:
        depth:
        width:

    Returns:

    """
    hash_functions = [hash_function(i) for i in range(depth)]
    sum_sketch = CountMinSketch(depth, width, hash_functions)

    return sum_sketch
