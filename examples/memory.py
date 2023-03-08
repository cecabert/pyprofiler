# coding=utf-8
"""
  @file: graph.py
  @data: 07 February 2023
  @author: cecabert

  Graph profiling samples
"""
import random
import numpy as np
from pyprofiler.profiler import ProfilerCase
from pyprofiler.utils import accept_profiler


_fake_leak = []


class Image:
    """" Simple memory consumer """

    def __init__(self,
                 height:int,
                 width: int,
                 channels: int = 3,
                 lazy: bool = True):
        self.size = (height, width, channels)
        self.lazy = lazy
        self._array = None
        if not lazy:
            self._array = self._load()

    def _load(self):
        return np.random.normal(size=self.size)

    def __enter__(self) -> np.ndarray:
        return self.array

    def __exit__(self, *args):
        if self.lazy:
            self._array = None

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = self._load()
        return self._array


def memory_consumer(chunk, size=(128, 128)):
    data = []
    for k in range(chunk):
        data.append(np.random.normal(size=size))
    return data


def simulate_memory_leak():
	"""Appends to a global array in order to simulate a memory leak"""
	_fake_leak.append(np.ones(10000, dtype=np.int64))


class MemoryUsage(ProfilerCase):

    @accept_profiler(('memory-profiler-time', 'memory-profiler-line'))
    def profile_simple(self):
        n_image = 50
        im_size = (512, 512, 3)

        self.trace('Images')
        images = []
        for k in range(n_image):
            image = Image(*im_size, lazy=False)
            images.append(image)
            with image as data:
                mean_pixel_value = np.mean(data)

        self.trace('LazyImages')
        images = []
        for k in range(n_image):
            image = Image(*im_size, lazy=True)
            images.append(image)
            with image as data:
                mean_pixel_value = np.mean(data)

        def _lambda_consumer(chunks):
            entries = []
            for _ in range(chunks):
                entries.append(random.randint(0, 1000))
            return images

        # Call locally defined function
        self.trace('Lambda')
        _lambda_consumer(100_000)

        # Call function define within `ProfilerCase` file
        self.trace('ExternalFunction')
        data = memory_consumer(256)
        data = []

    @accept_profiler('tracemalloc')
    def profile_memory_leak(self):

        for k in range(10):
            simulate_memory_leak()
            self.trace(f'Iteration: {k}')

