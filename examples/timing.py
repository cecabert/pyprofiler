# coding=utf-8
"""
  @file: timing.py
  @data: 16 January 2023
  @author: cecabert

  Time profiling samples
"""
from typing import List
from multiprocessing import Process
import numpy as np
from pyprofiler.profiler import ProfilerCase
from pyprofiler.utils.decorator import accept_profiler

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError(f'Parameter `n` can not be negative, got {n}')
    # Initial conditions
    elif n == 0:
        return 0
    elif n in (1, 2):
        return 1
    # Recursion
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def prime_number(n: int) -> List[int]:

    def isprime(num: int) -> bool:
        if num == 0 or num == 1:
            return False
        for x in range(2, num):
            if num % x == 0:
                return False
        else:
            return True

    return list(filter(isprime, range(0, n)))


def bubble_sort(array):
    n = len(array)
    for i in range(n):
        sorted_item = True
        for j in range(n - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                sorted_item = False
        if sorted_item:
            break
    return array


def _run(processes, func):
    for p in processes:
        fn = getattr(p, func)
        fn()


class TimeProfilingCases(ProfilerCase):

    # Select which profiler is suited for this profile scope
    @accept_profiler(['stopwatch', 'viztracer'])
    def profile_single_process(self):
        # Add trace in the profile if profile support it, otherwise it will just
        # ignore it
        self.trace('Check first 100 numbers for prime number')
        # Call function to be profiled by a single profiler
        prime_number(100)
        # Call another time consuming functions
        self.trace('Fibonnaci with `n`=20')
        fibonacci(20)

    @accept_profiler(['cprofile', 'stopwatch'])
    def profile_sort(self):

        data1 = np.random.randint(1, 100000, size=10000)
        data2 = np.random.randint(1, 100000, size=5000)
        self.trace('Generated data')

        bubble_sort(data1)
        self.trace('Full array sorted')

        bubble_sort(data2)
        self.trace('Half array sorted')

    @accept_profiler('viztracer')
    def profile_multiprocess(self):

        # Spawn two process executing our function to be profiled
        kw = {0: (10,), 1: (20,)}
        processes = [Process(target=fibonacci,
                             args=kw[k],
                             name=f'worker_{k}_a') for k in range(2)]
        _run(processes, 'start')
        _run(processes, 'join')
