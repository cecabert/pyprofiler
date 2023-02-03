# coding=utf-8
"""
  @file: stopwatch.py
  @data: 19 October 2022
  @author: cecabert


  StopWatch profiler measuring execution time
"""
from argparse import ArgumentParser
from time import time
from typing import Callable
from pyprofiler.utils import register_profiler
from pyprofiler.profiler import Profiler


_units = {'us': 1e6,
          'ms': 1e3,
          's': 1.0}


@register_profiler(name='stopwatch')
class StopWatchProfiler(Profiler):
    """ Simple StopWatch profiler measuring execution time """

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        parser.add_argument('--units',
                            type=str,
                            default='ms',
                            help='Measuring time units, one of `{}`'
                            .format('`, `'.join(_units.keys())))

    @classmethod
    def description(cls) -> str:
        return 'Simple execution time measurements'

    def __init__(self, units: str = 'ms'):
        if units not in _units.keys():
            raise ValueError('Unsupported type of units, must be one of `{}`'
                             .format('`, `'.join(_units.keys())))
        self._units = units
        self._times = []
        self._func_name = None

    def start(self):
        self._times = [('start', time())]

    def reset(self):
        self._times.append(('end', time()))

    def trace(self, name: str):
        self._times.append((name, time()))

    def __call__(self, func: Callable):
        self._func_name = func.__name__
        return super().__call__(func)

    def report(self, stream):
        # Header
        template = '{0:<6} {1:<57} {2:>12}'
        hdr = template.format('Index',
                              'Trace Name',
                              'Delta Time [{}]'.format(self._units))
        stream.write('Function name: ' + self._func_name)
        stream.write(hdr)
        stream.write('=' * len(hdr))
        # Traces
        scale = _units[self._units]
        for k, (name, curr_t) in enumerate(self._times):
            if k == 0:
                dt = 0.0
            else:
                dt = (curr_t - self._times[0][1]) * scale
            line = template.format(k, name, '{:5.1f}'.format(dt))
            stream.write(line)
        return super().report(stream)
