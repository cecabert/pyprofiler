# coding=utf-8
"""
  @file: cprofile.py
  @data: 13 October 2022
  @author: cecabert

  Interface for `cProfile`
  https://docs.python.org/3/library/profile.html
"""
from argparse import ArgumentParser
from cProfile import Profile as cProfiler
from pstats import Stats
from io import StringIO
from pyprofiler.utils import register_profiler
from pyprofiler.profiler import Profiler



@register_profiler(name='cprofile')
class cProfileProfiler(Profiler):

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        parser.add_argument('--sort',
                            type=str,
                            default='cumulative',
                            help='How stats are sorted')
        parser.add_argument('--strip-dirs',
                            action='store_true',
                            default=False,
                            help='If set, removes all leading path information'
                                 ' from file names.')
        parser.add_argument('--top-k',
                            type=int,
                            default=10,
                            help='Print only the top `k` functions')
        return parser

    @classmethod
    def description(cls) -> str:
        return 'Provides execution time statistics for single process/thread'\
               ' executable'

    def __init__(self,
                 sort: str = 'cumulative',
                 strip_dirs: bool = False,
                 top_k: int = 10):
        self._sort = sort
        self._strip_dirs = strip_dirs
        self._top_k = top_k

    def start(self):
        self._profiler = cProfiler()
        self._profiler.enable()
        return self

    def reset(self):
        self._profiler.disable()

    def report(self, stream):
        # Generate stats
        buffer = StringIO()
        stats = Stats(self._profiler, stream=buffer)
        if self._strip_dirs:
            stats = stats.strip_dirs()
        stats.sort_stats(self._sort).print_stats(self._top_k)
        # Rewind + dump reports
        buffer.seek(0)
        for line in buffer:
            stream.write(line.strip())
