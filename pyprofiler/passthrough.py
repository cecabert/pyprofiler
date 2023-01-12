# coding=utf-8
"""
  @file: passthrough.py
  @data: 17 October 2022
  @author: cecabert

  PassThrough profiler interface
"""
from argparse import ArgumentParser
from pyprofiler.utils import register_profiler
from pyprofiler.profiler import Profiler


@register_profiler(name='passthrough')
class PassThroughProfiler(Profiler):

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        # No arguments required!
        pass

    @classmethod
    def description(cls) -> str:
        return 'NoOp'

    def start(self):
        return super().start()

    def reset(self):
        return super().reset()

    def report(self, stream):
        return super().report(stream)
