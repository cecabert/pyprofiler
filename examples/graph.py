# coding=utf-8
"""
  @file: graph.py
  @data: 06 February 2023
  @author: cecabert

  Graph profiling samples
"""
import re
from pyprofiler.profiler import ProfilerCase
from pyprofiler.utils.decorator import accept_profiler


def compile():
    return re.compile('^[abetors]*$')


def match(reo):
    [reo.match(a) for a in words()]


def words():
    return ['abbreviation',
            'abbreviations',
            'abettor',
            'abettors',
            'abilities',
            'ability',
            'abrasion',
            'abrasions',
            'abrasive',
            'abrasives']


class Banana:

    def eat(self):
        pass


class Person:

    def __init__(self):
        self.no_bananas()

    def no_bananas(self):
        self.bananas = []

    def add_banana(self, banana):
        self.bananas.append(banana)

    def eat_bananas(self):
        [banana.eat() for banana in self.bananas]
        self.no_bananas()

class GraphCases(ProfilerCase):

    @accept_profiler('callgraph')
    def profile_execution_graph(self):
        # Compile
        reo = compile()
        match(reo=reo)

    @accept_profiler('callgraph')
    def profile_bananas(self):
        # Eat 10 bananas
        person = Person()
        for a in range(10):
            person.add_banana(Banana())
        person.eat_bananas()
