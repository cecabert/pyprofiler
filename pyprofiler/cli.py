#!/bin/env/python
# coding=utf-8
"""
  @file: cli.py
  @data: 13 October 2022
  @author: cecabert
"""
from argparse import ArgumentParser, Namespace
from inspect import getmembers
from typing import Optional, List
from pyprofiler.utils.logging import setup
from pyprofiler.utils.dispatchers import ArgsDispatcher
from pyprofiler.profiler import (Profiler,
                                 ProfilerFactory,
                                 ProfilerCase)
from pyprofiler.utils.io import Logger
from pyprofiler.utils.loader import load_module


def _load_cases(module,
                specific_case: Optional[str] = None) -> List[ProfilerCase]:

    def _is_profiler_case(x):
        # issubclass(TypeA, TypeA) will return True, and we don't want to pick
        # base class, only derived class. Therefore add `x is not ProfilerCase`
        # to filter out such case.
        return (isinstance(x, type) and issubclass(x, ProfilerCase) and
                x is not ProfilerCase)

    cases = []
    for name, class_name in getmembers(module, predicate=_is_profiler_case):
        if specific_case is None or (specific_case and specific_case == name):
            cases.append(class_name)
    return cases


def _parse_command_line() -> Namespace:

    # Main parser
    main_parser = ArgumentParser()
    main_parser.add_argument('input',
                             type=str,
                             help='File storing the profiling case(s)')
    main_parser.add_argument('--lists',
                             action='store_true',
                             help='List available profiling cases')
    subparser = main_parser.add_subparsers(title='profilers',
                                           description='Which profiler to use',
                                           dest='type')
    # Add each profiler
    for name, cls in ProfilerFactory.callables.items():
        cls: Profiler
        parser = subparser.add_parser(name=name.lower(),
                                      help=cls.description())
        cls.add_parser_args(parser=parser)
    # Parse
    args = main_parser.parse_args()
    return args


def _discover_profiler_cases(inputs: str) -> List[ProfilerCase]:
    """
    Discover ProfilerCase

    <filename>
    <filename>::<class_name>
    <filename>::<class_name>::<func_name>
    <module>
    <module>::<class_name>
    <module>::<class_name>::<func_name>

    :param inputs: Input locations
    """
    # Deal with specific case input
    filename_or_module = inputs
    specific_case = None
    specific_func = None
    if '::' in inputs:
        filename_or_module, specific_case = inputs.split('::', maxsplit=1)
        if '::' in specific_case:
            specific_case, specific_func = specific_case.split('::')
    # Load module
    module = load_module(filename_or_module)
    # Pick profiler cases
    cases = _load_cases(module=module, specific_case=specific_case)
    return cases, specific_func


def main(args: Namespace):
    # Stream
    stream = Logger(setup(name='PyProfiler', propagate=False))
    # Discover ProfileCase subclass from user `--input`
    if args.lists:
        inputs = args.input.split('::')[0]
        profiler_cases, _ = _discover_profiler_cases(inputs)
        stream.write('Available `ProfilerCase`')
        for case in profiler_cases:
            stream.write(' * {}'.format(case.__name__))
            for func in case.list_functions(case):
                stream.write('    └─ {}'.format(func.__name__))
        return
    else:
        profiler_cases, specific_func = _discover_profiler_cases(args.input)
    # Create profiler
    profiler: Profiler = ProfilerFactory.create(name=args.type, **args)
    # Run them
    for prof_case in profiler_cases:
        ctor = ArgsDispatcher(prof_case)
        case: ProfilerCase = ctor(profiler=profiler, func=specific_func)
        case.runCases(stream=stream)
    # Summarize
    profiler.summarize_session()


if __name__ == '__main__':
    args = _parse_command_line()
    main(args)
