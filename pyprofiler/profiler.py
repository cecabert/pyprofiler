# coding=utf-8
"""
  @file: profiler.py
  @data: 13 October 2022
  @author: cecabert

  Base profiling tool definitions
"""
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from os.path import splitext, dirname, basename, join
from glob import glob
from datetime import datetime
from inspect import getmembers
from typing import Optional, Callable, List, IO
from contextlib import redirect_stdout
from pyprofiler.utils.decorator import (ProfilerFactory,
                                        _ACCEPT_PROFILER_ATTR,
                                        _SKIP_CASE_ATTR,
                                        _SKIP_PROFILER_ATTR)
from pyprofiler.utils.logging import ANSIFormatter as ANSI
from pyprofiler.utils.io import ProfilerStream


class Profiler(ABC):
    """ Base interface for profiling tools """

    @classmethod
    @abstractmethod
    def add_parser_args(cls, parser: ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def description(cls) -> str:
        pass

    def __enter__(self):
        """ Context manager: Start """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Context manager: Closure """
        self.reset()

    def __call__(self, func: Callable):
        func()

    @abstractmethod
    def start(self):
        """ Start profile a new session """

    @abstractmethod
    def reset(self):
        """ Reset/Reinitialize profiler for a new session """
        pass

    def trace(self, name: str):
        """
        Add a new trace in the profiling session. Not every profiler supports
        it
        """
        pass

    @abstractmethod
    def report(self, stream: IO):
        """ Present profiling outcome """
        pass

    def summarize_session(self):
        """ Summarize all the profiling session """
        pass


def format_profile_filename(filename: str) -> str:
    # Format filename
    date = datetime.now().strftime('%Y-%m-%d')
    folder = dirname(filename)
    file, ext = splitext(basename(filename))
    filename = join(folder, f'{date}-{file}-*{ext}')
    # How many files is there already?
    n_entry = len(glob(filename))
    filename = filename.replace('*', f'{n_entry:04d}')
    return filename


def _should_skip_case(func: Callable) -> bool:
    return getattr(func, _SKIP_CASE_ATTR, False)


def _should_skip_profiler(func: Callable, profiler: Profiler) -> bool:
    if hasattr(func, _SKIP_PROFILER_ATTR):
        return profiler.__class__ in getattr(func, _SKIP_PROFILER_ATTR)
    if hasattr(func, _ACCEPT_PROFILER_ATTR):
        return profiler.__class__ not in getattr(func, _ACCEPT_PROFILER_ATTR)
    return False


class ProfilerCase:
    """ Base class for profiling session """

    @classmethod
    def with_profiler_name(cls, name: str, **kwargs):
        profiler = ProfilerFactory.create(name, **kwargs)
        return cls(profiler=profiler)

    @staticmethod
    def list_functions(object: 'ProfilerCase',
                       funcname: Optional[str] = None) -> List[str]:
        cases = [m for name, m in getmembers(object)
                 if name.startswith('profile')]
        if funcname is not None:
            cases = [case for case in cases if case.__name__ == funcname]
        return cases

    def __init__(self,
                 profiler: Profiler,
                 func: Optional[str] = None):
        # Set profiler and discover profiler cases
        self._profiler = profiler
        self._stream = None
        self._profile_cases = self.list_functions(self, func)

    def setUp(self):
        "Hook method for setting up the profile fixture before exercising it."
        pass

    def tearDown(self):
        "Hook method for deconstructing the profile fixture after testing it."
        pass

    def trace(self, name: str):
        """
        Add a trace event
        :param name: Name of the trace event
        """
        if self._profiler:
            self._stream.write(f'Add trace: `{name}`')
            self._profiler.trace(name)

    def runCases(self,
                 stream: ProfilerStream):
        """ Run all profile cases """
        self._stream = stream
        with redirect_stdout(stream):
            for case_fn in self._profile_cases:
                # Skipping?
                skipped = False
                reason = ''
                if _should_skip_case(case_fn):
                    skipped = True
                    reason = '\tMarked as `skipped` by user!'
                elif _should_skip_profiler(case_fn, self._profiler):
                    skipped = True
                    reason = '\tMarked as `skipped` because it does not ' \
                        'support {}!'.format(self._profiler.__class__.__name__)
                # Dump info
                status = 'Skipping' if skipped else 'Running'
                dotted_line = ANSI.BOLD + '{}'.format('-' * 80) + ANSI.RESET
                stream.write(dotted_line)
                stream.write('{}{}{}: {}:{}'.format(ANSI.BOLD,
                                                    status,
                                                    ANSI.RESET,
                                                    self.__class__.__name__,
                                                    case_fn.__name__))
                stream.write(dotted_line)
                # Start profiler using context manager
                if not skipped:
                    # Setup
                    self.setUp()
                    # Run profile fixture
                    with self._profiler as prof:
                        prof(case_fn)
                    # Tear down
                    self.tearDown()
                    # Report
                    self._profiler.report(stream)
                else:
                    stream.write(reason)
