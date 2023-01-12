# coding=utf-8
"""
  @file: decorator.py
  @data: 28 September 2022
  @author: cecabert


  Decorator utility
"""

import logging
from typing import Iterable, Union, Dict, Callable, Type, List
from pyprofiler.utils.dispatchers import ArgsDispatcher


logger = logging.getLogger(__name__)

_SKIP_CASE_ATTR = '__should_skip_profiler_case__'
_SKIP_PROFILER_ATTR = '__skip_profiler_type__'
_ACCEPT_PROFILER_ATTR = '__accept_profiler_type__'


class ProfilerFactory:
    """
    Dynamicall register classes into a factory to later one create object
    instances out of single name
    """

    callables: Dict[str, Callable[[str], Type]] = {}

    @classmethod
    def register(cls, name, cls_to_register):
        if not hasattr(cls, 'callables') or not isinstance(cls.callables,
                                                           dict):
            msg = 'Factory need a static property named `callables` of type' \
                ' Dict[str, Callable] in order to work'
            raise AttributeError(msg)
        if name in cls.callables:
            msg = 'There is already a `class` registered with: `{}`'
            raise ValueError(msg.format(name))
        cls.callables[name] = cls_to_register

    @classmethod
    def get(cls, name):
        if name not in cls.callables:
            raise ValueError('There is no class registered with name: `{}`'
                             .format(name))
        return cls.callables[name]

    @classmethod
    def create(cls, name: str, **kwargs):
        clz = cls.get(name)
        clz = ArgsDispatcher(clz)
        return clz(**kwargs)

    @classmethod
    def known_profilers(cls) -> List[str]:
        return list(cls.callables.keys())


class register_profiler:

    def __init__(self, name):
        """Create decorator

        :param name: Name to use to register this class
        """
        self.name = name
        self.factory_cls = ProfilerFactory

    def __call__(self, cls):
        self.factory_cls.register(self.name, cls)
        return cls


def skip_case(func):
    setattr(func, _SKIP_CASE_ATTR, True)
    return func


def skip_profiler(profilers: Union[str, Iterable[str]]):
    if not isinstance(profilers, (list, tuple)):
        profilers = [profilers]
    profilers_cls = []
    # Convert str to actual profiler class
    for prof in profilers:
        profilers_cls.append(ProfilerFactory.get(prof))
    # Wrap function

    def wrapper(func):
        setattr(func, _SKIP_PROFILER_ATTR, profilers_cls)
        return func

    return wrapper


def accept_profiler(profilers: Union[str, Iterable[str]]):
    if not isinstance(profilers, (list, tuple)):
        profilers = [profilers]
    # Convert str to actual profiler class
    profilers_cls = []
    for prof in profilers:
        profilers_cls.append(ProfilerFactory.get(prof))
    # Add passthrough as a valid one
    if 'passthrough' not in profilers:
        profilers_cls.append(ProfilerFactory.get('passthrough'))

    # Wrap function
    def wrapper(func):
        setattr(func, _ACCEPT_PROFILER_ATTR, profilers_cls)
        return func

    return wrapper
