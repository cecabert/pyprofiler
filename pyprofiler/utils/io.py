# coding=utf-8
"""
  @file: io.py
  @data: 13 October 2022
  @author: cecabert
"""
from abc import ABC, abstractmethod
from sys import stdout
from typing import Any


class ProfilerStream(ABC):
    """ Profiler output interface """

    def __init__(self, strip_newline: bool = True):
        self._strip_newline = strip_newline

    @abstractmethod
    def write(self, x: Any) -> None:
        pass

    def sanitize(self, x: Any) -> Any:
        if isinstance(x, str) and self._strip_newline:
            x = x.strip('\n')
        return x


class Console(ProfilerStream):
    """ Standard Console output """

    def __init__(self, strip_newline: bool = True):
        super().__init__(strip_newline)
        self._stream = stdout

    def write(self, x: Any) -> None:
        x = self.sanitize(x)
        self._stream.write(f'{x}\n')


class Logger(ProfilerStream):
    """ Interface for logging package """

    def __init__(self, logger):
        super().__init__(strip_newline=True)
        self._logger = logger

    def write(self, x: Any) -> None:
        x = self.sanitize(x)
        self._logger.info(x)
