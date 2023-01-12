# coding=utf-8
"""
  @file: logging.py
  @data: 28 September 2022
  @author: cecabert

  Logging utility
"""
from sys import stdout, platform
import logging
from logging import Formatter, StreamHandler, FileHandler
from typing import Optional, Iterable


_default_logging_fmt = '%(name)s@%(filename)s:%(lineno)d: %(levelname)s :' \
                       ' %(message)s'
_is_win = platform == 'Win32'


class ANSIFormatter:
    GRAY = '' if _is_win else '\033[0;37m'
    GREEN = '' if _is_win else '\033[0;32m'
    YELLOW = '' if _is_win else '\033[0;33m'
    RED = '' if _is_win else '\033[38;5;196m'
    BOLD_RED = '' if _is_win else '\033[31;1m'
    BOLD_PURPLE = '' if _is_win else '\033[1;35m'
    BOLD = '' if _is_win else '\033[1m'
    UNDERLINE = '' if _is_win else '\033[4m'
    RESET = '' if _is_win else '\033[0m'


def _add_color_format(fmt, style):
    if 'levelname' in fmt:
        if style == '%':
            fmt = fmt.replace('%(levelname)',
                              '%(color)s%(levelname)s%(reset)')
        elif style == '{':
            fmt = fmt.replace('{levelname}', '{color}{levelname}{reset}')
        elif style == '$':
            fmt = fmt.replace('$levelname', '$color$levelname$reset')
        else:
            raise ValueError(f'Un-supported style type: {style}')
    return fmt


class ColorFormatter(Formatter):
    """
    Logging colored formatter
    See: https://stackoverflow.com/a/56944256/3638629
    """

    def __init__(self,
                 fmt: Optional[str] = None,
                 datefmt: Optional[str] = None,
                 style: str = '%',
                 validate: bool = True,
                 *,
                 defaults=None):
        """
        Initialize the formatter with specified format strings.

        Initialize the formatter either with the specified format string, or a
        default as described above. Allow for specialized date formatting with
        the optional datefmt argument. If datefmt is omitted, you get an
        ISO8601-like (or RFC 3339-like) format.

        Use a style parameter of '%', '{' or '$' to specify that you want to
        use one of %-formatting, :meth:`str.format` (``{}``) formatting or
        :class:`string.Template` formatting in your format string.
        """
        fmt = _add_color_format(fmt, style)
        super().__init__(fmt=fmt,
                         datefmt=datefmt,
                         style=style,
                         validate=validate,
                         defaults=defaults)
        # Add color entry + reset
        self.colors = {logging.DEBUG: ANSIFormatter.GRAY,
                       logging.INFO: ANSIFormatter.GREEN,
                       logging.WARNING: ANSIFormatter.YELLOW,
                       logging.WARN: ANSIFormatter.YELLOW,
                       logging.ERROR: ANSIFormatter.RED,
                       logging.FATAL: ANSIFormatter.BOLD_RED,
                       logging.CRITICAL: ANSIFormatter.BOLD_PURPLE}
        self.reset = ANSIFormatter.RESET

    def _add_color_format(self):
        """ Update format to add color support on `levelno` """

    def format(self, record):
        record.color = self.colors[record.levelno]
        record.reset = self.reset
        return super().format(record)


def setup(name: Optional[str] = None,
          fmt: Optional[str] = None,
          with_color: bool = True,
          level: int = logging.INFO,
          propagate: bool = True,
          handlers: Optional[Iterable[logging.Handler]] = None):
    """
    Create logger

    :param name: Name of the logger to create. If None use root logger
    :param fmt: Format of the output message, defaults to None
    :param level: Logging level
    :param propagate: If `True`, events logged to this logger will be passed to
        the handlers of higher level (ancestor) loggers, in addition to any
        handlers attached to this logger.
    :param with_color: If set to true, message levels will be colored, defaults
        to True
    :return: logger instance
    """
    if name is None:
        name = 'pyprofiler'
    logger = logging.getLogger(name=name)
    logger.propagate = propagate
    # Add handler if needed
    if len(logger.handlers) == 0:
        if handlers is None:
            # No handler given log to console
            handlers = [StreamHandler(stream=stdout)]
        # Setup handlers to logger
        if fmt is None:
            fmt = _default_logging_fmt
        for h in handlers:
            if h.formatter is None:
                if with_color and not isinstance(h, FileHandler):
                    h.setFormatter(ColorFormatter(fmt))
                else:
                    h.setFormatter(Formatter(fmt))
            logger.addHandler(h)
    # Set level
    logger.setLevel(level=level)
    return logger


# Create root logger
_root = setup(name=None)
