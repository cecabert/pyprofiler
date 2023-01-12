# coding=utf-8
"""
  @file: viztracer.py
  @data: 13 October 2022
  @author: cecabert


  Interface for `VizTracer`
  https://github.com/gaogaotiantian/viztracer
"""
import os
from os.path import dirname, join
from glob import glob
from argparse import ArgumentParser
from viztracer import VizTracer
from viztracer.viewer import DirectoryViewer, ServerThread
from pyprofiler.profiler import (Profiler,
                                 format_profile_filename)
from pyprofiler.utils import register_profiler


def _count_profiler_cases(output_file: str) -> int:
    folder = dirname(output_file)
    n_cases = len(glob(join(folder, '*.json')))
    return n_cases


def _get_last_profiling_file(filename: str) -> str:
    folder = dirname(filename)
    files = glob(join(folder, '*.json'))
    files.sort()
    return files[-1]


def _directory_viewer(folder: str,
                      port: int,
                      server_only: bool,
                      once: bool,
                      flamegraph: bool,
                      timeout: int):
    cwd = os.getcwd()
    try:
        dir_viewer = DirectoryViewer(path=folder,
                                     port=port,
                                     server_only=server_only,
                                     flamegraph=flamegraph,
                                     timeout=timeout,
                                     use_external_processor=False)
        dir_viewer.run()
    finally:
        os.chdir(cwd)


def _file_viewer(output_file: str,
                 port: int,
                 server_only: bool,
                 once: bool,
                 flamegraph: bool,
                 timeout: int):
    path = _get_last_profiling_file(output_file)
    path = os.path.abspath(path)
    cwd = os.getcwd()
    try:
        server = ServerThread(path,
                              port=port,
                              once=once,
                              flamegraph=flamegraph,
                              timeout=timeout,
                              use_external_processor=False)
        server.start()
        server.ready.wait()
        if server.retcode is not None:
            return server.retcode
        if not server_only:
            from webbrowser import open_new_tab
            open_new_tab(f'http://127.0.0.1:{port}')
        while server.is_alive():
            server.join(timeout=1)
    except KeyboardInterrupt:
        if server.httpd is not None:
            server.httpd.shutdown()
        server.join(timeout=2)
    finally:
        os.chdir(cwd)


@register_profiler(name='viztracer')
class VizTracerProfiler(Profiler):
    """ VizTracer interface """

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        """ Add parser arguments """
        parser.add_argument('--tracer-entries',
                            type=int,
                            default=1000000)
        parser.add_argument('--verbose',
                            type=int,
                            default=1,
                            help='Verbose level of VizTracer. Can be set to '
                                 '`{0, 1}`')
        parser.add_argument('--max-stack-depth',
                            type=int,
                            default=-1,
                            help='Specify the maximum stack depth VizTracer '
                                 'will trace. -1 means infinite')
        parser.add_argument('--output-file',
                            type=str,
                            default='result.json',
                            help='Default file path to write report')
        # Visualization related
        parser.add_argument("--server-only",
                            default=False,
                            action="store_true",
                            help="Only start the server, do not open webpage")
        parser.add_argument("--port", "-p", nargs="?", type=int, default=9001,
                            help="Specify the port vizviewer will use")
        parser.add_argument("--flamegraph",
                            default=False,
                            action="store_true",
                            help="Show flamegraph of data")

    @classmethod
    def description(cls) -> str:
        return 'Low-overhead profiling tool that can trace and visualize your'\
               ' python code execution'

    def __init__(self,
                 tracer_entries: int = 1000000,
                 verbose: int = 1,
                 max_stack_depth: int = -1,
                 output_file: str = "result.json",
                 server_only: bool = False,
                 port: int = 9001,
                 flamegraph: bool = False):
        # Tracer options
        self._output_file = output_file
        self._tracer_opt = {'tracer_entries': tracer_entries,
                            'verbose': verbose,
                            'max_stack_depth': max_stack_depth}
        output_file = format_profile_filename(output_file)
        self._tracer = VizTracer(tracer_entries=tracer_entries,
                                 verbose=verbose,
                                 max_stack_depth=max_stack_depth,
                                 output_file=output_file)
        # Visualizer
        self._viz_opt = {'server_only': server_only,
                         'port': port,
                         'once': False,
                         'flamegraph': flamegraph,
                         'timeout': 10}

    def _initialize(self):
        output_file = format_profile_filename(self._output_file)
        return VizTracer(output_file=output_file, **self._tracer_opt)

    def start(self):
        self._tracer = self._initialize()
        self._tracer.start()

    def reset(self):
        self._tracer.stop()
        self._tracer.save()
        self._tracer.terminate()

    def trace(self, name: str):
        """ Add trace to profiling session """
        self._tracer.log_instant(name=name)

    def report(self, stream):
        stream.write('Case profiling is done!')

    def summarize_session(self):
        """ Summarize profiling session by starting webpage """
        # How many profiling cases is there ?
        n_cases = _count_profiler_cases(self._output_file)
        if n_cases > 1:
            folder = dirname(self._output_file)
            _directory_viewer(folder, **self._viz_opt)
        else:
            _file_viewer(self._output_file, **self._viz_opt)
