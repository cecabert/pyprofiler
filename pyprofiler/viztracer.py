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
import sys
import signal
import shutil
import time
from glob import glob
from argparse import ArgumentParser
import multiprocessing as mp
import threading
from tempfile import mkdtemp
from viztracer import VizTracer
from viztracer.patch import install_all_hooks
from viztracer.main import ReportBuilder, same_line_print
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


def _term_handler(signalnum, frame):
    sys.exit(0)



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
        parser.add_argument("--ignore_multiprocess",
                            action="store_true",
                            default=False,
                            help='Do not log any process other than the main '
                                 'process')
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
                 ignore_multiprocess: bool = False,
                 server_only: bool = False,
                 port: int = 9001,
                 flamegraph: bool = False):
        # Tracer options
        self.ofile = None
        self.ofile_template = output_file
        self.exiting = False
        self.parent_pid = None
        self.multiprocess_output_dir: str = mkdtemp()
        self.ignore_multiprocess = ignore_multiprocess
        self.cwd: str = os.getcwd()
        self.verbose = verbose
        # Tracer kwargs
        output_file = join(self.multiprocess_output_dir, 'result.json')
        self.tracer_opt = {'tracer_entries': tracer_entries,
                           'verbose': verbose,
                           'output_file': output_file,
                           'max_stack_depth': max_stack_depth,
                           'pid_suffix': True,
                           'file_info': False,
                           'register_global': True,
                           'dump_raw': True}
        self.tracer = None
        # Visualizer
        self.viz_opt = {'server_only': server_only,
                        'port': port,
                        'once': False,
                        'flamegraph': flamegraph,
                        'timeout': 10}

    def _exit_routine(self) -> None:
        if self.tracer is not None:
            if not self.exiting:
                self.exiting = True
                if self.verbose > 0:
                    same_line_print("Saving trace data, this could take a while")
                self.tracer.exit_routine()
                self._save()

    def _save(self):
        ofile = self.ofile
        self._wait_children_finish()
        reports = [join(self.multiprocess_output_dir, f)
                   for f in os.listdir(self.multiprocess_output_dir)
                   if f.endswith(".json")]
        builder = ReportBuilder(reports,
                                verbose=self.verbose)
        builder.save(output_file=ofile)
        shutil.rmtree(self.multiprocess_output_dir)

    def _wait_children_finish(self) -> None:
        try:
            if any((f.endswith(".viztmp") for f in os.listdir(self.multiprocess_output_dir))):
                same_line_print("Wait for child processes to finish, Ctrl+C to skip")
                while any((f.endswith(".viztmp") for f in os.listdir(self.multiprocess_output_dir))):
                    time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    def start(self):
        # Initialize tracer
        self.parent_pid = os.getpid()
        self.exiting = False
        # Update
        self.ofile = format_profile_filename(self.ofile_template)
        tracer = VizTracer(**self.tracer_opt)
        self.tracer = tracer

        # Patch multiprocessing
        install_all_hooks(tracer=tracer,
                          args=[],
                          patch_multiprocess=True)
        signal.signal(signal.SIGTERM, _term_handler)
        mp.util.Finalize(self, self._exit_routine, exitpriority=-1)

        # Start tracer
        self.tracer.start()

    def reset(self):
        # Stop tracer
        self.tracer.stop()

        if os.getpid() != self.parent_pid:
            mp.util.Finalize(self.tracer,
                             self.tracer.exit_routine,
                             exitpriority=-1)

        # issue141 - concurrent.future requires a proper release by executing
        # threading._threading_atexits or it will deadlock if not explicitly
        # release the resource in the code
        # Python 3.9+ has this issue
        try:
            if threading._threading_atexits:  # type: ignore
                for atexit_call in threading._threading_atexits:  # type: ignore
                    atexit_call()
                threading._threading_atexits = []  # type: ignore
        except AttributeError:
            pass

        # Call exit functions
        self._exit_routine()

        # self.tracer.save()
        # self.tracer.terminate()

    def trace(self, name: str):
        """ Add trace to profiling session """
        self.tracer.log_instant(name=name)

    def report(self, stream):
        stream.write('Case profiling is done!')

    def summarize_session(self):
        """ Summarize profiling session by starting webpage """
        # How many profiling cases is there ?
        n_cases = _count_profiler_cases(self.ofile)
        if n_cases > 1:
            folder = dirname(self.ofile)
            _directory_viewer(folder, **self.viz_opt)
        else:
            _file_viewer(self.ofile, **self.viz_opt)
