# coding=utf-8
"""
  @file: memory_profiler.py
  @data: 14 October 2022
  @author: cecabert

  Interface for `memory_profiler`
  https://github.com/pythonprofilers/memory_profiler
"""
from argparse import ArgumentParser
import time
import os
from glob import glob
from typing import Callable, List, Optional, NamedTuple, Dict, Tuple
from collections import namedtuple
from signal import SIGKILL
import psutil
import ast
import textwrap
import inspect
import numpy as np
from memory_profiler import (MemTimer,
                             LineProfiler,
                             CodeMap,
                             _get_memory,
                             _get_child_memory,
                             Pipe)
from mprof import plot_file
import matplotlib.pyplot as plt
from pyprofiler.utils.loader import load_module
from pyprofiler.utils import register_profiler
from pyprofiler.profiler import Profiler


PlotOptions = namedtuple('PlotOptions', ['slope', 'xlim'])


def _default_naming():
    exists = True
    while exists:
        filename = 'mprofile_{}.dat'.format(time.strftime("%Y-%m-%d-%H-%M-%S",
                                                          time.localtime()))
        exists = os.path.exists(filename)
    return filename


def _gather_files() -> List[str]:
    profiles = glob("mprofile_????-??-??-??-??-??.dat")
    profiles.sort()
    return profiles


def _generate_plots(files: List[str],
                    options: Optional[NamedTuple] = None,
                    timestamps: bool = False,
                    traces: Dict[str, List[Tuple[str, float]]] = {},
                    output_folder: Optional[str] = None,
                    keep_original_file: bool = False):
    if options is None:
        options = PlotOptions(slope=True, xlim=None)

    for file in files:
        # Create figure
        fig = plt.figure(figsize=(14, 6), dpi=90)
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        # Plot
        mprofile = plot_file(file,
                             timestamps=timestamps,
                             options=options)
        plt.xlabel("time (in seconds)")
        plt.ylabel("memory used (in MiB)")
        plt.title(mprofile['cmd_line'])

        leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        leg.get_frame().set_alpha(0.5)
        plt.grid()

        if traces:
            file_traces = traces.get(file, None)
            if file_traces is not None:
                # Taken from mprof.py:plot_file()
                # Find starting time
                t = mprofile['timestamp']
                ts = mprofile['func_timestamp']
                mem = mprofile['mem_usage']
                if len(ts) > 0:
                    for values in ts.values():
                        for v in values:
                            t.extend(v[:2])
                t = np.asarray(t)
                mem = np.asarray(mem)
                ind = t.argsort()
                mem = mem[ind]
                t = t[ind]
                global_start = float(t[0])
                max_mem = mem.max()
                # Loop over each trace
                for name, trace in file_traces:
                    t0 = trace - global_start
                    plt.vlines(x=t0,
                               ymin=0.0,
                               ymax=1.1*max_mem)
                    plt.text(x=t0,
                             y=1.1 * max_mem,
                             s=name,
                             rotation=90,
                             verticalalignment='top')

        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            filename = os.path.join(output_folder,
                                    file.replace('.dat', '.png'))
            plt.savefig(filename)
        else:
            plt.show(block=True)
        plt.close(fig=fig)
        if not keep_original_file:
            os.remove(file)


class MemTimerMultiprocess(MemTimer):
    """ Update version to support mutliprocess child memory usage tracking """

    def __init__(self, monitor_pid, interval, pipe, backend, max_usage=False,
                 *args, **kw):
        self.multiprocess = kw.pop('multiprocess', False)
        super().__init__(monitor_pid,
                         interval,
                         pipe,
                         backend,
                         max_usage,
                         *args,
                         **kw)

    def run(self):
        self.pipe.send(0)  # we're ready
        stop = False
        while True:
            cur_mem = _get_memory(self.monitor_pid,
                                  self.backend,
                                  timestamps=self.timestamps,
                                  include_children=self.include_children)
            if not self.max_usage:
                if self.multiprocess:
                    cur_mem = [cur_mem]
                    for chldmem in _get_child_memory(self.monitor_pid):
                        cur_mem.append((chldmem, time.time()))
                self.mem_usage.append(cur_mem)
            else:
                self.mem_usage[0] = max(cur_mem, self.mem_usage[0])
            self.n_measurements += 1
            if stop:
                break
            stop = self.pipe.poll(self.interval)
            # do one more iteration

        self.pipe.send(self.mem_usage)
        self.pipe.send(self.n_measurements)


def _memory_usage(proc,
                  backend,
                  stream,
                  interval=.1,
                  timestamps=False,
                  include_children=False,
                  multiprocess=False,
                  max_usage=False,
                  max_iterations=None):
    # for a Python function wait until it finishes
    max_iter = float('inf')
    if max_iterations is not None:
        max_iter = max_iterations
    if stream is not None:
        timestamps = True
    # Spawn memory timer
    current_iter = 0
    while True:
        child_conn, parent_conn = Pipe()  # this will store MemTimer's results
        p = MemTimerMultiprocess(os.getpid(),
                                 interval,
                                 child_conn,
                                 backend,
                                 timestamps=timestamps,
                                 max_usage=max_usage,
                                 include_children=include_children,
                                 multiprocess=multiprocess)
        p.start()
        parent_conn.recv()  # wait until we start getting memory

        # When there is an exception in the "proc" - the (spawned) monitoring
        # processes don't get killed. Therefore, the whole process hangs
        # indefinitely. Here, we are ensuring that the process gets killed!
        try:
            proc()
            parent_conn.send(0)  # finish timing
            ret = parent_conn.recv()
            n_measurements = parent_conn.recv()
            if not max_usage and stream is not None:
                # Dump info
                for entry in ret:
                    mem_usage = entry if isinstance(entry, tuple) else entry[0]
                    stream.write('MEM {0:.6f} {1:.4f}\n'.format(*mem_usage))
                    if multiprocess and isinstance(entry, list):
                        for idx, chldmem in enumerate(entry[1:]):
                            stream.write('CHLD {0} {1:.6f} {2:.4f}\n'
                                         .format(idx, *chldmem))
            if max_usage:
                # Convert the one element list produced by MemTimer to a single
                # value
                ret = ret[0]
        except Exception:
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                os.kill(child.pid, SIGKILL)
            p.join(0)
            raise

        p.join(5 * interval)

        if ((n_measurements > 4) or (current_iter == max_iter) or
                (interval < 1e-6)):
            break
        interval /= 10.
    if stream:
        return None
    return ret


class ParseCallNode(ast.NodeVisitor):

    def __init__(self):
        self.func = []

    def visit_Attribute(self, node):
        self.generic_visit(node)
        self.func.append(node.attr)

    def visit_Name(self, node):
        self.func.append(node.id)

    @property
    def function_name(self):
        return '.'.join(self.func)


class CollectFunctionDef(ast.NodeVisitor):

    def __init__(self):
        self._funcs = []

    def __call__(self, filename: str):
        with open(filename, 'rt') as f:
            content = f.read()
        tree = ast.parse(content)
        self.visit(tree)
        return self._funcs

    def visit_ClassDef(self, node: ast.ClassDef):
        # Skip class definition, we're not interested in method defined in a
        # class
        pass

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._funcs.append(node.name)
        self.generic_visit(node)


class GatherFunctionCall(ast.NodeVisitor):
    """
    Parse code and extract function calls
    See: https://stackoverflow.com/a/8659251
    """

    def __init__(self):
        self.func = []

    def visit_Call(self, node):
        p = ParseCallNode()
        p.visit(node.func)
        self.func.append(p.function_name)
        self.generic_visit(node)

    def __call__(self, code: str):
        sanitized_code = textwrap.dedent(code)
        tree = ast.parse(sanitized_code)
        self.visit(tree)
        return self.func


def _get_source(code):
    src = inspect.getsource(code)
    src = src.strip()
    offset = 0
    if src.startswith('@'):
        offset = src.find('def')
    return src[offset:]


def _is_inner_function_def(code, toplevel_code):
    if toplevel_code is not None:
        for subcode in filter(inspect.iscode, toplevel_code.co_consts):
            if code.co_name == subcode.co_name:
                return True
    return False


class RecursiveCodeMap(CodeMap):

    def __init__(self,
                 include_children, backend,
                 max_depth: Optional[int] = None):
        super().__init__(include_children=include_children, backend=backend)
        self._funcs = []
        self.max_depth = max_depth

    def add(self, code, toplevel_code=None, depth: int = 0):
        if code in self:
            return

        if self.max_depth and depth > self.max_depth:
            return

        # if toplevel_code is None:
        filename: str = code.co_filename
        if filename.endswith((".pyc", ".pyo")):
            filename = filename[:-1]
        if not os.path.exists(filename):
            print('ERROR: Could not find file ' + filename)
            if filename.startswith(("ipython-input", "<ipython-input")):
                print(
                    "NOTE: %mprun can only be used on functions defined in"
                    " physical files, and not in the IPython environment.")
            return

        # Top level
        toplevel = toplevel_code is None
        sub_lines, start_line = inspect.getsourcelines(code)
        linenos = range(start_line,
                        start_line + len(sub_lines))
        if not _is_inner_function_def(code, toplevel_code):
            self._funcs.append((toplevel, filename, code, linenos, sub_lines))
        self[code] = {} if toplevel else self[toplevel_code]

        # Do we call other functions defined in same file as profile_cases ?
        # src = _get_source(code)
        for func in CollectFunctionDef()(filename):
            if func in code.co_names:
                m = load_module(filename)
                fn_called = getattr(m, func)
                self.add(fn_called.__code__,
                         toplevel_code=code,
                         depth=depth + 1)

        # Deal with lambda functions defined inside
        for subcode in filter(inspect.iscode, code.co_consts):
            self.add(subcode, toplevel_code=code, depth=depth)

    def items(self):
        """Iterate on the toplevel code blocks."""
        fn_iter = iter(self._funcs)
        while (fn_entry := next(fn_iter, None)) is not None:
            toplevel, filename, code, lineos, sub_lines = fn_entry
            measures = self[code]
            if not measures:
                continue
            line_it = ((line,
                        name,
                        measures.get(line)) for line, name in zip(lineos,
                                                                  sub_lines))
            yield toplevel, filename, code.co_name, line_it


def show_results(prof, stream, precision=1):
    template = '{0:<10} {1:>6} {2:>12} {3:>12}  {4:>10}   {5:<}'

    for (toplevel, filename, fcn_name, lines) in prof.code_map.items():
        if toplevel:
            header = template.format('File',
                                     'Line #',
                                     'Mem usage',
                                     'Increment',
                                     'Occurrences',
                                     'Line Contents')

            stream.write(u'Function name: ' + fcn_name + '\n\n')
            stream.write(header + u'\n')
            stream.write(u'=' * len(header) + '\n')

        float_format = u'{0}.{1}f'.format(precision + 4, precision)
        template_mem = u'{0:' + float_format + '} MiB'
        for k, (lineno, fcn_name, mem) in enumerate(lines):
            if mem:
                inc = mem[0]
                total_mem = mem[1]
                total_mem = template_mem.format(total_mem)
                occurrences = mem[2]
                inc = template_mem.format(inc)
            else:
                total_mem = u''
                inc = u''
                occurrences = u''
            fname = os.path.basename(filename)
            code = fcn_name
            tmp = template.format(fname,
                                  lineno,
                                  total_mem,
                                  inc,
                                  occurrences,
                                  code)
            stream.write(tmp)
        stream.write(u'\n\n')


@register_profiler(name='memory-profiler-time')
class MemoryProfilerTime(Profiler):
    """ memory_profiler interface: Over time analysis """

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        parser.add_argument("--interval", "-T",
                            dest="interval",
                            default="0.1",
                            type=float,
                            action="store",
                            help="Sampling period (in seconds), defaults to "
                                 "0.1")
        parser.add_argument("--include-children", "-C",
                            dest="include_children",
                            action="store_true",
                            help="Monitors forked processes as well (sum up"
                                 " all process memory)")
        parser.add_argument("--multiprocess", "-M",
                            dest="multiprocess",
                            action="store_true",
                            help="Monitors forked processes creating "
                                 "individual plots for each child")
        parser.add_argument("--backend",
                            dest="backend",
                            choices=["psutil",
                                     "psutil_pss",
                                     "psutil_uss",
                                     "posix",
                                     "tracemalloc"],
                            default="psutil",
                            help="Current supported backends: 'psutil', "
                            "'psutil_pss', 'psutil_uss', 'posix', "
                            "'tracemalloc'. Defaults to 'psutil'.")
        # Summary
        parser.add_argument('--plots-directory',
                            type=str,
                            default=None,
                            help='Location where to save generated plots')
        parser.add_argument('--keep_file',
                            action='store_true',
                            default=False,
                            help='If set, will keep the original trace file, '
                                 'otherwise discard it.')

    @classmethod
    def description(cls) -> str:
        return 'Generate a full over time memory usage report of an executa'\
               'ble and to plot/export it'

    def __init__(self,
                 interval: float = 0.1,
                 include_children: bool = False,
                 multiprocess: bool = False,
                 backend: str = 'psutils',
                 plots_directory: Optional[str] = None,
                 keep_original_file: bool = False):
        self._mprofile_output: str = None
        self._interval = interval
        self._include_children = include_children
        self._multiprocess = multiprocess
        self._backend = backend
        self._plot_directory = plots_directory
        self._keep_original_file = keep_original_file
        self._traces: Dict[str, List[Tuple[str, float]]] = {}

    def start(self):
        self._mprofile_output = _default_naming()
        self._traces[self._mprofile_output] = []

    def reset(self):
        return super().reset()

    def trace(self, name: str):
        self._traces[self._mprofile_output].append((name, time.time()))

    def __call__(self, func: Callable):
        with open(self._mprofile_output, 'a') as f:
            f.write("CMDLINE {0}\n".format(func.__name__))
            _memory_usage(proc=func,
                          interval=self._interval,
                          timestamps=True,
                          include_children=self._include_children,
                          multiprocess=self._multiprocess,
                          stream=f,
                          backend=self._backend)

    def report(self, stream):
        stream.write(f'Profile saved in: {self._mprofile_output}')

    def summarize_session(self):
        """ Show all plots """
        # Get profiling files
        mprof_files = _gather_files()
        # Generates figures
        _generate_plots(files=mprof_files,
                        traces=self._traces,
                        output_folder=self._plot_directory,
                        keep_original_file=self._keep_original_file)


@register_profiler(name='memory-profiler-line')
class MemoryProfilerLineByLine(Profiler):
    """ memory_profiler interface: Line by line analysis """

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        parser.add_argument("--include-children", "-C",
                            dest="include_children",
                            action="store_true",
                            help="Monitors forked processes as well (sum up"
                                 " all process memory)")
        parser.add_argument("--backend",
                            dest="backend",
                            choices=["psutil",
                                     "psutil_pss",
                                     "psutil_uss",
                                     "posix",
                                     "tracemalloc"],
                            default="psutil",
                            help="Current supported backends: 'psutil', "
                            "'psutil_pss', 'psutil_uss', 'posix', "
                            "'tracemalloc'. Defaults to 'psutil'.")
        parser.add_argument('--max-depth',
                            type=int,
                            default=1,
                            help='Maximum level of tracing to allow. Negative'
                                 ' value means no limit.')

    @classmethod
    def description(cls) -> str:
        return 'Generate a line-by-line memory usage report of an executable'

    def __init__(self, backend: str = 'psutil',
                 include_children: bool = False,
                 max_depth: int = 1):
        self._backend = backend
        self._include_children = include_children
        if max_depth < 0:
            max_depth = None
        self._max_depth = max_depth
        self._profiler = None

    def start(self):
        self._profiler = LineProfiler(backend=self._backend,
                                      include_children=self._include_children)
        self._profiler.code_map = RecursiveCodeMap(self._include_children,
                                                   self._backend,
                                                   self._max_depth)

    def reset(self):
        return super().reset()

    def __call__(self, func: Callable):
        if self._profiler:
            self._profiler(func)()

    def report(self, stream):
        if self._profiler:
            show_results(self._profiler, stream=stream)
