# coding=utf-8
"""
  @file: callgraph.py
  @data: 06 February 2023
  @author: cecabert

  Interface for `CallgraphProfiler`
"""
from typing import Optional, Callable, List, Dict, IO, Union, Any
import sys
from distutils.sysconfig import get_python_lib
from os import getcwd
import os.path as osp
import time
import multiprocessing as mp
from queue import Empty
from inspect import getmodule
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass
from argparse import ArgumentParser
from fnmatch import fnmatch
import colorsys
import operator as op
import graphviz as gv
from matplotlib.cm import get_cmap
from pyprofiler.profiler import Profiler
from pyprofiler.utils import register_profiler


FUNC_CALLMAP = Dict[str, Dict[str, int]]


def memoize(func):
    # Simple cache mecanism
    cache = {}

    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]

    return wrapper


def _init_stdlibpath():
    lib_path = get_python_lib()
    path = osp.split(lib_path)
    if path[1] == 'site-packages':
        lib_path = path[0]
    return lib_path.lower()


_getmodule = memoize(getmodule)
_stdlib_path = _init_stdlibpath()


class TraceEntry:

    def __init__(self, frame, event):
        co = frame.f_code                   # Code object
        # Module name
        #   Do work here as code object is not serializable
        module = _getmodule(co)
        self.module_name: Optional[str] = None
        self.module_path: Optional[str] = None
        if module:
            self.module_name = module.__name__
            self.module_path = module.__file__
        # Class name if any
        self.class_name = None
        if 'self' in frame.f_locals:
            self.class_name = frame.f_locals['self'].__class__.__name__
        # Function name
        self.func_name = co.co_name
        # Others
        self.filename = co.co_filename      # filename where `co` was created
        self.lineno = frame.f_lineno
        self.event = event

    @property
    def fullname(self):
        return '.'.join([a for a in (self.module_name,
                                     self.class_name,
                                     self.func_name) if a is not None])

    @property
    def from_stdlib(self) -> bool:
        if self.module_path:
            return self.module_path.lower().startswith(_stdlib_path)
        return False


class GlobFilter:

    def __init__(self, include=None, exclude=None):
        if include is None and exclude is None:
            include, exclude = ['*'], []
        elif include is None:
            include = ['*']
        elif exclude is None:
            exclude = []

        self.include = include
        self.exclude = exclude

    def __call__(self, fullname: str) -> bool:
        # Exclude pattern
        for pattern in self.exclude:
            if fnmatch(fullname, pattern):
                return False
        # Include pattern
        for pattern in self.include:
            if fnmatch(fullname, pattern):
                return True
        # If reach here, there is no match
        return False


class ModuleGrouper:

    def __call__(self, fullpath: str) -> str:
        return fullpath.split('.', 1)[0]


class Stats:
    """ Node's Statistics """

    def __init__(self,
                 value: Union[int, float],
                 total: Union[int, float]):
        self.v = value
        self.total = total
        self.frac: float = value / total if total != 0 else 0.0
    @property
    def value(self):
        return self.v

    @property
    def fraction(self):
        return self.frac


@dataclass(kw_only=True)
class Node:
    name: str
    group: str
    calls: Stats
    time: Stats
    units: str

    def value(self) -> float:
        return float(self.time.fraction * 2.0 + self.calls.fraction) / 3.0

    def label(self) -> Dict[str, str]:
        parts = [f'{self.name}',
                 f'calls: {self.calls.value:n}',
                 f'time: {self.time.value:.2f}{self.units}']
        return '\n'.join(parts)


@dataclass(kw_only=True)
class Edge:
    name: str
    group: str
    calls: Stats
    time: Stats
    from_node: str
    to_node: str

    def value(self) -> float:
        return float(self.time.fraction * 2.0 + self.calls.fraction) / 3.0

    def label(self) -> str:
        return f'{self.calls.value}'


@dataclass
class Color:
    red: float
    green: float
    blue: float
    alpha: Optional[float] = 1.0

    @classmethod
    def hsv(cls, h, s, v, a=1):
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return cls(r, g, b, a)

    @classmethod
    def plt(cls,
            value: float,
            alpha:float = 1.0,
            cmap: Optional[str] = None):
        cm = get_cmap(cmap)
        r, g, b, a = cm(value, alpha=alpha)
        return cls(red=r, green=g, blue=b, alpha=a)

    @classmethod
    def from_node(cls, node: Node, cmap: Optional[str] = None):
        return cls.plt(node.value(), alpha=0.9, cmap=cmap)

    @classmethod
    def from_edge(cls, edge: Edge, cmap: Optional[str] = None):
        return cls.plt(edge.value(), alpha=0.7, cmap=cmap)

    @classmethod
    def black(cls):
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def white(cls):
        return cls(1.0, 1.0, 1.0, 1.0)

    @property
    def r255(self):
        return int(self.red * 255)

    @property
    def g255(self):
        return int(self.green * 255)

    @property
    def b255(self):
        return int(self.blue * 255)

    @property
    def a255(self):
        return int(self.alpha * 255)

    def rgb_web(self):
        '''Returns a string with the RGB components as a HTML hex string.'''
        return '#{0.r255:02x}{0.g255:02x}{0.b255:02x}'.format(self)

    def rgba_web(self):
        '''Returns a string with the RGBA components as a HTML hex string.'''
        return '{0}{1.a255:02x}'.format(self.rgb_web(), self)


def _dict_int_init():
    return defaultdict(int)


class Graph:

    _units = {'s': 1.0,
              'ms': 1e3,
              'us': 1e6}

    def __init__(self,
                 max_depth: Optional[int] = None,
                 units: str = 'ms',
                 include_stdlib: bool = False,
                 func_filter: Optional[Callable[[str], bool]] = None,
                 func_grouper: Optional[Callable[[str], str]] = None):
        # Callstack depth
        if max_depth is None:
            max_depth = sys.maxsize
        self.max_depth = max_depth
        # Units
        if units not in self._units:
            raise ValueError('Unsupportded unit `{}`, must be one of: `{}`'
                             .format(units,
                                     '`, `'.join(self._units.keys())))
        self.units = units
        # standard library
        self.include_stdlib = include_stdlib
        # Function's name filter
        if func_filter is None:
            func_filter = GlobFilter(include=['*'],
                                     exclude=['pyprofiler.*'])
        self.func_filter = func_filter
        # Grouper
        if func_grouper is None:
            func_grouper = ModuleGrouper()
        self.func_grouper = func_grouper
        # Function's name
        self.callstack: List[str] = []
        # Function's timint
        self.callstack_time: List[float] = []
        # A mapping of which function called which other function
        self.callmap: FUNC_CALLMAP = defaultdict(_dict_int_init)
        # Counters for each function
        self.func_counter:Dict[str, int] = defaultdict(int)
        self.func_counter_max: int = 0
        # Cumulative time per function
        self.func_time: Dict[str, float] = defaultdict(float)
        self.func_time_max: float = 0.0

    def insert(self, trace: TraceEntry):
        """ Build callgraph iterativelly """

        if trace.event == 'call':   # Enter a new function
            keep = True
            # Findout call name <module>.<class>.<function>
            fullname = trace.fullname

            # Include stdlib
            if not self.include_stdlib and trace.from_stdlib:
                keep = False

            # Reach maximum depth ?
            if len(self.callstack) > self.max_depth:
                keep = False

            # Filter by name ?
            if keep and self.func_filter:
                keep = self.func_filter(fullname)

            # Store the call information
            if keep:
                # Function calling other functions
                if self.callstack:
                    src_func = self.callstack[-1]
                else:
                    src_func = None
                self.callmap[src_func][fullname] += 1
                # Update counter
                self.func_counter[fullname] += 1
                self.func_counter_max = max(self.func_counter_max,
                                            self.func_counter[fullname])
                # Update stack + timer
                self.callstack.append(fullname)
                self.callstack_time.append(time.time())
            else:
                self.callstack.append('<skip>')
                self.callstack_time.append(None)


        elif trace.event == 'return':
            # Exit a function
            if self.callstack:
                fullname = self.callstack.pop(-1)

                if self.callstack_time:
                    start_time = self.callstack_time.pop(-1)
                else:
                    start_time = None

                if start_time:
                    call_time = time.time() - start_time
                    call_time *= self._units[self.units]

                    self.func_time[fullname] += call_time
                    self.func_time_max = max(self.func_time_max,
                                             self.func_time[fullname])

    def nodes(self):    # Generator <Node>
        for func, n_calls in self.func_counter.items():
            grp = self.func_grouper(func)
            calls = Stats(n_calls, self.func_counter_max)
            dt = Stats(self.func_time.get(func, 0), self.func_time_max)
            node = Node(name=func,
                        group=grp,
                        calls=calls,
                        time=dt,
                        units=self.units)
            yield node

    def edges(self):    # Generator <Edge>
        for src_func, dests in self.callmap.items():
            if src_func is None or src_func == '<skip>':
                continue
            # New edge
            for dst_func, n_calls in dests.items():
                grp = self.func_grouper(dst_func)
                calls = Stats(n_calls, self.func_counter_max)
                dt = Stats(self.func_time.get(dst_func, 0), self.func_time_max)
                edge = Edge(name=dst_func,
                            group=grp,
                            calls=calls,
                            time=dt,
                            from_node=src_func,
                            to_node=dst_func)
                yield edge

    def groups(self):   # Generator <group>, list<func>
        _groups = defaultdict(list)
        for node in self.nodes():
            _groups[node.group].append(node)
        for grp in _groups.items():
            yield grp

    def generate(self,
                 filename: str,
                 font: str = 'Verdana',
                 font_size: int = 7,
                 group_font_size: int = 10,
                 cmap: Optional[str] = 'jet',
                 comparator: str = 'greater'):

        black = Color.black().rgba_web()
        white = Color.white().rgba_web()
        graph_attr = {'overlap': 'scalexy',
                      'fontname': font,
                      'fontsize': str(font_size),
                      'fontcolor': Color(0.0, 0.0, 0.0, 0.5).rgba_web(),
                      'label': 'Generated by CallGraphProfiler'}
        node_attr = {'fontname': font,
                     'fontsize': str(font_size),
                     'fontcolor': Color(0.0, 0.0, 0.0, 1.0).rgba_web(),
                     'style': 'filled',
                     'shape': 'rect'}
        edge_attr = {'fontname': font,
                     'fontsize': str(font_size),
                     'fontcolor': Color(0.0, 0.0, 0.0, 1.0).rgba_web()}

        graph = gv.Digraph(filename=filename,
                           format='pdf',
                           #engine='neato',
                           graph_attr=graph_attr,
                           node_attr=node_attr,
                           edge_attr=edge_attr)
        # Add groups
        for k, (name, nodes) in enumerate(self.groups()):
            with graph.subgraph(name=f'cluster_{k}') as c:
                c.attr(label=name,
                       fontsize=str(group_font_size),
                       fontcolor='black',
                       style='bold',
                       color=Color(0.0, 0.0, 0.0, 0.8).rgba_web())
                for node in nodes:
                    c.node(node.name)

        # Add nodes
        _cmp = {'lower': op.lt,
                'greater': op.gt}
        for node in self.nodes():
            fcn = _cmp[comparator]
            fontcolor = white if fcn(node.value(), 0.5) else black
            attrs = {'color': Color.from_node(node, cmap).rgba_web(),
                     'fontcolor': fontcolor}
            graph.node(node.name,
                       label=node.label(),
                       **attrs)
        # Add edges
        for edge in self.edges():
            attrs = {'color': Color.from_edge(edge, cmap).rgba_web()}
            graph.edge(tail_name=edge.from_node,
                       head_name=edge.to_node,
                       label=edge.label(),
                       **attrs)


        fname = graph.render(cleanup=True)
        return fname


class TraceWorker(mp.Process):

    def __init__(self,
                 max_depth: Optional[int] = None,
                 units: str = 'ms',
                 include_stdlib: bool = False):
        super().__init__()
        self.queue = mp.Queue()         # Trace queue
        self.graph_queue = mp.Queue()   # Where graph will be posted
        self.running = mp.Event()
        self.running.set()
        self.graph = Graph(max_depth=max_depth,
                           units=units,
                           include_stdlib=include_stdlib)
        self.start()

    def run(self) -> None:
        while self.running.is_set():
            try:
                trace = self.queue.get(timeout=0.1)
                if trace is None:   # Stopping condition
                    self.running.clear()
                    continue
                self.graph.insert(trace)
            except Empty:
                pass
        # Done post graph
        self.graph_queue.put(self.graph)

    def add(self, trace: TraceEntry):
        self.queue.put(trace, block=False)

    def join(self, timeout=None) -> None:
        # Add dummy trace to indicate the end of the processing, wait till queue
        # has finished processing all the nodes
        self.queue.put(None)
        while self.running.is_set():
            time.sleep(0.1)
        return super().join(timeout)

    def collect_graph(self):
        return self.graph_queue.get(block=True)


@register_profiler(name='callgraph')
class CallGraphProfiler(Profiler):
    """ Track functions calls """

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        parser.add_argument('--max_depth',
                            type=int,
                            default=None,
                            help='Maximum callstack depth, default infinite')
        parser.add_argument('--units',
                            type=str,
                            default='ms',
                            choices=('s', 'ms', 'us'),
                            help='Timing units')
        parser.add_argument('--include_stdlib',
                            action='store_true',
                            default=False,
                            help='If set, include call to stdlib into graph')
        parser.add_argument('--font',
                            type=str,
                            default='Verdana',
                            help='Graphviz font')
        parser.add_argument('--font_size',
                            type=int,
                            default=7,
                            help='Graphviz font size')
        parser.add_argument('--group_font_size',
                            type=int,
                            default=10,
                            help='Graphviz group font size')
        parser.add_argument('--colormap',
                            type=str,
                            default=None,
                            help='Matplotlib color name used for color '
                                 'formatting')
        parser.add_argument('--comparator',
                            type=str,
                            default='greater',
                            choices=('greater', 'lower'),
                            help='Font color comparator function')
        parser.add_argument('--folder',
                            type=str,
                            default=None,
                            help='Location where to place the generated graph,'
                                 ' default to current working directory.')

    @classmethod
    def description(cls) -> str:
        return 'Track function calls'

    def __init__(self,
                 max_depth: Optional[int] = None,
                 units: str = 'ms',
                 include_stdlib: bool = False,
                 font: str = 'Verdana',
                 font_size: int = 7,
                 group_font_size: int = 10,
                 colormap: Optional[str] = None,
                 comparator: str = 'greater',
                 folder: Optional[str] = None):
        super().__init__()
        self.worker = None
        self._original_trace_func = None
        self._case_name = None
        # Graph options
        self.max_depth = max_depth
        self.units = units
        self.include_stdlib = include_stdlib
        self.font = font
        self.font_size = font_size
        self.group_font_size = group_font_size
        self.colormap = colormap
        self.comparator = comparator
        if folder is None:
            folder = getcwd()
        self.folder = folder

    def _trace_cb(self, frame, event, args):
        # Call graph tracer
        if self.worker and event in ('call', 'return'):
            self.worker.add(TraceEntry(frame, event))
        # Original callback if any
        if self._original_trace_func:
            self._original_trace_func(frame, event, args)
        return self._trace_cb

    def __call__(self, func: Callable):
        self._case_name = func.__name__
        return super().__call__(func)

    def start(self):
        # Spawn worker + set callback
        self.worker = TraceWorker(max_depth=self.max_depth,
                                  units=self.units,
                                  include_stdlib=self.include_stdlib)
        self._original_trace_func = sys.gettrace()
        sys.settrace(self._trace_cb)

    def reset(self):
        # Set callback to original value + stop worker
        sys.settrace(self._original_trace_func)
        self.worker.join()

    def report(self, stream: IO):
        # Generate graphviz output
        graph: Graph = self.worker.collect_graph()
        filename = osp.join(self.folder, f'{self._case_name}_callgraph')
        gfilename = graph.generate(filename,
                                   font=self.font,
                                   font_size=self.font_size,
                                   group_font_size=self.group_font_size,
                                   cmap=self.colormap,
                                   comparator=self.comparator)
        stream.write(f'Generate call graph into `{gfilename}`')

