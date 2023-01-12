# coding=utf-8
"""
  @file: tracemalloc.py
  @data: 17 October 2022
  @author: cecabert

  TraceMalloc profiler interface
  https://docs.python.org/3/library/tracemalloc.html
"""
from argparse import ArgumentParser
from typing import List
import tracemalloc as tm
from linecache import getline
from pyprofiler.profiler import Profiler
from pyprofiler.utils import register_profiler
from pyprofiler.utils.logging import ANSIFormatter as ANSI


def _filter_snapshot(x: tm.Snapshot) -> tm.Snapshot:
    """
    Filter out snapshot
    See: https://docs.python.org/3/library/tracemalloc.html#pretty-top
    """
    filters = [tm.Filter(False, '<frozen importlib._bootstrap>'),
               tm.Filter(False, '<unknown>')]
    return x.filter_traces(filters)


@register_profiler(name='tracemalloc')
class TraceMallocProfiler(Profiler):

    @classmethod
    def add_parser_args(cls, parser: ArgumentParser):
        parser.add_argument('--top-k',
                            type=int,
                            default=5,
                            help='Number of entries to list for a given '
                                 'snapshot')
        parser.add_argument('--index',
                            type=int,
                            default=-1,
                            help='Index of the snapshot to report')
        parser.add_argument('--key',
                            type=str,
                            default='lineno',
                            help='Key to use to sort entries within the '
                            'selected snapshot. Options are: `filename`, '
                            '`lineno` and `traceback`.')

    @classmethod
    def description(cls) -> str:
        return 'Trace memory blocks allocated by a single process/thread ' \
               'executable'

    def __init__(self,
                 top_k: int = 5,
                 index: int = -1,
                 key: str = 'lineno'):
        """Constructor

        :param top_k: Number of entries to list for a given snapshot, defaults
            to 5
        :param index: Index of the snapshot to show, defaults to -1
        :param key: Key to use to sort entries within the selected
            snapshot. Options are: `filename`, `lineno` and `traceback`.
            Defaults to 'lineno'
        """
        self._top_k = top_k
        self._index = index
        if key not in ('lineno', 'filename', 'traceback'):
            raise ValueError('The `key` parameter must be one of `{}`'
                             .format('`, '.join(('lineno',
                                                 'filename',
                                                 'traceback'))))
        self._key = key
        self._snapshots: List[tm.Snapshot] = []

    def start(self):
        if not tm.is_tracing():
            self._snapshots = []
            tm.start()

    def reset(self):
        tm.stop()

    def trace(self, name: str):
        self._snapshots.append(tm.take_snapshot())

    def report(self, stream):
        # Get snapshot
        if len(self._snapshots):
            if self._index < 0:
                self._index += len(self._snapshots)
            snap_idx = min(self._index, len(self._snapshots))
            snap = self._snapshots[snap_idx]
            # Stats
            snap = _filter_snapshot(snap)
            stats = snap.statistics(self._key)
            top_k = min(self._top_k, len(stats))
            hdr = f'Top {top_k} stats grouped by filename and line number'
            stream.write(hdr)
            # Top-k
            for idx, s in enumerate(stats[:top_k]):
                # Stats
                frame = s.traceback[0]
                stream.write('{}#{}{}: {}:{}: {:.1f} KiB'
                             .format(ANSI.BOLD,
                                     idx,
                                     ANSI.RESET,
                                     frame.filename,
                                     frame.lineno,
                                     s.size / 1024))
                # File location
                line = getline(frame.filename, frame.lineno).strip()
                if line:
                    stream.write(f'\t{line}')
            remaining = stats[top_k:]
            if remaining:
                size = sum(stat.size for stat in remaining)
                stream.write('{}{} other{}: {:.1f} KiB'
                             .format(ANSI.BOLD,
                                     ANSI.RESET,
                                     len(remaining),
                                     size / 1024))
        else:
            stream.write('No snapshots recorded, use `self.trace()` methods '
                         'within `ProfilerCase` methods to records them')
