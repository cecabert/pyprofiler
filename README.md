# PyProfiler

The goal of `PyProfiler` is to unify the way python code gets profiled (i.e. exectution time, memory footprint). It is inspired by the `unittest` in order to write profiling code only once independantly of the type of profiling done.

To achieve this goal, the task is splitted into two interfaces. The `Profiler` class, responsible of handling various type of profiler, and the `ProfilerCase` class that holds all the profiling case defined by the user.

# Install

The package can be directly installed from this repository using the command `pip install --upgrade git+https://github.com/cecabert/pyprofiler.git`, make sure it is invoked within the correct python environment.

# Profiler

The `Profiler` interface is responsible for interfacing with available profiter through a simple unified interface. Not all the cases will be covered, therefore some profilers might not fit into it.

Out of the box, there are a few profilers shipped with this package splitted into two categories: `time` for measuring execution time, and `memory` for measuring memory usage. The list below gives the exhaustive list:

- Time
  - `StopWatchProfiler`
  - `cProfileProfiler` interface for [cProfile](https://docs.python.org/3/library/profile.html) supporting single thread only
  - `VizTracerProfiler` interface for [viztracer](https://github.com/gaogaotiantian/viztracer) supporting single/mutli thread
- Memory
  - `TraceMallocProfiler` interface for [tracemalloc](https://docs.python.org/3/library/tracemalloc.html) supporting single thread
  - `MemoryProfilerLine` and `MemoryProfilerTime` interface for [memory-profiler](https://github.com/pythonprofilers/memory_profiler) supporting single/mutli thread
- Relationship
  - `CallGraphProfiler` generate function call graphs, adapted from [pycallgraph](https://github.com/gak/pycallgraph) and [pycallgraph2](https://github.com/daneads/pycallgraph2/tree/master)

# ProfilerCase

The `ProfilerCase` interface is where the user define what code has to be profiled in the similar fashion as `unittest.TestCase` works. To register a new profiler case, implementating a method named following the `profile_<method name>` convention is enough (i.e. samilar to `test_<method name>` with `unittest`). Later on, the profiler cases are discovered automatically at runtime.

The example below shows this concept:

```python
from pyprofiler.profiler import ProfilerCase

class ProflingExample(ProfilerCase):

    def setUp(self):
        # Do some setup before running profile case
        pass

    def tearDown(self):
        # Do cleanup here after profile case has been run
        pass

    def profile_my_func(self):
        # Add any code that need to be profiled

        # <USER CODE>
        pass
```

At runtime, the profiler is runed against all the registered cases.

## Decorators

However, some cases might not be meaningfull for a specific type of profiler. Therefore, the profiling granularity can be tweaked using various derorators to allow / disallow specific profilers

### Disable

The `@skip_case` decorator allow the user to disable some specific cases for whatever reasons.

### Disable profiler

Some profiler cases might not be relevant for a specific type of profiler. The `@skip_profiler(<profiler names>)` decorator allow to disable the underlying case for profilers matching `<profiler names>`.

### Specific profiler

On the other hand, some profile cases might be relevant for only a single type of profiler. The `@accept_profiler(<profiler names>`) enable the underlying case for only a subset of specific profilers matching the `<profiler names>`.

### Usage

The example below shows how decorators can be used to refine the profiling granularity.

```python
from pyprofiler.profiler import ProfilerCase

class ProflingExample(ProfilerCase):

    @skip_case
    def profile_disabled_case(self):
        # User disabled case
        pass

    @skip_profiler('tracemalloc')
    def profile_case_a(self):
        # Accept every profiler except `tracemalloc`
        pass

    @accept_profiler('cprofile')
    def profile_cprofile(self):
        # Case run only against `cProfile` profiler
        pass
```

# CLI

The simplest way to start a profiling session is to execute the profile cases through the `pyprofiler.cli` module, running the following command:

```bash
python -m pyprofiler.cli <path> <profiler> <arguments>
```

where `<path>` is a path to user defined file storing the profile cases, `<profiler>` is the name of the profiler to use, and `<arguments>` are the arguments passed to the profiler.

By default all the cases stored in the given file will be executed. However, it is possible to select a subset a cases to run using the following synthax:

```bash
python -m pyprofiler.cli <path>::<class> <profiler> <arguments>
```

where `<class>` is the name of the `ProfilerCase` class storing the cases to be runed. It is also possible to select a single case as follow:

```bash
python -m pyprofiler.cli <path>::<class>::<case> <profiler> <arguments>
```

where `<case>` is the name of the profile case that need to be run.

# Examples

The examples are located in this [folder](./examples/). Different usecases are highlighted in the following files:

- [timing.py](./examples/timing.py)
- [memory.py](./examples/memory.py)
- [graph.py](./examples/graph.py)
