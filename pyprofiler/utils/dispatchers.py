# coding=utf-8
"""
  @file: dispatchers.py
  @data: 28 September 2022
  @author: cecabert


  Dispatcher utility
"""
from inspect import signature, Parameter
from typing import Any, Dict, Optional, Callable


class ArgsDispatcher:
    """
    Wrap a function and make sure it is called with correct parameters.
    Perform parameter's name mapping if provided. If there is missing
    parameters at call size, will use default one if any, otherwise raise
    exception
    """

    def __init__(self,
                 func: Callable,
                 mapping: Optional[Dict[str, str]] = None,
                 cast: bool = False):
        """
        Constructor

        :param func: Function to wrap
        :param mapping: Dictionary storing parameter's name mapping (i.e. func
            -> new func_name), defaults to None meaning no mapping is applied
        :param cast: If true will cast parameters with information provided by
            function's signature, defaults to False
        """
        self.fn = func
        self.fn_sig = signature(func)
        self.mapping = mapping
        self.cast = cast

    def _cast_if_needed(self, p, p_info):
        if self.cast and p_info.annotation != Parameter.empty:
            p = p_info.annotation(p)
        return p

    def __call__(self, **kwargs: Any) -> Any:
        fn_args = {}
        missing_args = []
        has_kwargs = False
        for p_name, p_value in self.fn_sig.parameters.items():
            kw_name = p_name
            if self.mapping is not None and p_name in self.mapping:
                kw_name = self.mapping[p_name]
            if kw_name in kwargs:
                # Parameters if provided
                fn_args[p_name] = self._cast_if_needed(kwargs[kw_name],
                                                       p_value)
                if p_name != kw_name:
                    # Element have been mapped and used, removed it from
                    # `kwargs` to not re-add it later on with `update()`
                    # function
                    del kwargs[kw_name]
            elif p_value.default != Parameter.empty:
                # Not provided but with default value
                fn_args[p_name] = p_value.default
            elif p_value.kind == Parameter.VAR_KEYWORD:
                # A dict of keyword arguments that aren’t bound to any other
                # parameter. This corresponds to a **kwargs parameter in a
                # Python function definition -> Skip
                has_kwargs = True
                continue
            else:
                # Missing arguments
                missing_args.append(p_name)
        # Might have extra arguments in `kwargs` that need to be passed to `fn`
        # Do we need to update `fn_args` dict ?
        if has_kwargs:
            fn_args.update(kwargs)
        # Missing args?
        if missing_args:
            msg = 'The following parameters are missing: `{}`. The signature '\
                  'is: `{}({})`'
            fn_p = []
            for k, v in self.fn_sig.parameters.items():
                opt = ''
                if v.default != Parameter.empty:
                    opt = ' (optional)'
                fn_p.append(k + opt)
            fn_p = ', '.join(fn_p)
            raise TypeError(msg.format('`, `'.join(missing_args),
                                       self.fn.__name__,
                                       fn_p))
        # All arguments are given
        return self.fn(**fn_args)
