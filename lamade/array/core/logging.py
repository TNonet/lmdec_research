import time
from decorator import decorator
from inspect import getfullargspec
from typing import Tuple, Any, Union


# def tlog(func: Callable) -> Callable:
#     """
#     Wrapper to automatically add start and end times to funcitons
#
#     Default is always to log, unless specified otherwise
#     https://stackoverflow.com/questions/34832573/python-decorator-to-display-passed-and-default-kwargs
#     """
#
#     @wraps(func)
#     def time_logging(*args, **kwargs) -> Union[Tuple[Any, ...], Any]:
#
#         argspec = getfullargspec(func)
#         positional_count = len(argspec.args) - len(argspec.defaults)
#         defaults = dict(zip(argspec.args[positional_count:], argspec.defaults))
#
#         params = copy(defaults)
#         params.update(kwargs)
#
#         if params['log'] < 0:
#             raise Exception("Logging must be non-negative integer.")
#         elif params['log']:
#             params = {k: (v if isinstance(v, (float, str, int, bool)) else str(v)) for k, v in params.items()}
#             wrapper_log = {'start': time.time(), 'params': params}
#             return_tuple = func(*args, **kwargs)
#             wrapper_log['end'] = time.time()
#
#             if len(return_tuple) == 2:
#                 return_object, flog = return_tuple
#             else:
#                 return_object, flog = return_tuple[0:-1], return_tuple[-1]
#             flog.update(wrapper_log)
#
#             if isinstance(return_object, tuple):
#                 return (*return_object, flog)
#             else:
#                 return return_object, flog
#         else:
#             return func(*args, **kwargs)
#
#     return time_logging

@decorator
def tlog(func, *args, **kwargs) -> Union[Tuple[Any, ...], Any]:
    """
    Wrapper to automatically add start and end times to funcitons

    Default is always to log, unless specified otherwise
    https://stackoverflow.com/questions/34832573/python-decorator-to-display-passed-and-default-kwargs

    :param func:
    :param args:
    :param kwargs:
    :return:
    """

    argspec = getfullargspec(func)
    non_defaults = len(args) - len(argspec.defaults)
    defaults_arg_names = argspec.args[non_defaults:]
    params = dict(zip(defaults_arg_names, args[non_defaults:]))

    if params['log'] < 0:
        raise Exception("Logging must be non-negative integer.")
    elif params['log']:
        params = {k: (v if isinstance(v, (float, str, int, bool)) else str(v)) for k, v in params.items()}
        wrapper_log = {'start': time.time(), 'params': params}
        return_tuple = func(*args, **kwargs)
        wrapper_log['end'] = time.time()

        if len(return_tuple) == 2:
            return_object, flog = return_tuple
        else:
            return_object, flog = return_tuple[0:-1], return_tuple[-1]
        flog.update(wrapper_log)

        if isinstance(return_object, tuple):
            return (*return_object, flog)
        else:
            return return_object, flog
    else:
        return func(*args, **kwargs)
