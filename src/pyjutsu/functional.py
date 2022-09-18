"""
Some helpers for functional programming
"""
from typing import Tuple, Any, Callable, Iterable


def apply_pipeline(
    subject,
    functions: Iterable[Callable],
    # In future, we will perhaps add these
    # key_argv: Callable[[Any, Any], Any] = None,
    # key_kwargs: Callable[[Any, Any], Any] = None,
):
    """
    Applies pipeline of functions to some subject
    :param subject: contents we about to pass through the pipeline
    :param functions: functions pipeline
    :return:
    """

    res = subject
    for f in functions:
        res = f(res)
    return res
