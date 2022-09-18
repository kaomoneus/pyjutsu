"""
Now is only about 'parse_and_run' function.
1. Load all commands which are defined in given package.
   Each command should be defined in separate module.
2. Parse and execute whatever is given in command line.
"""
import dataclasses
import importlib
import inspect
import logging
import pkgutil
from types import ModuleType
from typing import Callable, Dict, Iterable

import click

from pyjutsu.errors import Error

LOG = logging.getLogger(__name__)


@click.group()
def _root():
    pass


@dataclasses.dataclass
class CollectedCommand:
    name: str
    action_module: ModuleType
    process_function: Callable = None


CollectedCommandsType = Dict[str, CollectedCommand]


def _collect_commands(commands_module) -> CollectedCommandsType:
    LOG.debug("Collecting commands...")
    discovered_actions = {}

    for m in pkgutil.iter_modules(commands_module.__path__):
        action_name = m.name
        module_str = ".".join([commands_module.__name__, m.name])
        LOG.debug(f"    '{action_name}'")
        action_module = importlib.import_module(module_str)

        module_commands: Iterable[click.Command] = inspect.getmembers(
            action_module, lambda obj: isinstance(obj, click.Command)
        )

        for cmd_name, cmd in module_commands:
            cmd_descr = CollectedCommand(
                cmd_name, action_module
            )
            _root.add_command(cmd)
            discovered_actions[cmd_name] = cmd_descr

    return discovered_actions


def parse_and_run(commands_module) -> int:
    """
    1. Load all commands which are defined in given package.
       Each command should be defined in separate module.
    2. Parse and execute whatever is given in command line.
    Example:
    ```
    import commands
    from utils.click_helper import load_commands

    if __name__ == "__main__":
        parse_and_run(commands)
    ```

    :param commands_module: module instance commands are defined in
    :return:
    """
    try:
        _collect_commands(commands_module)
        _root()
        return 0
    except Error as e:
        LOG.error(f"Error: {e.message}")
        return e.exitcode
