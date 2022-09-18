"""
Some additional helpers for 'os' package.
"""

import os
from os import listdir
from pathlib import Path
from typing import Set, Dict, List, Iterable, Optional, Union


def _get_locs_recursive(
    parent: Path,
    image_extensions: Set[str],
    recursive: bool,
):
    dir_items: List[str] = listdir(parent)
    for di in map(Path, dir_items):
        di_full = parent / di
        if recursive and os.path.isdir(di_full):
            subgen = _get_locs_recursive(di_full, image_extensions, recursive=True)
            for didi_full in subgen:
                yield didi_full
        else:
            if image_extensions and di.suffix not in image_extensions:
                continue
            yield di_full


def get_files(
    root_dir: Path,
    extensions: Union[Set[str], str] = None,
    recursive: bool = True,
) -> Iterable[Path]:
    """
    Resolves all file locations within given root directory.
    If recursive=True, then it is very similar
    to `find {root_dir} -name <list of extensions>`
    :param root_dir: root directory
    :param extensions: Optional extensions filter.
        Files with all other extensions will be skipped.
        It is possible to pass it as a set, or as s single extension (str)
    :param recursive: if set then also scans recursively all its subdirectories.
    :return: collection of discovered locations
    """
    if isinstance(extensions, str):
        extensions = {extensions}

    gen = _get_locs_recursive(root_dir, extensions, recursive)
    for loc in gen:
        yield loc


def resolve_path(
    p: Path,
    *parent_candidates,
) -> Optional[Path]:
    """
    Resolves path.
    If it is absolute - will keep it as is.
    Then it checks whether it is given relative to current working directory.
    Otherwise, will look for first existing "rel_candidate[i] / p"
    and returns it
    :param p: path to be checked
    :param parent_candidates: possible parent candidates
    :return: resolved path in absolute form.
    """

    if isinstance(p, str):
        p = Path(p)

    if p.is_absolute():
        return p

    parents = [Path.cwd()] + list(parent_candidates)

    for parent in parents:
        candidate = parent / p
        if candidate.exists():
            return candidate.absolute()

    return None
