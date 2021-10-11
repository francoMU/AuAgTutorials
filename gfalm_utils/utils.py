# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)

"""
Module containing adapters for the structure object of :py:class:`pymatgen.structure.Structure`
"""
from dataclasses import fields
from dataclasses import is_dataclass
import inspect
import typing

import numpy as np


def normalize_concentrations(
    concentration: typing.Sequence[float], *, boundary=0.0001, precision=6
):
    """
    Fixes the boundary conditions and the rounding errors

    This is a routine from Adamant package
    """
    concentration = np.array(concentration, copy=True, dtype=np.float64)
    concentration[concentration <= boundary] = 0
    concentration = concentration / np.sum(concentration)
    concentration = concentration / 2
    concentration = np.around(concentration, decimals=precision)
    concentration[-1] = 0.5 - np.sum(concentration[0:-1])
    concentration = concentration * 2
    concentration = np.around(concentration, decimals=precision)
    concentration = concentration / np.sum(concentration)
    concentration = np.around(concentration, decimals=precision)

    return concentration


def change_concentrations(
    concentration: typing.Sequence[float], diff: float, index_element: int
):
    """
    Changes the concentration for the finite difference calculations
    """

    _concentration = np.array(concentration, copy=True, dtype=np.float64)
    number_of_elements = len(_concentration)
    _concentration /= np.sum(_concentration)
    _concentration += diff / (number_of_elements - 1.0)
    _concentration[index_element] -= diff + diff / (number_of_elements - 1.0)
    _concentration /= np.sum(_concentration)
    _concentration[_concentration <= 0] = 0
    _concentration[_concentration >= 1] = 1
    return normalize_concentrations(_concentration / np.sum(_concentration))


def print_variable(variable: typing.Any):
    """
    Pretty print variable in markdown
    """
    from IPython.display import Markdown as md

    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find("(") + 1 : -1].split(",")

    names = []
    for i in args:
        if i.find("=") != -1:
            names.append(i.split("=")[1].strip())

        else:
            names.append(i)

    md(f"{names[0]}: {variable}")


def pretty_print(obj: typing.Any, indent=4, use_none=True) -> None:
    """
    Pretty prints a (possibly deeply-nested) dataclass.
    Each new block will be indented by `indent` spaces (default is 4).
    """
    print(stringify(obj, indent, use_none))


def stringify(obj: typing.Any, indent=4, use_none=True, *, _indents=0) -> str:
    """
    Stringify a dataclass
    """
    if isinstance(obj, str):
        return f"'{obj}'"

    if not is_dataclass(obj) and not isinstance(obj, (typing.Mapping, typing.Iterable)):
        return str(obj)

    this_indent = indent * _indents * " "
    next_indent = indent * (_indents + 1) * " "
    start, end = (
        f"{type(obj).__name__}(",
        ")",
    )

    if is_dataclass(obj):
        # noinspection PyDataclass
        body = "\n".join(
            f"{next_indent}{field.name}="
            f"{stringify(getattr(obj, field.name), indent, _indents=_indents + 1)},"
            for field in fields(obj)
            if getattr(obj, field.name) is not None or use_none
        )

    elif isinstance(obj, typing.Mapping):
        if isinstance(obj, dict):
            start, end = "{}"

        body = "\n".join(
            f"{next_indent}{stringify(key, indent, _indents=_indents + 1)}: "
            f"{stringify(value, indent, _indents=_indents + 1)},"
            for key, value in obj.items()
        )

    else:
        if isinstance(obj, list):
            start, end = "[]"
        elif isinstance(obj, tuple):
            start = "("

        if isinstance(obj, np.ndarray):
            body = "\n".join(
                f"{next_indent}{stringify(item, indent, _indents=_indents + 1)},"
                for item in np.atleast_1d(obj)
            )
        else:
            body = "\n".join(
                f"{next_indent}{stringify(item, indent, _indents=_indents + 1)},"
                for item in obj
            )

    return f"{start}\n{body}\n{this_indent}{end}"
