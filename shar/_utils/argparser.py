from typing import Union, Any, TypeVar
from argparse import ArgumentParser, _ArgumentGroup


class WithDefaultsWrapper:
    """
    Automatically add default values to argument help text.

    Wrapper for argument parser (and ArgumentGroups) to automatically add
    default values at the end of the help text (and automatically wrap any
    subgroup with this wrapper).
    """
    def __init__(
        self, argument_parser: Union["WithDefaultsWrapper", ArgumentParser,
                                     _ArgumentGroup]
    ) -> None:
        """
        Parameters
        ----------
        argument_parser : ArgumentParser or _ArgumentGroup object
        """
        self._argument_parser = argument_parser

    def __getattr__(self, name: str) -> Any:
        """
        Pass access of everything not definedhere directly on to the parser.
        """
        return getattr(self._argument_parser, name)

    def add_argument(self, *args, **kwargs) -> None:
        """
        Add default value to end of help text.

        If help text and default value are defined, appends the default value
        to the help text and passes everything on to the argument parser object
        """
        if ("help" in kwargs and "default" in kwargs
                and kwargs["default"] != "==SUPPRESS=="
                and kwargs["default"] is not None):
            kwargs["help"] += " (default is {})".format(kwargs["default"])
        self._argument_parser.add_argument(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs) -> "WithDefaultsWrapper":
        """
        Wrap new subgroups with this wrapper.
        """
        group = self._argument_parser.add_argument_group(*args, **kwargs)
        group = WithDefaultsWrapper(group)
        return group


ParserType = TypeVar("ParserType", ArgumentParser, WithDefaultsWrapper)
