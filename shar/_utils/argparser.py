class WithDefaultsWrapper:
    def __init__(self, argument_parser):
        self._argument_parser = argument_parser

    def __getattr__(self, name):
        return getattr(self._argument_parser, name)

    def add_argument(self, *args, **kwargs):
        if ("help" in kwargs and "default" in kwargs
                and kwargs["default"] != "==SUPPRESS=="
                and kwargs["default"] is not None):
            kwargs["help"] += " (default is {})".format(kwargs["default"])
        self._argument_parser.add_argument(*args, **kwargs)

    def add_argument_group(self, *args, **kwargs):
        group = self._argument_parser.add_argument_group(*args, **kwargs)
        group = WithDefaultsWrapper(group)
        return group
