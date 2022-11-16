from typing import Iterator

class PyratOption:
    def __init__(self, *args, prefix="-S", **kwargs):
        """Create a pyrat option

        Call signature:
            * PyratOption(key=value, prefix=prefix)
            * PyratOption(argument, prefix=prefix)
            * PyratOption(prefix=prefix)

        Prefixes can take the form of any argument prefix that
        `pyrat` accepts. For example, "-s" is a valid prefix for
        arguments that will be evaluated as strings in `pyrat` and
        added to the macro's database.
        """
        self.prefix = prefix

        if len(args) == 0 and len(kwargs) == 0:
            key, value = "", ""
        elif len(args) == 1:
            key, value = args[0], ""
        elif len(kwargs) == 1:
            key, value = kwargs.popitem()
        else:
            raise ValueError("Must provide one keyword argument! (kwargs={})".format(kwargs))

        self.key = key
        self.value = value

    def cmd(self) -> str:
        return f"{self.prefix} {self.key} {self.value}"

    def __str__(self) -> str:
        return self.cmd()

    def __repr__(self) -> str:
        return f"Option({self.key}={self.value})"


class PyratOptions(dict):
    def add_option(self, *args, **kwargs):
        """Add an option to our dict of options

        Call signature:
            * add_option(option)
            * add_option(key=value, prefix=prefix)
            * add_option(argument, prefix=prefix)
            * add_option(prefix=prefix)
        """
        if len(args) == 1:
            option = args[0]
        else:
            option = PyratOption(*args, **kwargs)
        self[option.key] = option

    def __iter__(self) -> Iterator[PyratOption]:
        return iter(self.values())

    def __repr__(self) -> str:
        out = "Options("
        opt_kv = [f"{o.key}={o.value}" for o in self.values()]
        out += ", ".join(opt_kv)
        out += ")"
        return out

    def __str__(self) -> str:
        return " ".join([o.cmd() for o in self.values()])
