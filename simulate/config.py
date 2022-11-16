import os
from subprocess import PIPE, Popen

from ..log import logger
from .options import PyratOptions
from .utils import _keyval_to_line, _option_to_line


class PyratConfiguration:
    def __init__(self, macro: str, pyrat_path: str = "pyrat", cuda_device_id: int = None):
        """PyratConfiguration

        Initialize a pyrat configuration.

        Parameters
        ----------
        macro : str
            The `pyrat` macro to run.
        pyrat_path : str, optional
            Path to the `pyrat` program, by default "pyrat"
        cuda_device_id : int, optional
            What CUDA device to use. By default, use any device.
        """
        self.pyrat = pyrat_path
        self.macro = macro
        self.options = PyratOptions()

        self.cuda_device_id = cuda_device_id

    def add_option(self, *args, **kwargs):
        self.options.add_option(*args, **kwargs)

    def add_options(self, *args):
        """Add options to the configuration

        Call signature:
            add_options(option1, option2, ...)
        """
        for opt in args:
            self.add_option(opt)

    def cmd(self) -> str:
        """cmd

        Return the command to run pyrat with this configuration to be 
        inputted into a terminal.
        """
        cmd = [self.pyrat, self.macro, str(self.options)]
        if self.cuda_device_id:
            cmd = [f"CUDA_VISIBLE_DEVICES={self.cuda_device_id}"] + cmd
        return " ".join(cmd)

    def run(self):
        logger.info("Running pyrat with configuration %s", self)
        cmd = self.cmd()
        if "CUDA_VISIBLE_DEVICES" in cmd:
            cmd = cmd.split()[1:]
        else:
            cmd = cmd.split()
        env_dict = os.environ.copy()
        if self.cuda_device_id:
            env_dict.update({"CUDA_VISIBLE_DEVICES": str(self.cuda_device_id)})

        with Popen(cmd, stdout=PIPE, stderr=PIPE, bufsize=1, universal_newlines=True, env=env_dict) as p:
            for line in p.stdout:
                print(line, end='')  # process line here
            for line in p.stderr:
                print(line, end='')  # process line here

    def __str__(self) -> str:
        return self.cmd()

    def __repr__(self) -> str:
        out = f"Pyrat Configuration"
        if self.cuda_device_id:
            out += f"\t(CUDA device {self.cuda_device_id})"
        out += "\n"

        for k in ['pyrat', 'macro']:
            out += _keyval_to_line(k, getattr(self, k))

        for opt in self.options:
            out += _option_to_line(opt)

        return out
