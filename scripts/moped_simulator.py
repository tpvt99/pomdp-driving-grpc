from multiprocessing import Process
import collections
import sys
from os.path import expanduser

scripts_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.dirname(scripts_path)
moped_server_path = os.path.join(src_path, 'moped')

import moped_pyro4_server

def print_flush(msg):
    print(msg)
    sys.stdout.flush()


class MopedSimulatorAccessories(Process):
    def __init__(self, cmd_args, config):
        Process.__init__(self)
        self.verbosity = config.verbosity

        Args = collections.namedtuple('args', 'host mopedpyroport')

        # Spawn meshes.
        self.args = Args(
            host='127.0.0.1',
            mopedpyroport = config.mopedpyroport)

    def run(self):
        if self.verbosity > 0:
            print_flush("[moped_simulator.py] Running moped simulator to genrate future plans")
        moped_pyro4_server.main(self.args)


