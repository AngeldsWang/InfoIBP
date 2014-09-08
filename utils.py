import logging, sys
from optparse import OptionParser


class LogFile(object):
    def __int__(self, name=None):
        self.logger = logging.getLogger(name)

    def write(self, msg, level=logging.INFO):
        self.logger.log(level, msg)

    def flush(self):
        for handler in self.logger.handlers:
            handler.flush()


def parse_args():
    parser = OptionParser()
    parser.set_defaults(alpha=0.3, logfile=None)

    parser.add_option("--alpha", type="float", dest="alpha",
                      help="the mass parameter for new influential")
    parser.add_option("--logfile", type="string", dest="logfile",
                      help="the filename to save logs")
    parser.add_option("--cores_num", type="int", dest="cores_num",
                      help="the cpu cores will be used")

    (options, args) = parser.parse_args()
    return options