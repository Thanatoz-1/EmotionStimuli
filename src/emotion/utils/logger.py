__author__ = "Tushar Dhyani"

import logging

FORMAT = "%(asctime)-15s %(clientip)s %(user)-8s %(message)s"
logging.basicConfig(level=logging.WARN, format=FORMAT)
