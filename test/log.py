import unittest
from rd3d.api import log
from pathlib import Path


class LogTestCase(unittest.TestCase):

    def test_add_log(self):
        runner_logger = log.create_logger(name="runner")
        bank_logger = log.create_logger(name="bank", log_file=Path(__file__).parent / 'data/bank.log')
        runner_logger.info("runner")
        bank_logger.info("bank")
        logger = log.create_logger(name="bank")
        logger.warning("logger warn")


if __name__ == '__main__':
    unittest.main()
