import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
import os

if __name__ == '__main__':
    try:
        os.remove("output.txt")
    except OSError:
        pass

    suite = unittest.defaultTestLoader.discover('tests')
    with open('results.json', 'w') as f:
        JSONTestRunner(visibility='visible', stream=f).run(suite)