"""
This script will be used to run all available unit tests
and to evalulate their results and save them to a file

Version: 1.0.1
Author: Petar Nikolov
"""

import os
import sys
import subprocess

from logger import Logger
from logger import LogLevel
from datetime import datetime
from config_parser import read_configuration


class TestExecutor:
    """
    This class will scan the current directory for all *.py files
    and will try to execute them. Each test will return its result as a single boolean value

    Attributes:
        dir: [str], a path to the directory where unit tests are located
        exclude_names: [list], a list of str for files names to be excluded
        extension: [str], an extension of files of interest
        output_filename: [str], a filename + path where the results  will be saved
    """

    def __init__(self):
        """
        Init TestExecutor object
        """
        try:
            logging_folder = read_configuration("logging")["foldername"]
        except:
            logging_folder = "Logs"
        abs_path_logging = os.path.abspath(logging_folder)
        if not os.path.exists(abs_path_logging):
            os.makedirs(abs_path_logging)

        self.dir = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "UnitTests"

        self.extension = ".py"
        self.exclude_names = []
        # exclude current filename - usually located in the same dir
        self.exclude_names.append(sys.argv[0])
        self.output_filename = logging_folder + "testExecution_" + datetime.now().strftime("%d.%b %Y.%H%M%S") + ".txt"
        
        self._test_result_lines = []
        self._statistics = []
        self._logger = Logger("TestRunner")
		
		
    def run(self):
        """
        Test execution runner.
        """
        
        number_of_tests = 0
        number_of_success = 0
        for file in os.listdir(self.dir):
            if file.endswith(self.extension):
                if any(file in s for s in self.exclude_names):
                    continue
                
                fullpath = os.path.join(self.dir, file)
                filename_only = os.path.basename(file)
                
                
                try:
                    # Run a process for each test file
                    proc = subprocess.Popen(
                        ['python', fullpath],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
                    
                    test_result = str(proc.communicate()[0])
                    resultLine = filename_only + ": "

                    if "Test Result: True" in test_result:
                        resultLine += "Passed"
                        number_of_success += 1
                    else:
                        resultLine += "Not passed"
                    self._test_result_lines.append(resultLine)

                    # parse line and save to statistivs self._statistics
                except Exception as err:
                    self._logger.log_message(
                        ("An exception has been thrown while running test \"" +
                            filename_only +
                            "\". The exact error is: " +
                            str(err)),
                        LogLevel.ERROR)
                number_of_tests += 1

        success_rate = 0
        if number_of_tests != 0:
            success_rate = 100 * (number_of_success / float(number_of_tests))
            
        self._statistics.append("Number of tests: " + str(number_of_tests))
        self._statistics.append("Success rate: " + str(success_rate) + "%")
        self._logger.log_message(
            "Test runner has been executed. Result: " + str(success_rate) + "%",
            LogLevel.INFO)

        # write test results to a file
        with open(self.output_filename, 'w') as f:
            for item in self._test_result_lines:
                f.write("%s\n" % item)
            f.write("\n\nStatistics \n")
            for item in self._statistics:
                f.write("%s\n" % item)

                
if __name__ == '__main__':
    test_runner = TestExecutor()
    test_runner.run()