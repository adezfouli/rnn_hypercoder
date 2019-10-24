import shutil

from util import DLogger


class TestLog:
    def __init__(self, test_class, expected_input):
        DLogger.remove_handlers()
        with open(expected_input, 'r') as myfile:
            self.true_output = myfile.read()
        self.test_output_buf = DLogger.get_string_logger()
        self.test_class = test_class

    def __enter__(self):
        pass
        # DLogger.logger().debug("version control: " + str(get_git()))

    def __exit__(self, type, value, traceback):
        self.test_output_buf.seek(0)
        test_output = self.test_output_buf.read()
        self.test_class.assertEqual(self.true_output, test_output)


class SaveLog:
    def __init__(self, output_file):
        self.log_capture_string = DLogger.get_string_logger()
        self.output_file = output_file

    def __enter__(self):
        pass
        # DLogger.logger().debug("version control: " + str(get_git()))

    def __exit__(self, type, value, traceback):
        with open(self.output_file, 'w') as fd:
            self.log_capture_string.seek(0)
            shutil.copyfileobj(self.log_capture_string, fd)
