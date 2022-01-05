from abc import ABC, abstractmethod


class UnitTest(ABC):
    """
    Base Test method.
    Implement all 3 abstract methods
    """

    @abstractmethod
    def execute():
        """
        main method which is executing the test
        """
        pass

    @abstractmethod
    def get_time():
        """
        return time needed for test execution in string format
        """
        pass

    @abstractmethod
    def get_result():
        """
        return boolean result (True/False)
        """
        pass

    @abstractmethod
    def get_message():
        """
        return a message to be logged. The message could be either 'None' or a string. example:
            "Accuracy result: 70%"
        """
        pass

    @abstractmethod
    def get_test_description():
        """
        return a short test description - to be visualizied in logs
        """
        pass

