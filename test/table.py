import unittest
from prettytable import PrettyTable


class TestInstanceBank(unittest.TestCase):

    def test3_coverage(self):
        tb = PrettyTable(title="title", field_names=['class', 'gt', 'anno'])
        tb.add_row(['car',1,2])
        print(tb.get_string())


if __name__ == '__main__':
    unittest.main()
