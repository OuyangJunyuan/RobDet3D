import unittest


class TestMisc(unittest.TestCase):
    def test_replace_attr(self):
        from rd3d.utils.base import replace_attr

        class Model:
            def __init__(self):
                self.name = 'cnn'

        m = Model()
        with replace_attr(m, name='spc'):
            self.assertEqual(m.name, 'spc')
        self.assertEqual(m.name, 'cnn')


if __name__ == '__main__':
    unittest.main()
