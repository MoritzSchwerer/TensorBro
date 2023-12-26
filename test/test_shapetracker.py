import unittest

from tensorbro.lazy import ShapeTracker

class TestShapeTracker(unittest.TestCase):
    def test_init_shape_tracker(self):
        st = ShapeTracker.from_shape((10, 10))
        self.assertEqual(len(st._views), 1)
        self.assertEqual(st.view, (10, 10))
        self.assertEqual(st.view, st._views[-1])
        self.assertEqual(st.stride, st._strides[-1])


if __name__ == "__main__":
    unittest.main()
