from unittest import TestCase

import numpy as np

from do.shakti import compute_laplacian


class TestImageProcessing(TestCase):

    def test_compute_laplacian(self):
        f = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]],
                     dtype=np.float32)
        Lf = np.empty(f.shape, dtype=np.float32)
        compute_laplacian(f, Lf)
        self.assertAlmostEqual(np.linalg.norm(Lf), 0., delta=1e-8)

    def test_compute_laplacian_2(self):
        f = np.array([[0,  1,  4,  9],
                      [1,  2,  5, 10],
                      [4,  5,  8, 13],
                      [9, 10, 13, 18]],
                     dtype=np.float32)
        Lf = np.empty(f.shape, dtype=np.float32)
        compute_laplacian(f, Lf)

        expected_central_Lf = 4 * np.ones((2, 2), dtype=np.float32)

        self.assertAlmostEqual(
            np.linalg.norm(Lf[1:-1, 1:-1] - expected_central_Lf),
            0., delta=1e-8)
