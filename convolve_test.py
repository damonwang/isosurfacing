#!/usr/bin/env python

import numpy as np
import unittest
import cnv
import Image
import tempfile
from os.path import join
from shutil import rmtree

class TestEmbed(unittest.TestCase):

    def assertArraysSimilar(self, tolerance, A, B):
        self.assertTrue(tolerance >= abs(A - B).max())

    def test_embed(self):
        '''embed'''

        A = np.arange(21).reshape((3,7))
        expected = np.array([[  0.,   0.,   1.,   2.,   3.,   4.,   5.,   6.,   6.,   6.],
            [  7.,   7.,   8.,   9.,  10.,  11.,  12.,  13.,  13.,  13.],
            [ 14.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  20.,  20.]])
        self.assertTrue((cnv.embed(A, 3) == expected).all())

    def test_convolve_f1d(self):
        '''convolve fast axis'''

        A = np.arange(21).reshape((3,7))
        expected = np.array([[  2.,   5.,   8.,  11.,  14.,  17.,  18.],
            [ 23.,  26.,  29.,  32.,  35.,  38.,  39.],
            [ 44.,  47.,  50.,  53.,  56.,  59.,  60.]])
        self.assertTrue((cnv.convolve_f1d(A, np.arange(3)) == expected).all())

    def test_convolve_png(self):
        '''convolve phoenix head, x-axis'''

        expected = cnv.png_to_ndarray('data/head-dx.png')
        A = cnv.convolve_png('data/head-16.png', cnv.first_d)
        self.assertArraysSimilar(1, cnv.igreyscale(A, max=255), expected)

    def test_part1a(self):
        '''part1a on the head'''

        tempdir = tempfile.mkdtemp(prefix='scivis')
        saveas=join(tempdir, 'part1a')
        cnv.part1a(filename='data/head-16.png', saveas=saveas, aspect=(1,1))
        for op in ['dx', 'dy', 'gm']:
            self.assertArraysSimilar(1, 
                    cnv.png_to_ndarray('data/head-%s.png' % op), 
                    cnv.igreyscale(cnv.png_to_ndarray('%s-%s.png' % (saveas, op)), max=255))
        rmtree(tempdir)

if __name__ == '__main__': 
    tests = [ unittest.TestLoader().loadTestsFromTestCase(v)
            for k,v in globals().items() if k.startswith("Test")]
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(tests))
