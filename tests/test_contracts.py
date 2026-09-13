"""Small contract tests for the codec, filter and accounting paths."""
from __future__ import annotations
import gzip
import struct
import unittest
import numpy as np
from geoneural.codecs.eat1 import decode, encode
from geoneural.data.build import expand_bilinear, smooth_decimate
from geoneural.codecs.contracts import DeploymentBytes


class CodecContracts(unittest.TestCase):
    def test_quantized_signed_roundtrip(self):
        grid=np.linspace(-251.371,813.529,65*65).reshape(65,65)
        packed=encode(grid,2,0.01)
        restored,header=decode(packed,32+65*65*4)
        self.assertEqual(header['level'],2)
        self.assertLessEqual(float(np.max(np.abs(restored-grid))),0.005000001)
        self.assertEqual(encode(grid,2,0.01),packed)

    def test_corrupt_truncated_trailing_and_nan_refuse(self):
        packed=encode(np.zeros((5,5)),0)
        for invalid in (packed[:-1],packed+b'extra',b'not gzip'):
            with self.assertRaises(Exception): decode(invalid)
        raw=bytearray(gzip.decompress(packed)); struct.pack_into('<I',raw,12,1)
        with self.assertRaises(ValueError): decode(gzip.compress(raw))
        with self.assertRaises(ValueError): encode(np.full((5,5),np.nan),0)

    def test_flat_filter_and_linear_interpolator(self):
        self.assertTrue(np.array_equal(smooth_decimate(np.full((65,65),15,dtype=np.float32)),np.full((33,33),15)))
        y,x=np.mgrid[0:5,0:5]
        expanded=expand_bilinear(2*x+3*y,4)
        yy,xx=np.mgrid[0:17,0:17]
        self.assertTrue(np.allclose(expanded,.5*xx+.75*yy))

    def test_complete_deployment_accounting(self):
        self.assertEqual(DeploymentBytes(1,2,3,4,5,6).total(),21)
        with self.assertRaises(ValueError): DeploymentBytes(-1,0,0,0,0,0).total()


if __name__=='__main__': unittest.main()
