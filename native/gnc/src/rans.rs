//! rANS decoder with the fixed tables of `geoneural.codecs.rans`.
//!
//! Each symbol is a token decoded under one of `BINS` tables plus raw offset bits from a separate stream. Token 0
//! is zero; tokens 2e+1 and 2e+2 are positive and negative magnitudes in [2^e, 2^(e+1)) with e raw bits (most
//! significant first) below the leading one. State: 32 bits with byte-wise renormalisation, read forwards.

use crate::tables::{BINS, CDF, FREQ, PROB_BITS, RANS_L, TOKENS};
use crate::{fail, Result};

pub struct Decoder<'a> {
    stream: &'a [u8],
    raw: &'a [u8],
    x: u64,
    pos: usize,
    bit: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(stream: &'a [u8], raw: &'a [u8]) -> Result<Decoder<'a>> {
        if stream.len() < 4 {
            return fail("rANS stream shorter than its state");
        }
        let x = stream[..4].iter().fold(0u64, |x, &b| (x << 8) | b as u64);
        Ok(Decoder {
            stream,
            raw,
            x,
            pos: 4,
            bit: 0,
        })
    }

    /// Next symbol under table `bin` (< BINS).
    #[inline]
    pub fn take(&mut self, bin: usize) -> Result<i64> {
        debug_assert!(bin < BINS);
        let cdf = &CDF[bin];
        let c = self.x & ((1u64 << PROB_BITS) - 1);
        // The same bisection as the Python decoder (any exact search finds the same token).
        let (mut lo, mut hi) = (0usize, TOKENS);
        while hi - lo > 1 {
            let mid = (lo + hi) >> 1;
            if cdf[mid] as u64 <= c {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let tok = lo;
        let f = FREQ[bin][tok] as u64;
        let start = cdf[tok] as u64;
        self.x = f * (self.x >> PROB_BITS) + c - start;
        while self.x < RANS_L {
            let Some(&b) = self.stream.get(self.pos) else {
                return fail("rANS stream ends before the traversal");
            };
            self.x = (self.x << 8) | b as u64;
            self.pos += 1;
        }
        if tok == 0 {
            return Ok(0);
        }
        let e = (tok - 1) >> 1;
        let mut o = 0i64;
        for _ in 0..e {
            let Some(&byte) = self.raw.get(self.bit >> 3) else {
                return fail("raw bit stream ends before the traversal");
            };
            o = (o << 1) | ((byte >> (7 - (self.bit & 7))) & 1) as i64;
            self.bit += 1;
        }
        let a = (1i64 << e) + o;
        Ok(if (tok - 1) & 1 == 1 { -a } else { a })
    }

    /// True when the stream is used up, every raw bit was read (up to the padding of the last byte) and the
    /// state is back at its initial value, as in the Python decoder.
    pub fn finished(&self) -> bool {
        self.pos == self.stream.len() && self.x == RANS_L && self.bit.div_ceil(8) == self.raw.len()
    }
}
