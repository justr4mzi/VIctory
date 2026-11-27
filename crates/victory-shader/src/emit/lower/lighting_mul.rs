use super::{Bin, Fields, Inst, Lower, Operand, V, Why, pred};
use crate::ir::Reg;

const SIGN: u32 = 0x8000_0000;
const MIN_NORMAL: u32 = 0x0080_0000;
const INFINITY: u32 = 0x7f80_0000;
const NEGATIVE_MAX: u32 = 0xff7f_ffff;

impl Lower<'_> {
    /// RDNA2 lighting multiply in the emitter's default f32 mode (nearest
    /// even, IEEE off, denormals flushed, DX10 clamp on). Measurements:
    /// https://github.com/boykopovar/AnyPS5/pull/501
    /// https://github.com/boykopovar/AnyPS5/pull/2046
    /// https://github.com/boykopovar/AnyPS5/pull/25011
    pub(super) fn lighting_mul(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let c = self.cur;
        let sign_mask = self.s.m.i32(SIGN);
        let mag_mask = self.s.m.i32(!SIGN);
        let zero = self.s.m.i32(0);
        let minimum = self.s.m.i32(MIN_NORMAL);
        let infinity = self.s.m.i32(INFINITY);
        let lowest = self.s.m.i32(NEGATIVE_MAX);

        let mut inputs = [zero; 3];
        for (k, input) in inputs.iter_mut().enumerate() {
            let operand = v.src.get(k).ok_or("a missing operand")?;
            let mut bits = self.operand(inst, operand)?;
            // Use the bits so negating a signaling NaN doesn't quiet it.
            if v.abs >> k & 1 != 0 {
                bits = self.s.m.bin(c, Bin::And, bits, mag_mask);
            }
            if v.neg >> k & 1 != 0 {
                bits = self.s.m.bin(c, Bin::Xor, bits, sign_mask);
            }
            *input = bits;
        }
        let [lhs, rhs, limit] = inputs;
        let lhs_mag = self.s.m.bin(c, Bin::And, lhs, mag_mask);
        let rhs_mag = self.s.m.bin(c, Bin::And, rhs, mag_mask);

        // Bits in [MIN_NORMAL, INFINITY] encode precisely the positive limits
        // allowed in this mode. The right factor has its own sentinel.
        let limit_positive = self.s.m.cmp(c, pred::IUGE, limit, minimum);
        let limit_ordered = self.s.m.cmp(c, pred::IULE, limit, infinity);
        let limit_ok = self.s.m.bin(c, Bin::And, limit_positive, limit_ordered);
        let rhs_nan = self.s.m.cmp(c, pred::IUGT, rhs_mag, infinity);
        let rhs_lowest = self.s.m.cmp(c, pred::IUGE, rhs, lowest);
        let invalid_rhs = self.s.m.bin(c, Bin::Or, rhs_nan, rhs_lowest);

        let product = self.lighting_product(lhs, rhs, lhs_mag, rhs_mag);
        let lhs_nan = self.s.m.cmp(c, pred::IUGT, lhs_mag, infinity);
        let product = self.s.m.select(c, lhs_nan, lhs, product);
        let lhs_zero = self.s.m.cmp(c, pred::IULT, lhs_mag, minimum);
        let rhs_zero = self.s.m.cmp(c, pred::IULT, rhs_mag, minimum);
        let either_zero = self.s.m.bin(c, Bin::Or, lhs_zero, rhs_zero);
        let product = self.s.m.select(c, either_zero, zero, product);
        let product = self.s.m.select(c, invalid_rhs, lowest, product);
        let mut result = self.s.m.select(c, limit_ok, product, lowest);

        if v.omod != 0 {
            result = self.scale_lighting_result(result, v.omod);
        }
        if v.clamp {
            let one = self.s.m.i32(1.0f32.to_bits());
            let too_large = self.s.m.cmp(c, pred::IUGT, result, one);
            let clamped = self.s.m.select(c, too_large, one, result);
            let nonnegative = self.s.m.cmp(c, pred::IULE, result, infinity);
            result = self.s.m.select(c, nonnegative, clamped, zero);
        }
        self.write(inst, Reg::V(reg), result, guard);
        Ok(())
    }

    fn lighting_product(&mut self, lhs: V, rhs: V, lhs_mag: V, rhs_mag: V) -> V {
        let c = self.cur;
        let a = self.as_float(lhs);
        let b = self.as_float(rhs);
        let product = self.s.m.bin(c, Bin::Mul, a, b);
        let product = self.as_int(product);

        // RDNA2 checks tininess after rounding to 24 significant bits.
        // When a product is near 2^-126, multiplying its smaller normal
        // factor by 2^64 is exact and leaves the scaled product far above
        // the host's underflow boundary. See AnyPS5 #2839.
        let lhs_smaller = self.s.m.cmp(c, pred::IULE, lhs_mag, rhs_mag);
        let small = self.s.m.select(c, lhs_smaller, a, b);
        let large = self.s.m.select(c, lhs_smaller, b, a);
        let scale = self.s.m.f32(0x5f80_0000); // 2^64
        let small_scaled = self.s.m.bin(c, Bin::Mul, small, scale);
        let scaled = self.s.m.bin(c, Bin::Mul, small_scaled, large);
        let scaled = self.as_int(scaled);
        let magnitude = self.s.m.i32(!SIGN);
        let scaled_mag = self.s.m.bin(c, Bin::And, scaled, magnitude);
        let edge = self.s.m.i32(0x2080_0000); // 2^-126 * 2^64
        let tiny_or_edge = self.s.m.cmp(c, pred::IULE, scaled_mag, edge);
        let at_edge = self.s.m.cmp(c, pred::EQ, scaled_mag, edge);

        let sign_mask = self.s.m.i32(SIGN);
        let sign_bits = self.s.m.bin(c, Bin::Xor, lhs, rhs);
        let sign = self.s.m.bin(c, Bin::And, sign_bits, sign_mask);
        let min_normal = self.s.m.i32(MIN_NORMAL);
        let signed_min = self.s.m.bin(c, Bin::Or, sign, min_normal);
        let flushed = self.s.m.select(c, at_edge, signed_min, sign);
        self.s.m.select(c, tiny_or_edge, flushed, product)
    }

    /// Scaling by two or four changes only the exponent until overflow.
    /// Halving exponent one flushes the result to a signed zero. An existing
    /// zero becomes +0, and an infinity or a NaN keeps its original bits.
    fn scale_lighting_result(&mut self, result: V, omod: u8) -> V {
        let c = self.cur;
        let magnitude = self.s.m.i32(!SIGN);
        let sign_mask = self.s.m.i32(SIGN);
        let mag = self.s.m.bin(c, Bin::And, result, magnitude);
        let sign = self.s.m.bin(c, Bin::And, result, sign_mask);
        let infinity = self.s.m.i32(INFINITY);
        let scaled = if omod == 3 {
            let two_min = self.s.m.i32(2 * MIN_NORMAL);
            let step = self.s.m.i32(MIN_NORMAL);
            let tiny = self.s.m.cmp(c, pred::IULT, mag, two_min);
            let half = self.s.m.bin(c, Bin::Sub, result, step);
            self.s.m.select(c, tiny, sign, half)
        } else {
            let step = u32::from(omod) * MIN_NORMAL;
            let first_overflow = self.s.m.i32(INFINITY - step);
            let step = self.s.m.i32(step);
            let overflow = self.s.m.cmp(c, pred::IUGE, mag, first_overflow);
            let signed_inf = self.s.m.bin(c, Bin::Or, sign, infinity);
            let larger = self.s.m.bin(c, Bin::Add, result, step);
            self.s.m.select(c, overflow, signed_inf, larger)
        };
        let zero = self.s.m.i32(0);
        let is_zero = self.s.m.cmp(c, pred::EQ, mag, zero);
        let scaled = self.s.m.select(c, is_zero, zero, scaled);
        let special = self.s.m.cmp(c, pred::IUGE, mag, infinity);
        self.s.m.select(c, special, result, scaled)
    }
}
