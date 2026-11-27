//! IEC 61966-2-1 sRGB encode/decode helpers and sRGB texture UAV RGB
//! conversion for `image_store` and `image_load` (decision 0007).

use super::{Lower, Source, Why};
use crate::emit::bitcode::{Bin, Term, V, pred};
use crate::emit::shader::VIEW_SRGB;

impl Lower<'_> {
    /// IEC 61966-2-1 linear-to-sRGB transfer function for one float channel:
    /// clamps to `[0, 1]` (NaN to 0), uses `12.92 * c` for `c <= 0.0031308`,
    /// and `1.055 * exp2(log2(fmax(c, 1e-12)) / 2.4) - 0.055` otherwise.
    pub(super) fn linear_to_srgb(&mut self, v: V) -> V {
        let c = self.unary(7, v);
        let cur = self.cur;
        let threshold = self.s.m.f32(0.003_130_8f32.to_bits());
        let scale_lo = self.s.m.f32(12.92f32.to_bits());
        let eps = self.s.m.f32(1e-12f32.to_bits());
        let gamma = self.s.m.f32(2.4f32.to_bits());
        let scale_hi = self.s.m.f32(1.055f32.to_bits());
        let bias = self.s.m.f32(0.055f32.to_bits());
        let lo = self.s.m.bin(cur, Bin::Mul, c, scale_lo);
        let safe = self.binary(35, c, eps);
        let lg = self.unary(23, safe);
        let scaled = self.s.m.bin(cur, Bin::SDiv, lg, gamma);
        let pw = self.unary(21, scaled);
        let hi_mul = self.s.m.bin(cur, Bin::Mul, pw, scale_hi);
        let hi = self.s.m.bin(cur, Bin::Sub, hi_mul, bias);
        let is_lo = self.s.m.cmp(cur, pred::OLE, c, threshold);
        self.s.m.select(cur, is_lo, lo, hi)
    }

    /// IEC 61966-2-1 sRGB-to-linear transfer function for one float channel:
    /// clamps to `[0, 1]` (NaN to 0), uses `c / 12.92` for `c <= 0.04045`,
    /// and `exp2(log2((c + 0.055) / 1.055) * 2.4)` otherwise.
    pub(super) fn srgb_to_linear(&mut self, v: V) -> V {
        let c = self.unary(7, v);
        let cur = self.cur;
        let threshold = self.s.m.f32(0.04045f32.to_bits());
        let scale_lo = self.s.m.f32(12.92f32.to_bits());
        let bias = self.s.m.f32(0.055f32.to_bits());
        let scale_hi = self.s.m.f32(1.055f32.to_bits());
        let gamma = self.s.m.f32(2.4f32.to_bits());
        let lo = self.s.m.bin(cur, Bin::SDiv, c, scale_lo);
        let shifted = self.s.m.bin(cur, Bin::Add, c, bias);
        let base = self.s.m.bin(cur, Bin::SDiv, shifted, scale_hi);
        let lg = self.unary(23, base);
        let scaled = self.s.m.bin(cur, Bin::Mul, lg, gamma);
        let hi = self.unary(21, scaled);
        let is_lo = self.s.m.cmp(cur, pred::OLE, c, threshold);
        self.s.m.select(cur, is_lo, lo, hi)
    }

    /// For a heap-indexed texture UAV, branches on the format mark and applies `convert` to each RGB
    /// channel in `rgb`, leaving alpha untouched.
    pub(super) fn convert_srgb_uav_rgb(
        &mut self,
        source: &Source,
        rgb: &mut [V],
        convert: fn(&mut Self, V) -> V,
    ) -> Result<(), Why> {
        if rgb.is_empty() || !super::super::srgb_uav() {
            return Ok(());
        }
        let Some(word) = self.slot_word(source)? else {
            return Ok(());
        };
        let head = self.cur;
        let bit = self.s.m.i32(VIEW_SRGB);
        let zero = self.s.m.i32(0);
        let masked = self.s.m.bin(head, Bin::And, word, bit);
        let is_srgb = self.s.m.cmp(head, pred::NE, masked, zero);
        let srgb_bb = self.s.m.new_block();
        let merge_bb = self.s.m.new_block();
        self.s
            .m
            .terminate(head, Term::CondBr(is_srgb, srgb_bb, merge_bb));
        self.cur = srgb_bb;
        let converted: Vec<V> = rgb.iter().map(|&v| convert(self, v)).collect();
        let srgb_end = self.cur;
        self.s.m.terminate(srgb_end, Term::Br(merge_bb));
        self.cur = merge_bb;
        let f32t = self.s.t.f32;
        for (val, conv) in rgb.iter_mut().zip(converted) {
            let phi = self.s.m.phi(merge_bb, f32t);
            self.s.m.set_phi(phi, vec![(*val, head), (conv, srgb_end)]);
            *val = phi;
        }
        Ok(())
    }
}
