//! 16-bit VALU and packed VOP3P lowering (`v_*_f16`, `v_*_u16`, `v_*_i16`, `v_pk_*`),
//! a child module so it shares `Lower<'a>` state without growing `lower.rs`.

use super::{
    Attr, Bin, Cast, Fields, Inst, Lower, Operand, Reg, V, Why, f16_nearest_on, host_native16, pred,
};

impl<'a> Lower<'a> {
    /// Whether 16-bit float arithmetic is done in native `half` here.
    fn native16(&self) -> bool {
        host_native16()
    }

    /// The low 16 bits of `x` as a `half`.
    fn as_half(&mut self, x: V) -> V {
        let c = self.cur;
        let i16t = self.s.m.int(16);
        let t = self.s.m.cast(c, Cast::Trunc, x, i16t);
        let ht = self.s.m.half();
        self.s.m.cast(c, Cast::Bitcast, t, ht)
    }

    /// The bits of the `half` `h` in the low half of an i32, the high half zero.
    fn half_bits(&mut self, h: V) -> V {
        let c = self.cur;
        let i16t = self.s.m.int(16);
        let t = self.s.m.cast(c, Cast::Bitcast, h, i16t);
        self.s.m.cast(c, Cast::ZExt, t, self.s.t.i32)
    }

    /// A `dx.op` on `half` operands given as bits (the low 16 of each), its result as bits.
    fn half_op(&mut self, name: &str, op: u32, xs: &[V]) -> V {
        let hs: Vec<V> = xs.iter().map(|&x| self.as_half(x)).collect();
        let ht = self.s.m.half();
        let r = self.s.op(self.cur, name, Attr::ReadNone, ht, op, &hs);
        self.half_bits(r)
    }

    /// `a op b` in `half`, of f16 bits, as bits.
    fn half_bin(&mut self, op: Bin, a: V, b: V) -> V {
        let (a, b) = (self.as_half(a), self.as_half(b));
        let r = self.s.m.bin(self.cur, op, a, b);
        self.half_bits(r)
    }

    /// `a * b + c` of f16 bits in `half`, rounded once where the host fuses its 16-bit `FMad`.
    fn fma16(&mut self, a: V, b: V, c: V) -> V {
        self.half_op("tertiary.f16", 46, &[a, b, c])
    }

    /// [`Self::max_min`] of f16 bits (the high halves zero) in `half`: +0 unless both are -0 (max),
    /// -0 unless neither (min); keeps 16-bit denormals.
    fn max_min16(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.half_op("binary.f16", if max { 35 } else { 36 }, &[a, b]);
        let either = self.s.m.bin(c, Bin::Or, a, b);
        let magnitude = self.s.m.i32(0x7FFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, a, b);
        self.s.m.select(c, zeros, sign, r)
    }

    /// Whether the f16 bits `x` (the high half zero) are a NaN.
    fn nan16(&mut self, x: V) -> V {
        let c = self.cur;
        let magnitude = self.s.m.i32(0x7FFF);
        let a = self.s.m.bin(c, Bin::And, x, magnitude);
        let inf = self.s.m.i32(0x7C00);
        self.s.m.cmp(c, pred::IUGT, a, inf)
    }

    /// [`Self::med3`] of f16 bits in `half`.
    fn med3_16(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.max_min16(false, a, b);
        let hi = self.max_min16(true, a, b);
        let t = self.max_min16(false, hi, third);
        let median = self.max_min16(true, lo, t);
        let least = self.max_min16(false, lo, third);
        let (na, nb, nc) = (self.nan16(a), self.nan16(b), self.nan16(third));
        let nan = self.s.m.bin(c, Bin::Or, na, nb);
        let nan = self.s.m.bin(c, Bin::Or, nan, nc);
        self.s.m.select(c, nan, least, median)
    }

    /// Clamp of an f16 result (bits, the high half zero) to [0, 1]: +0 for NaN and -0.
    fn clamp16(&mut self, x: V) -> V {
        let c = self.cur;
        let inf = self.s.m.i32(0x7C00);
        let bad = self.s.m.cmp(c, pred::IUGT, x, inf);
        let one = self.s.m.i32(0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, x, one);
        let r = self.s.m.select(c, big, one, x);
        let zero = self.s.m.i32(0);
        self.s.m.select(c, bad, zero, r)
    }

    /// An f16 source's bits (the high half zero) with its abs/neg modifiers.
    fn half_bits_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut x = self.half_src(inst, k)?;
        let c = self.cur;
        if v.abs >> k & 1 != 0 {
            let m = self.s.m.i32(0x7FFF);
            x = self.s.m.bin(c, Bin::And, x, m);
        }
        if v.neg >> k & 1 != 0 {
            let m = self.s.m.i32(0x8000);
            x = self.s.m.bin(c, Bin::Xor, x, m);
        }
        Ok(x)
    }

    /// The accumulator of `v_fmac_f16`/`v_mac_f16`: the destination's written half as bits.
    fn mac_acc_bits(&mut self, inst: &Inst) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("an accumulator");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
        let x = if v.opsel >> 3 & 1 != 0 {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Native `half` lowering for f16 VALU ops without output modifiers (`omod == 0`).
    fn valu16_native(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const NATIVE: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
            "v_fma_f16",
            "v_fmac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
        ];
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        if v.omod != 0 || !self.native16() || !NATIVE.contains(&name) {
            return None;
        }
        Some(self.valu16_native_op(inst, name, guard))
    }

    fn valu16_native_op(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let (a, b) = (self.half_bits_src(inst, 0)?, self.half_bits_src(inst, 1)?);
        let r = match name {
            "v_add_f16" => self.half_bin(Bin::Add, a, b),
            "v_sub_f16" => self.half_bin(Bin::Sub, a, b),
            "v_subrev_f16" => self.half_bin(Bin::Sub, b, a),
            "v_mul_f16" => self.half_bin(Bin::Mul, a, b),
            "v_max_f16" => self.max_min16(true, a, b),
            "v_min_f16" => self.max_min16(false, a, b),
            _ => {
                let third = if name == "v_fmac_f16" {
                    self.mac_acc_bits(inst)?
                } else {
                    self.half_bits_src(inst, 2)?
                };
                match name {
                    "v_fma_f16" | "v_fmac_f16" => self.fma16(a, b, third),
                    "v_max3_f16" => {
                        let t = self.max_min16(true, a, b);
                        self.max_min16(true, t, third)
                    }
                    "v_min3_f16" => {
                        let t = self.max_min16(false, a, b);
                        self.max_min16(false, t, third)
                    }
                    _ => self.med3_16(a, b, third),
                }
            }
        };
        let r = if v.clamp { self.clamp16(r) } else { r };
        self.write_half(inst, r, guard)
    }
    ///products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    pub(super) fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
            "v_div_fixup_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if let Some(r) = self.valu16_native(inst, name, guard) {
            return r;
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            match v.omod {
                0 => {}
                k => {
                    let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                    let k = l.s.m.f32(factor.to_bits());
                    f = l.s.m.bin(l.cur, Bin::Mul, f, k);
                }
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            // The console rounds to nearest; the host's conversion may round toward zero
            // (NVIDIA's does), one step of the last place lower where the two differ.
            // shortcut: the host's conversion unless `VICTORY_F16_NEAREST=1`
            // ([`f16_nearest_on`]), which costs a dozen instructions per result.
            let h = if f16_nearest_on() {
                l.f16_nearest(f)
            } else {
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            };
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let x = self.mac_acc_bits(inst)?;
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                "v_div_fixup_f16" => self.div_fixup(a, b, acc),
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_u16_f16" | "v_cvt_i16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let a = self.half_fsrc(inst, 0)?;
            let (lo, hi) = if name == "v_cvt_u16_f16" {
                (0.0f32, 65535.0f32)
            } else {
                (-32768.0, 32767.0)
            };
            let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
            let t = self.binary(35, a, lo);
            let t = self.binary(36, t, hi);
            let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
            let nan = self.s.m.cmp(c, pred::UNO, a, a);
            let z = self.s.m.i32(0);
            let r = self.s.m.select(c, nan, z, r);
            return self.write_half(inst, r, guard);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = {
            let (mut a, mut b) = (self.half_src(inst, 0)?, self.half_src(inst, 1)?);
            let k16 = self.s.m.i32(16);
            let sext = |l: &mut Self, x: V| {
                let t = l.s.m.bin(c, Bin::Shl, x, k16);
                l.s.m.bin(c, Bin::AShr, t, k16)
            };
            if name.ends_with("_i16") {
                a = sext(self, a);
                b = sext(self, b);
            }
            let k15 = self.s.m.i32(15);
            match name {
                "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                    // src0 is the shift (4 bits), src1 the value.
                    let sh = self.s.m.bin(c, Bin::And, a, k15);
                    let x = if name == "v_ashrrev_i16" {
                        sext(self, b)
                    } else {
                        b
                    };
                    let op = match name {
                        "v_lshlrev_b16" => Bin::Shl,
                        "v_lshrrev_b16" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    self.s.m.bin(c, op, x, sh)
                }
                _ => {
                    let p = match name {
                        "v_max_u16" => pred::IUGT,
                        "v_min_u16" => pred::IULT,
                        "v_max_i16" => pred::SGT,
                        _ => pred::SLT,
                    };
                    let t = self.s.m.cmp(c, p, a, b);
                    self.s.m.select(c, t, a, b)
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: exact f16 * {32767, 65535}
    /// rounded to nearest even; NaN gives 0.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// `v_div_fixup`'s special cases, on f32 values.
    fn div_fixup(&mut self, q: V, den: V, num: V) -> V {
        let c = self.cur;
        let (qi, di, ni) = (self.as_int(q), self.as_int(den), self.as_int(num));
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let (ksign, kabs, kinf, kqnan, k0) = (
            k(self, 0x8000_0000),
            k(self, 0x7FFF_FFFF),
            k(self, 0x7F80_0000),
            k(self, 0x0040_0000),
            k(self, 0),
        );
        let x = self.s.m.bin(c, Bin::Xor, ni, di);
        let sign = self.s.m.bin(c, Bin::And, x, ksign);
        let q_abs = self.s.m.bin(c, Bin::And, qi, kabs);
        let d_abs = self.s.m.bin(c, Bin::And, di, kabs);
        let n_abs = self.s.m.bin(c, Bin::And, ni, kabs);
        let d_nan = self.s.m.cmp(c, pred::IUGT, d_abs, kinf);
        let n_nan = self.s.m.cmp(c, pred::IUGT, n_abs, kinf);
        let d_inf = self.s.m.cmp(c, pred::EQ, d_abs, kinf);
        let n_inf = self.s.m.cmp(c, pred::EQ, n_abs, kinf);
        let d_zero = self.s.m.cmp(c, pred::EQ, d_abs, k0);
        let n_zero = self.s.m.cmp(c, pred::EQ, n_abs, k0);
        let both_zero = self.s.m.bin(c, Bin::And, d_zero, n_zero);
        let both_inf = self.s.m.bin(c, Bin::And, d_inf, n_inf);
        let invalid = self.s.m.bin(c, Bin::Or, both_zero, both_inf);
        let to_inf = self.s.m.bin(c, Bin::Or, d_zero, n_inf);
        let to_zero = self.s.m.bin(c, Bin::Or, d_inf, n_zero);
        let r = self.s.m.bin(c, Bin::Or, sign, q_abs);
        let signed_inf = self.s.m.bin(c, Bin::Or, sign, kinf);
        let q_overflow = self.s.m.cmp(c, pred::IUGE, q_abs, kinf);
        let r = self.s.m.select(c, q_overflow, signed_inf, r);
        let r = self.s.m.select(c, to_zero, sign, r);
        let r = self.s.m.select(c, to_inf, signed_inf, r);
        let default_nan = k(self, 0x7FC0_0000);
        let r = self.s.m.select(c, invalid, default_nan, r);
        let d_q = self.s.m.bin(c, Bin::Or, di, kqnan);
        let r = self.s.m.select(c, d_nan, d_q, r);
        let n_q = self.s.m.bin(c, Bin::Or, ni, kqnan);
        let r = self.s.m.select(c, n_nan, n_q, r);
        self.as_float(r)
    }

    pub(in crate::emit) fn packed_src(
        &mut self,
        inst: &Inst,
        k: usize,
        high: bool,
        float: bool,
    ) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    pub(super) fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        let saturates = matches!(
            op,
            "add_u16" | "sub_u16" | "mad_u16" | "add_i16" | "sub_i16" | "mad_i16"
        );
        if v.omod != 0 || !float && ((v.clamp && !saturates) || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let native = float && self.native16();
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if native {
                    let neg = if high { v.neg_hi } else { v.neg };
                    xs.push(if neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        self.s.m.bin(c, Bin::Xor, h, sign)
                    } else {
                        h
                    });
                } else if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        f = self.fneg(f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if native {
                let r = match op {
                    "add_f16" => self.half_bin(Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.half_bin(Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min16(false, xs[0], xs[1]),
                    "max_f16" => self.max_min16(true, xs[0], xs[1]),
                    _ => self.fma16(xs[0], xs[1], xs[2]),
                };
                if v.clamp { self.clamp16(r) } else { r }
            } else if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                let r = match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                };
                if v.clamp && saturates {
                    match op {
                        "add_u16" | "mad_u16" => {
                            let hi = self.s.m.i32(0xFFFF);
                            let over = self.s.m.cmp(c, pred::IUGT, r, hi);
                            self.s.m.select(c, over, hi, r)
                        }
                        "sub_u16" => {
                            let z = self.s.m.i32(0);
                            let under = self.s.m.cmp(c, pred::SLT, r, z);
                            self.s.m.select(c, under, z, r)
                        }
                        _ => {
                            let lo = self.s.m.i32(i32::from(i16::MIN) as u32);
                            let hi = self.s.m.i32(i32::from(i16::MAX) as u32);
                            let under = self.s.m.cmp(c, pred::SLT, r, lo);
                            let t = self.s.m.select(c, under, lo, r);
                            let over = self.s.m.cmp(c, pred::SGT, t, hi);
                            self.s.m.select(c, over, hi, t)
                        }
                    }
                } else {
                    r
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }
}
