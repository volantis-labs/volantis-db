//! fp32 CPU kernels: binary16 widening, dot products and tiled linear layers.
//!
//! Every SIMD path is chosen at run time. Callers pass slices only; each
//! kernel checks lengths before it enters `unsafe` code, and each `unsafe`
//! block reads and writes inside those checked lengths.

use std::sync::OnceLock;

use crate::bundle::TensorDtype;
use crate::runtime::RuntimeError;

/// fp32 values in one per-worker weight tile (64 KiB).
pub(crate) const TILE_FLOATS: usize = 16_384;

/// Kernel family selected once per process from the running CPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kernels {
    /// Portable Rust. It is also the reference for the SIMD paths.
    Scalar,
    /// AVX2 + FMA dot products and F16C widening.
    #[cfg(target_arch = "x86_64")]
    Avx2Fma,
    /// NEON dot products. FCVTL widening needs the FP16 extension.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon {
        /// FEAT_FP16 is present, so FCVTL may widen halves.
        fp16: bool,
    },
}

impl Kernels {
    /// Returns the best kernel family this CPU supports.
    pub(crate) fn detect() -> Self {
        static SELECTED: OnceLock<Kernels> = OnceLock::new();
        *SELECTED.get_or_init(|| {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
            {
                return Self::Avx2Fma;
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Self::Neon {
                    fp16: std::arch::is_aarch64_feature_detected!("fp16"),
                };
            }
            Self::Scalar
        })
    }

    /// Returns every kernel family this CPU can run, the scalar one first.
    #[cfg(test)]
    pub(crate) fn available() -> Vec<Self> {
        let mut kernels = vec![Self::Scalar];
        let best = Self::detect();
        if best != Self::Scalar {
            kernels.push(best);
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if best == (Self::Neon { fp16: true }) {
            kernels.push(Self::Neon { fp16: false });
        }
        kernels
    }

    /// Stable name for diagnostics and evidence.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2Fma => "avx2+fma+f16c",
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon { fp16: true } => "neon+fp16",
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon { fp16: false } => "neon",
        }
    }

    /// Widens little-endian tensor values into `dst`.
    ///
    /// `src` holds exactly `dst.len()` values of `dtype`.
    pub(crate) fn widen(
        self,
        dtype: TensorDtype,
        src: &[u8],
        dst: &mut [f32],
    ) -> Result<(), RuntimeError> {
        let width = match dtype {
            TensorDtype::F32 => 4,
            TensorDtype::F16 => 2,
            other => return Err(RuntimeError::UnsupportedDtype(format!("{other:?}"))),
        };
        if dst.len().checked_mul(width) != Some(src.len()) {
            return Err(RuntimeError::Shape(format!(
                "cannot widen {} bytes into {} values",
                src.len(),
                dst.len()
            )));
        }
        if dtype == TensorDtype::F32 {
            for (value, bytes) in dst.iter_mut().zip(src.chunks_exact(4)) {
                let mut word = [0_u8; 4];
                word.copy_from_slice(bytes);
                *value = f32::from_le_bytes(word);
            }
            return Ok(());
        }
        match self {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: detection established AVX2, FMA and F16C. The length
            // check above gives src exactly 2 * dst.len() bytes.
            Self::Avx2Fma => unsafe { x86::widen_f16(src, dst) },
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            // SAFETY: detection established NEON and FEAT_FP16. The length
            // check above gives src exactly 2 * dst.len() bytes.
            Self::Neon { fp16: true } => unsafe { neon::widen_f16(src, dst) },
            _ => widen_f16_scalar(src, dst),
        }
        Ok(())
    }

    /// Returns the fp32 dot product over the shorter of the two slices.
    ///
    /// Each kernel family adds the products in one fixed order for a given
    /// length, so equal inputs always give equal bits.
    pub(crate) fn dot(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: detection established AVX2 and FMA; the kernel reads
            // only the first min(a.len(), b.len()) values of each slice.
            Self::Avx2Fma => unsafe { x86::dot(a, b) },
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            // SAFETY: detection established NEON; the kernel reads only the
            // first min(a.len(), b.len()) values of each slice.
            Self::Neon { .. } => unsafe { neon::dot(a, b) },
            Self::Scalar => dot_scalar(a, b),
        }
    }
}

/// Converts one IEEE 754 binary16 value to binary32, exactly.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;
    match exponent {
        0 if fraction == 0 => f32::from_bits(sign),
        0 => {
            // A subnormal is fraction * 2^-24; both steps are exact in f32.
            let magnitude = f32::from(fraction) * 2.0_f32.powi(-24);
            if sign == 0 { magnitude } else { -magnitude }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (u32::from(fraction) << 13)),
        _ => f32::from_bits(
            sign | (u32::from(exponent + (127 - 15)) << 23) | (u32::from(fraction) << 13),
        ),
    }
}

fn widen_f16_scalar(src: &[u8], dst: &mut [f32]) {
    for (value, bytes) in dst.iter_mut().zip(src.chunks_exact(2)) {
        let mut word = [0_u8; 2];
        word.copy_from_slice(bytes);
        *value = f16_to_f32(u16::from_le_bytes(word));
    }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    // Eight independent sums let the compiler use the baseline vector unit.
    let mut sums = [0.0_f32; 8];
    let mut a_blocks = a.chunks_exact(8);
    let mut b_blocks = b.chunks_exact(8);
    for (x, y) in a_blocks.by_ref().zip(b_blocks.by_ref()) {
        for ((sum, p), q) in sums.iter_mut().zip(x).zip(y) {
            *sum += p * q;
        }
    }
    let [s0, s1, s2, s3, s4, s5, s6, s7] = sums;
    let mut total = ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7));
    for (p, q) in a_blocks.remainder().iter().zip(b_blocks.remainder()) {
        total += p * q;
    }
    total
}

/// One checked row-major weight matrix in the mapped bundle.
#[derive(Clone, Copy)]
pub(crate) struct WeightView<'a> {
    /// Exact tensor bytes, `rows * cols` values of `dtype`.
    pub(crate) bytes: &'a [u8],
    /// Stored scalar type.
    pub(crate) dtype: TensorDtype,
    /// Output features.
    pub(crate) rows: usize,
    /// Input features.
    pub(crate) cols: usize,
}

impl WeightView<'_> {
    fn element_bytes(&self) -> Result<usize, RuntimeError> {
        match self.dtype {
            TensorDtype::F32 => Ok(4),
            TensorDtype::F16 => Ok(2),
            other => Err(RuntimeError::UnsupportedDtype(format!("{other:?}"))),
        }
    }

    /// Widens weight rows `rows` into the front of `tile`.
    pub(crate) fn widen_rows(
        &self,
        kernels: Kernels,
        rows: std::ops::Range<usize>,
        tile: &mut [f32],
    ) -> Result<usize, RuntimeError> {
        let element = self.element_bytes()?;
        let values = rows
            .len()
            .checked_mul(self.cols)
            .ok_or_else(|| RuntimeError::Shape("weight tile overflow".to_owned()))?;
        let start = rows
            .start
            .checked_mul(self.cols)
            .and_then(|value| value.checked_mul(element))
            .ok_or_else(|| RuntimeError::Shape("weight offset overflow".to_owned()))?;
        let end = values
            .checked_mul(element)
            .and_then(|value| value.checked_add(start))
            .ok_or_else(|| RuntimeError::Shape("weight offset overflow".to_owned()))?;
        let src = self
            .bytes
            .get(start..end)
            .ok_or_else(|| RuntimeError::Shape("weight rows are out of range".to_owned()))?;
        let dst = tile
            .get_mut(..values)
            .ok_or_else(|| RuntimeError::Shape("weight tile is too small".to_owned()))?;
        kernels.widen(self.dtype, src, dst)?;
        Ok(values)
    }
}

/// Computes `out = x * W^T + bias` for `x` with `W.cols` columns.
///
/// Weight rows are widened one tile at a time into `tile` and used for every
/// input row before the next tile. Each output is `dot(x_row, w_row) + bias`,
/// so the result does not depend on the tile size or on how callers split rows.
pub(crate) fn linear(
    kernels: Kernels,
    x: &[f32],
    weight: WeightView<'_>,
    bias: Option<&[f32]>,
    out: &mut [f32],
    tile: &mut [f32],
) -> Result<(), RuntimeError> {
    if weight.cols == 0 || weight.rows == 0 || !x.len().is_multiple_of(weight.cols) {
        return Err(RuntimeError::Shape(
            "linear input width does not match the weight".to_owned(),
        ));
    }
    let tokens = x.len() / weight.cols;
    if tokens.checked_mul(weight.rows) != Some(out.len()) {
        return Err(RuntimeError::Shape(
            "linear output does not match the weight".to_owned(),
        ));
    }
    if bias.is_some_and(|bias| bias.len() != weight.rows) {
        return Err(RuntimeError::Shape(
            "linear bias does not match the weight".to_owned(),
        ));
    }
    let tile_rows = (tile.len() / weight.cols).min(weight.rows);
    if tile_rows == 0 {
        return Err(RuntimeError::Shape(
            "weight tile holds no complete row".to_owned(),
        ));
    }
    let mut first = 0;
    while first < weight.rows {
        let last = first.saturating_add(tile_rows).min(weight.rows);
        let widened = weight.widen_rows(kernels, first..last, tile)?;
        let block = tile
            .get(..widened)
            .ok_or_else(|| RuntimeError::Shape("weight tile is too small".to_owned()))?;
        for (input, output) in x
            .chunks_exact(weight.cols)
            .zip(out.chunks_exact_mut(weight.rows))
        {
            let outputs = output
                .get_mut(first..last)
                .ok_or_else(|| RuntimeError::Shape("linear output row is short".to_owned()))?;
            for (index, (value, row)) in outputs
                .iter_mut()
                .zip(block.chunks_exact(weight.cols))
                .enumerate()
            {
                let shift = bias
                    .and_then(|bias| bias.get(first.saturating_add(index)))
                    .copied()
                    .unwrap_or(0.0);
                *value = kernels.dot(input, row) + shift;
            }
        }
        first = last;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::{
        __m256, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps, _mm_loadu_si128, _mm256_add_ps,
        _mm256_castps256_ps128, _mm256_cvtph_ps, _mm256_extractf128_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };

    /// Widens `dst.len()` halves from `src`.
    ///
    /// # Safety
    /// The CPU supports AVX2, FMA and F16C, and `src.len() == 2 * dst.len()`.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub(super) unsafe fn widen_f16(src: &[u8], dst: &mut [f32]) {
        let mut src_blocks = src.chunks_exact(16);
        let mut dst_blocks = dst.chunks_exact_mut(8);
        for (input, output) in src_blocks.by_ref().zip(dst_blocks.by_ref()) {
            // SAFETY: input holds 16 bytes and output holds 8 floats; both
            // the load and the store are unaligned forms.
            unsafe {
                let halves = _mm_loadu_si128(input.as_ptr().cast());
                _mm256_storeu_ps(output.as_mut_ptr(), _mm256_cvtph_ps(halves));
            }
        }
        super::widen_f16_scalar(src_blocks.remainder(), dst_blocks.into_remainder());
    }

    /// # Safety
    /// The CPU supports AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let length = a.len().min(b.len());
        let (a, b) = (
            a.get(..length).unwrap_or(&[]),
            b.get(..length).unwrap_or(&[]),
        );
        let mut a_blocks = a.chunks_exact(32);
        let mut b_blocks = b.chunks_exact(32);
        let mut sums: [__m256; 4] = [_mm256_setzero_ps(); 4];
        for (x, y) in a_blocks.by_ref().zip(b_blocks.by_ref()) {
            for ((sum, p), q) in sums
                .iter_mut()
                .zip(x.chunks_exact(8))
                .zip(y.chunks_exact(8))
            {
                // SAFETY: p and q each hold 8 floats; unaligned loads.
                unsafe {
                    *sum = _mm256_fmadd_ps(
                        _mm256_loadu_ps(p.as_ptr()),
                        _mm256_loadu_ps(q.as_ptr()),
                        *sum,
                    );
                }
            }
        }
        let [mut s0, s1, s2, s3] = sums;
        let mut a_rest = a_blocks.remainder().chunks_exact(8);
        let mut b_rest = b_blocks.remainder().chunks_exact(8);
        for (p, q) in a_rest.by_ref().zip(b_rest.by_ref()) {
            // SAFETY: p and q each hold 8 floats; unaligned loads.
            unsafe {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p.as_ptr()), _mm256_loadu_ps(q.as_ptr()), s0);
            }
        }
        let vector = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        let half = _mm_add_ps(
            _mm256_castps256_ps128(vector),
            _mm256_extractf128_ps::<1>(vector),
        );
        let pair = _mm_hadd_ps(half, half);
        let mut total = _mm_cvtss_f32(_mm_hadd_ps(pair, pair));
        for (p, q) in a_rest.remainder().iter().zip(b_rest.remainder()) {
            total += p * q;
        }
        total
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon {
    use std::arch::aarch64::{
        float32x4_t, uint16x8_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32,
        vld1q_u8, vreinterpretq_u16_u8, vst1q_f32,
    };
    use std::arch::asm;

    /// Widens `dst.len()` halves from `src` with FCVTL and FCVTL2.
    ///
    /// # Safety
    /// The CPU supports NEON and FEAT_FP16, and `src.len() == 2 * dst.len()`.
    #[target_feature(enable = "neon,fp16")]
    pub(super) unsafe fn widen_f16(src: &[u8], dst: &mut [f32]) {
        let mut src_blocks = src.chunks_exact(16);
        let mut dst_blocks = dst.chunks_exact_mut(8);
        for (input, output) in src_blocks.by_ref().zip(dst_blocks.by_ref()) {
            // SAFETY: input holds 16 bytes (eight little-endian halves) and
            // output holds 8 floats. vld1q/vst1q have no alignment demand.
            unsafe {
                let halves = vreinterpretq_u16_u8(vld1q_u8(input.as_ptr()));
                let (low, high) = widen8(halves);
                vst1q_f32(output.as_mut_ptr(), low);
                vst1q_f32(output.as_mut_ptr().wrapping_add(4), high);
            }
        }
        super::widen_f16_scalar(src_blocks.remainder(), dst_blocks.into_remainder());
    }

    #[target_feature(enable = "neon,fp16")]
    unsafe fn widen8(input: uint16x8_t) -> (float32x4_t, float32x4_t) {
        let low: float32x4_t;
        let high: float32x4_t;
        // SAFETY: runtime FP16 detection permits FCVTL/FCVTL2, which convert
        // every binary16 class (subnormals, infinities, NaN) exactly. All
        // operands are vector registers and no memory is touched. Inline
        // assembly avoids the unstable `vcvt_f32_f16` intrinsic, as in
        // volantis-db kernels/neon.rs.
        unsafe {
            asm!(
                "fcvtl {low:v}.4s, {input:v}.4h",
                "fcvtl2 {high:v}.4s, {input:v}.8h",
                low = out(vreg) low,
                high = out(vreg) high,
                input = in(vreg) input,
                options(pure, nomem, nostack)
            );
        }
        (low, high)
    }

    /// # Safety
    /// The CPU supports NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let length = a.len().min(b.len());
        let (a, b) = (
            a.get(..length).unwrap_or(&[]),
            b.get(..length).unwrap_or(&[]),
        );
        let mut a_blocks = a.chunks_exact(16);
        let mut b_blocks = b.chunks_exact(16);
        let mut sums = [vdupq_n_f32(0.0); 4];
        for (x, y) in a_blocks.by_ref().zip(b_blocks.by_ref()) {
            for ((sum, p), q) in sums
                .iter_mut()
                .zip(x.chunks_exact(4))
                .zip(y.chunks_exact(4))
            {
                // SAFETY: p and q each hold 4 floats; vld1q has no alignment demand.
                unsafe {
                    *sum = vfmaq_f32(*sum, vld1q_f32(p.as_ptr()), vld1q_f32(q.as_ptr()));
                }
            }
        }
        let [mut s0, s1, s2, s3] = sums;
        let mut a_rest = a_blocks.remainder().chunks_exact(4);
        let mut b_rest = b_blocks.remainder().chunks_exact(4);
        for (p, q) in a_rest.by_ref().zip(b_rest.by_ref()) {
            // SAFETY: p and q each hold 4 floats; vld1q has no alignment demand.
            unsafe {
                s0 = vfmaq_f32(s0, vld1q_f32(p.as_ptr()), vld1q_f32(q.as_ptr()));
            }
        }
        let vector = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        let mut total = vaddvq_f32(vector);
        for (p, q) in a_rest.remainder().iter().zip(b_rest.remainder()) {
            total += p * q;
        }
        total
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Literal (binary16, binary32) bit pairs, including every class.
    const PAIRS: &[(u16, u32)] = &[
        (0x0000, 0x0000_0000), // +0
        (0x8000, 0x8000_0000), // -0
        (0x0001, 0x3380_0000), // smallest subnormal, 2^-24
        (0x03ff, 0x387f_c000), // largest subnormal
        (0x0400, 0x3880_0000), // smallest normal, 2^-14
        (0x3c00, 0x3f80_0000), // 1
        (0xc000, 0xc000_0000), // -2
        (0x3555, 0x3eaa_a000), // 0.333...
        (0x7bff, 0x477f_e000), // max finite, 65504
        (0xfbff, 0xc77f_e000), // -65504
        (0x7c00, 0x7f80_0000), // +inf
        (0xfc00, 0xff80_0000), // -inf
    ];

    #[test]
    fn wf991_half_tiles() {
        println!("selected CPU kernels: {}", Kernels::detect().name());
        for kernels in Kernels::available() {
            // Literal pairs through the tile widener, at an unaligned start
            // and for lengths on both sides of every vector width.
            for length in [1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 64] {
                for offset in [0, 1] {
                    let mut src = vec![0xa5_u8; offset];
                    let mut expected = Vec::new();
                    for index in 0..length {
                        let (half, single) = PAIRS[index % PAIRS.len()];
                        src.extend_from_slice(&half.to_le_bytes());
                        expected.push(single);
                    }
                    let mut dst = vec![f32::NAN; length];
                    kernels
                        .widen(TensorDtype::F16, &src[offset..], &mut dst)
                        .expect("widen");
                    let bits = dst.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits, expected, "{} length {length}", kernels.name());
                }
            }
            // A quiet NaN stays a NaN with its payload and sign.
            let mut nan = [0.0_f32];
            kernels
                .widen(TensorDtype::F16, &0xfe01_u16.to_le_bytes(), &mut nan)
                .expect("widen NaN");
            assert_eq!(nan[0].to_bits(), 0xffc0_2000, "{}", kernels.name());
            // Bad lengths and dtypes fail before any vector code runs.
            assert!(matches!(
                kernels.widen(TensorDtype::F16, &[0; 3], &mut [0.0; 2]),
                Err(RuntimeError::Shape(_))
            ));
            assert!(matches!(
                kernels.widen(TensorDtype::Bf16, &[0; 4], &mut [0.0; 2]),
                Err(RuntimeError::UnsupportedDtype(_))
            ));
            // Dot products of exact small integers are exact in every order.
            for length in [0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100] {
                let a = (0..length)
                    .map(|i| (i % 7) as f32 - 3.0)
                    .collect::<Vec<_>>();
                let b = (0..length)
                    .map(|i| (i % 5) as f32 + 1.0)
                    .collect::<Vec<_>>();
                let expected: f32 = a.iter().zip(&b).map(|(p, q)| p * q).sum();
                assert_eq!(kernels.dot(&a[..], &b[..]), expected, "{}", kernels.name());
                if length > 1 {
                    // Unaligned views and a longer second slice.
                    assert_eq!(
                        kernels.dot(&a[1..], &b[1..]),
                        a[1..].iter().zip(&b[1..]).map(|(p, q)| p * q).sum::<f32>()
                    );
                }
            }
            // A known linear result: 3 tokens, 5 inputs, 7 outputs with a
            // two-row tile, so tiles end inside the output rows.
            let (tokens, cols, rows) = (3, 5, 7);
            let weights = (0..rows * cols)
                .map(|i| ((i * 3) % 11) as f32 * 0.5 - 2.0)
                .collect::<Vec<_>>();
            // Every weight is k/2 - 2 with 0 <= k <= 10: exact in binary16.
            let halves = weights
                .iter()
                .flat_map(|value| exact_half(*value).to_le_bytes())
                .collect::<Vec<_>>();
            let x = (0..tokens * cols)
                .map(|i| i as f32 - 4.0)
                .collect::<Vec<_>>();
            let bias = (0..rows).map(|i| i as f32).collect::<Vec<_>>();
            let mut out = vec![0.0; tokens * rows];
            let mut tile = vec![0.0; 2 * cols + 1];
            let view = WeightView {
                bytes: &halves,
                dtype: TensorDtype::F16,
                rows,
                cols,
            };
            linear(kernels, &x, view, Some(&bias), &mut out, &mut tile).expect("linear");
            for token in 0..tokens {
                for row in 0..rows {
                    let expected = (0..cols)
                        .map(|c| {
                            f64::from(x[token * cols + c]) * f64::from(weights[row * cols + c])
                        })
                        .sum::<f64>()
                        + f64::from(bias[row]);
                    assert_eq!(f64::from(out[token * rows + row]), expected);
                }
            }
            assert!(matches!(
                linear(kernels, &x[1..], view, None, &mut out, &mut tile),
                Err(RuntimeError::Shape(_))
            ));
            assert!(matches!(
                linear(kernels, &x, view, None, &mut out, &mut tile[..cols - 1]),
                Err(RuntimeError::Shape(_))
            ));
        }
    }

    /// Rounds an f32 that binary16 represents exactly (normal range only).
    fn exact_half(value: f32) -> u16 {
        (0..=u16::MAX)
            .find(|half| f16_to_f32(*half) == value)
            .expect("value is exact in binary16")
    }
}
