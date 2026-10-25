use std::collections::{HashMap, HashSet};

use polars::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use regex::Regex;

use crate::checks::{
    exact_i128_value, exact_u128_value, int_domain, is_supported_numeric_dtype, CheckSpec,
    IntDomain,
};
use crate::column::ColumnSpec;
use crate::dataframe_checks::{
    cmp_mixed_int_uint, int128_cells, is_temporal_dtype, ns_per_tick, uint128_cells, CompareOp,
    DataFrameCheckSpec,
};
use crate::schema::Schema;

/// Known string patterns from inference: generate a canned valid value.
fn canned_match(pattern: &str) -> Option<String> {
    if pattern.contains("@") {
        Some("user@example.com".to_string())
    } else if pattern.contains("https?") {
        Some("https://example.com/a".to_string())
    } else if pattern.contains("0-9a-fA-F") {
        Some("123e4567-e89b-12d3-a456-426614174000".to_string())
    } else if pattern.contains("\\d{4}") {
        Some("2026-09-27".to_string())
    } else if pattern == r"^\d{5}$" {
        Some("75001".to_string())
    } else if pattern == r"^[A-Z]{2}$" {
        Some("FR".to_string())
    } else {
        None
    }
}

#[derive(Clone, Copy)]
struct NumericBounds {
    lo: f64,
    lo_inclusive: bool,
    hi: f64,
    hi_inclusive: bool,
}

impl NumericBounds {
    fn unbounded() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            lo_inclusive: true,
            hi: f64::INFINITY,
            hi_inclusive: true,
        }
    }

    fn tighten_lo(&mut self, value: f64, inclusive: bool) {
        if value > self.lo || (value == self.lo && !inclusive && self.lo_inclusive) {
            self.lo = value;
            self.lo_inclusive = inclusive;
        }
    }

    fn tighten_hi(&mut self, value: f64, inclusive: bool) {
        if value < self.hi || (value == self.hi && !inclusive && self.hi_inclusive) {
            self.hi = value;
            self.hi_inclusive = inclusive;
        }
    }

    fn is_empty(&self) -> bool {
        self.lo > self.hi || (self.lo == self.hi && (!self.lo_inclusive || !self.hi_inclusive))
    }
}

fn numeric_bounds(specs: &[CheckSpec]) -> NumericBounds {
    let mut bounds = NumericBounds::unbounded();
    for spec in specs {
        match spec {
            CheckSpec::GreaterThan(v) => bounds.tighten_lo(*v, false),
            CheckSpec::GreaterThanOrEqual(v) => bounds.tighten_lo(*v, true),
            CheckSpec::LessThan(v) => bounds.tighten_hi(*v, false),
            CheckSpec::LessThanOrEqual(v) => bounds.tighten_hi(*v, true),
            CheckSpec::Between(lo, hi) => {
                bounds.tighten_lo(*lo, true);
                bounds.tighten_hi(*hi, true);
            }
            _ => {}
        }
    }
    bounds
}

fn sample_f64_in(
    rng: &mut ChaCha8Rng,
    bounds: &NumericBounds,
    fallback_lo: f64,
    fallback_hi: f64,
    min_width: f64,
) -> PolarsResult<f64> {
    if bounds.is_empty() {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable numeric bounds in schema".into(),
        ));
    }
    // Unbounded sides inherit a window from the fallback range, widened to at
    // least min_width (e.g. room for `n` distinct values under `unique()`).
    // Deriving the window from the bounded side keeps one-sided constraints
    // such as `gt(2000)` satisfiable.
    let span = (fallback_hi - fallback_lo).max(min_width).max(1.0);
    let mut hi = if bounds.hi.is_finite() {
        if bounds.hi_inclusive {
            bounds.hi
        } else {
            bounds.hi.next_down()
        }
    } else {
        f64::INFINITY
    };
    let mut lo = if bounds.lo.is_finite() {
        if bounds.lo_inclusive {
            bounds.lo
        } else {
            bounds.lo.next_up()
        }
    } else {
        f64::NEG_INFINITY
    };
    if !lo.is_finite() && hi.is_finite() {
        lo = hi - span;
    } else if lo.is_finite() && !hi.is_finite() {
        hi = lo + span;
    } else if !lo.is_finite() {
        lo = fallback_lo;
        hi = lo + span;
    }
    if lo > hi {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable numeric bounds in schema".into(),
        ));
    }
    if (hi - lo).is_infinite() {
        return Ok(lo + rng.random_range(0.0..1.0));
    }
    if lo == hi {
        return Ok(lo);
    }
    Ok(rng.random_range(lo..=hi))
}

fn one_of_values(specs: &[CheckSpec]) -> Option<Vec<String>> {
    for spec in specs {
        if let CheckSpec::OneOf(values) = spec {
            return Some(values.clone());
        }
    }
    None
}

fn matches_patterns(specs: &[CheckSpec]) -> Vec<String> {
    specs
        .iter()
        .filter_map(|spec| match spec {
            CheckSpec::Matches(pattern) => Some(pattern.clone()),
            _ => None,
        })
        .collect()
}

fn one_of_numeric_values(specs: &[CheckSpec]) -> Option<Vec<f64>> {
    let mut values: Option<Vec<f64>> = None;
    for spec in specs {
        if let CheckSpec::OneOfNumeric(next) = spec {
            values = Some(match values {
                // Intersect multiple one_of constraints; disjoint sets are
                // reported as unsatisfiable by callers when nothing remains.
                Some(mut allowed) => {
                    allowed.retain(|value| next.contains(value));
                    allowed
                }
                None => next.clone(),
            });
        }
    }
    values
}

/// String constraints derived from the string checks: every generated value
/// must satisfy the length bounds, never be blank, start with every locked
/// prefix, and end with every locked suffix.
#[derive(Default)]
struct StringConstraints {
    min_len: Option<usize>,
    max_len: Option<usize>,
    prefixes: Vec<String>,
    suffixes: Vec<String>,
    forbid_blank: bool,
}

fn string_constraints(specs: &[CheckSpec]) -> StringConstraints {
    let mut out = StringConstraints::default();
    for spec in specs {
        match spec {
            CheckSpec::StrLengthBetween(lo, hi) => {
                out.min_len = Some(out.min_len.map_or(*lo, |min| min.max(*lo)));
                out.max_len = Some(out.max_len.map_or(*hi, |max| max.min(*hi)));
            }
            CheckSpec::NotBlank => out.forbid_blank = true,
            CheckSpec::StartsWith(prefix) => out.prefixes.push(prefix.clone()),
            CheckSpec::EndsWith(suffix) => out.suffixes.push(suffix.clone()),
            _ => {}
        }
    }
    out
}

fn longest_constraint(values: &[String]) -> Option<&str> {
    values
        .iter()
        .max_by_key(|value| value.chars().count())
        .map(|value| value.as_str())
}

fn satisfies_string_constraints(value: &str, constraints: &StringConstraints) -> bool {
    if constraints.forbid_blank && value.trim().is_empty() {
        return false;
    }
    let len = value.chars().count();
    if let Some(min_len) = constraints.min_len {
        if len < min_len {
            return false;
        }
    }
    if let Some(max_len) = constraints.max_len {
        if len > max_len {
            return false;
        }
    }
    constraints
        .prefixes
        .iter()
        .all(|prefix| value.starts_with(prefix))
        && constraints
            .suffixes
            .iter()
            .all(|suffix| value.ends_with(suffix))
}

/// Builds the smallest value satisfying the constraints: the longest
/// prefix/suffix plus a filler core. Returns `None` when constraints
/// conflict (e.g. a suffix alone longer than `max_len`), so callers report
/// unsatisfiability honestly instead of emitting invalid values.
fn fitting_string(constraints: &StringConstraints) -> Option<String> {
    let prefix = longest_constraint(&constraints.prefixes).unwrap_or("");
    let suffix = longest_constraint(&constraints.suffixes).unwrap_or("");
    let overhead = prefix.chars().count() + suffix.chars().count();
    let min_len = constraints.min_len.unwrap_or(0);
    let max_len = constraints.max_len.unwrap_or(usize::MAX);
    if min_len > max_len || overhead > max_len {
        return None;
    }
    let core = if min_len > overhead + 3 {
        min_len - overhead
    } else {
        3
    };
    let value = format!("{prefix}{}{suffix}", "x".repeat(core));
    satisfies_string_constraints(&value, constraints).then_some(value)
}

/// Extends (or rejects) a candidate so it satisfies the string constraints.
/// Extension only ever adds characters; when the result would exceed
/// `max_len`, `None` signals that the candidate cannot be repaired.
fn enforce_string_constraints(base: String, constraints: &StringConstraints) -> Option<String> {
    let mut value = base;
    if let Some(prefix) = longest_constraint(&constraints.prefixes) {
        if !value.starts_with(prefix) {
            value = format!("{prefix}{value}");
        }
    }
    if let Some(min_len) = constraints.min_len {
        if value.chars().count() < min_len {
            // Pad the core, before any suffix, so the suffix stays last.
            value = format!("{value}{}", "x".repeat(min_len - value.chars().count()));
        }
    }
    if let Some(suffix) = longest_constraint(&constraints.suffixes) {
        if !value.ends_with(suffix) {
            value = format!("{value}{suffix}");
        }
    }
    satisfies_string_constraints(&value, constraints).then_some(value)
}

fn has_unique(specs: &[CheckSpec]) -> bool {
    specs
        .iter()
        .any(|spec| matches!(spec, CheckSpec::UniqueValues))
}

fn list_len_bounds(specs: &[CheckSpec]) -> (usize, usize) {
    let mut lo = 0usize;
    let mut hi = 3usize;
    let mut found = false;
    for spec in specs {
        match spec {
            CheckSpec::ListLenBetween(min_len, max_len) => {
                lo = lo.max(*min_len);
                if !found {
                    hi = *max_len;
                    found = true;
                } else {
                    hi = hi.min(*max_len);
                }
            }
            CheckSpec::ListNotEmpty => {
                lo = lo.max(1);
            }
            _ => {}
        }
    }
    (lo, hi)
}

fn struct_field_specs(specs: &[CheckSpec], field: &str) -> Vec<CheckSpec> {
    let mut out = Vec::new();
    for spec in specs {
        if let CheckSpec::StructField(name, inner) = spec {
            if name == field {
                out.extend(inner.clone());
            }
        }
    }
    out
}

/// String values satisfying membership, regex, and the string constraints,
/// or `None` when no membership constraint exists. Errors on empty
/// intersections so callers never sample values that fail their own schema.
fn allowed_strings(specs: &[CheckSpec]) -> PolarsResult<Option<Vec<String>>> {
    let Some(allowed) = one_of_values(specs) else {
        return Ok(None);
    };
    if allowed.is_empty() {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable one_of with no allowed values".into(),
        ));
    }
    let constraints = string_constraints(specs);
    let mut candidates: Vec<String> = allowed
        .into_iter()
        .filter(|value| satisfies_string_constraints(value, &constraints))
        .collect();
    let patterns = matches_patterns(specs);
    if !patterns.is_empty() {
        let regexes = patterns
            .iter()
            .map(|pattern| {
                Regex::new(pattern).map_err(|_| {
                    PolarsError::ComputeError(format!("Invalid regex: {pattern}").into())
                })
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        candidates.retain(|value| regexes.iter().all(|regex| regex.is_match(value)));
    }
    if candidates.is_empty() {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable one_of and string constraints in schema".into(),
        ));
    }
    Ok(Some(candidates))
}

fn gen_string_value(rng: &mut ChaCha8Rng, specs: &[CheckSpec], row: usize) -> PolarsResult<String> {
    if let Some(allowed) = allowed_strings(specs)? {
        return Ok(allowed[rng.random_range(0..allowed.len())].clone());
    }
    let constraints = string_constraints(specs);
    let patterns = matches_patterns(specs);
    if patterns.is_empty() {
        let base = format!("value-{row}");
        if let Some(value) = enforce_string_constraints(base, &constraints) {
            return Ok(value);
        }
        return match fitting_string(&constraints) {
            Some(value) => Ok(value),
            None => Err(PolarsError::ComputeError(
                "Unsatisfiable string constraints in schema".into(),
            )),
        };
    }
    let regexes = patterns
        .iter()
        .map(|pattern| {
            Regex::new(pattern)
                .map_err(|_| PolarsError::ComputeError(format!("Invalid regex: {pattern}").into()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    // Try each known canned value against every pattern; synthesizing
    // arbitrary intersections is out of scope, so report honestly when no
    // candidate satisfies the whole set.
    for pattern in &patterns {
        if let Some(canned) = canned_match(pattern) {
            if regexes.iter().all(|regex| regex.is_match(&canned))
                && satisfies_string_constraints(&canned, &constraints)
            {
                return Ok(canned);
            }
        }
    }
    Err(PolarsError::ComputeError(
        "Cannot synthesize a string satisfying every pattern in schema".into(),
    ))
}

fn cast_to_dtype(series: Series, dtype: &DataType) -> PolarsResult<Series> {
    if series.dtype() == dtype {
        Ok(series)
    } else {
        let mut cast = series.cast(dtype)?;
        cast.rename(series.name().clone());
        Ok(cast)
    }
}

fn gen_int_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    dtype: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    if let Some(allowed) = one_of_numeric_values(specs) {
        if *dtype == DataType::UInt128 {
            let candidates: Vec<u128> = allowed
                .iter()
                .filter_map(|&v| exact_u128_value(v))
                .collect();
            if candidates.is_empty() {
                return Err(PolarsError::ComputeError(
                    "Unsatisfiable one_of values for the UInt128 column in schema".into(),
                ));
            }
            let values: Vec<u128> = (0..n)
                .map(|_| candidates[rng.random_range(0..candidates.len())])
                .collect();
            return cast_to_dtype(Series::new(name.clone(), values), dtype);
        }
        // Sample in the integer domain so every value satisfies the bounds; membership
        // is exact.
        let (dmin, dmax, _) = int_domain_bounds(dtype)?;
        let candidates: Vec<i128> = allowed
            .iter()
            .filter_map(|&v| exact_i128_value(v))
            .filter(|v| *v >= dmin && *v <= dmax)
            .collect();
        if candidates.is_empty() {
            return Err(PolarsError::ComputeError(
                format!("Unsatisfiable one_of values for the {dtype} column in schema").into(),
            ));
        }
        let values: Vec<i128> = (0..n)
            .map(|_| candidates[rng.random_range(0..candidates.len())])
            .collect();
        return cast_to_dtype(Series::new(name.clone(), values), dtype);
    }
    // Sample in the integer domain so every value satisfies the bounds
    // exactly; float sampling plus truncation can land outside them.
    let min_width = int_min_width(specs, n);
    if *dtype == DataType::UInt128 {
        let fallback = (0u128, 1_000_000u128);
        let (lo, hi) = int_range_u128(specs, fallback, min_width)?;
        let values: Vec<u128> = (0..n).map(|_| rng.random_range(lo..=hi)).collect();
        return cast_to_dtype(Series::new(name.clone(), values), dtype);
    }
    let (dmin, dmax, fallback) = int_domain_bounds(dtype)?;
    let (lo, hi) = int_range_i128(specs, dmin, dmax, fallback, min_width)?;
    let values: Vec<i128> = (0..n).map(|_| rng.random_range(lo..=hi)).collect();
    cast_to_dtype(Series::new(name.clone(), values), dtype)
}

/// Domain bounds and default sampling window used by signed integer
/// generation.
fn int_domain_bounds(dtype: &DataType) -> PolarsResult<(i128, i128, (i128, i128))> {
    match dtype {
        DataType::Int8 => Ok((i128::from(i8::MIN), i128::from(i8::MAX), (-100, 100))),
        DataType::UInt8 => Ok((0, i128::from(u8::MAX), (0, 200))),
        DataType::UInt16 => Ok((0, i128::from(u16::MAX), (0, 1000))),
        DataType::UInt32 => Ok((0, i128::from(u32::MAX), (0, 100_000))),
        DataType::UInt64 => Ok((0, i128::from(u64::MAX), (0, 1_000_000))),
        DataType::Int128 => Ok((i128::MIN, i128::MAX, (0, 1000))),
        _ => Ok((i128::from(i64::MIN), i128::from(i64::MAX), (0, 1000))),
    }
}

/// Inclusive integer sampling range from check bounds.
/// Errors when no representable integer satisfies the bounds.
///
/// Windows already compatible with the old float-fallback behavior are kept
/// as-is; only a fallback side that contradicts a real bound is re-derived
/// from the bounded side (e.g. `gt(2000)` with a `0..1000` fallback).
fn int_range_i128(
    specs: &[CheckSpec],
    dmin: i128,
    dmax: i128,
    fallback: (i128, i128),
    min_unique: u128,
) -> PolarsResult<(i128, i128)> {
    let bounds = numeric_bounds(specs);
    if bounds.lo == f64::INFINITY || bounds.hi == f64::NEG_INFINITY {
        return Err(unsatisfiable_bounds());
    }
    // Smallest/largest integers satisfying each side (`as` casts saturate,
    // so out-of-range thresholds stay sound and never panic). Unbounded
    // sides start from the fallback window, preserving established
    // sampling distributions.
    let raw_lo: i128 = if bounds.lo.is_finite() {
        if bounds.lo_inclusive {
            bounds.lo.ceil() as i128
        } else {
            (bounds.lo.floor() as i128).saturating_add(1)
        }
    } else {
        fallback.0
    };
    let raw_hi: i128 = if bounds.hi.is_finite() {
        if bounds.hi_inclusive {
            bounds.hi.floor() as i128
        } else {
            (bounds.hi.ceil() as i128).saturating_sub(1)
        }
    } else {
        fallback.1
    };
    if raw_lo > dmax || raw_hi < dmin {
        return Err(unsatisfiable_bounds());
    }
    let mut lo = raw_lo.clamp(dmin, dmax);
    let mut hi = raw_hi.clamp(dmin, dmax);
    if lo <= hi {
        // Widen for unique() only on unbounded sides; a bounded window that
        // is too small is genuinely unsatisfiable for sampling.
        if min_unique > 0 && int_span(hi, lo) < min_unique {
            if !bounds.lo.is_finite() {
                let room = int_span(hi, dmin);
                lo = hi.saturating_sub(min_unique.min(room) as i128).max(dmin);
            } else if !bounds.hi.is_finite() {
                let room = int_span(dmax, lo);
                hi = lo.saturating_add(min_unique.min(room) as i128).min(dmax);
            } else {
                return Err(unsatisfiable_bounds());
            }
            if int_span(hi, lo) < min_unique {
                return Err(unsatisfiable_bounds());
            }
        }
        return Ok((lo, hi));
    }
    // Contradiction between a real bound and a fallback default: re-derive
    // the fallback side from the bounded side.
    let span = (fallback.1 - fallback.0).max(0) as u128;
    let span = span.max(min_unique).max(1);
    if bounds.lo.is_finite() && !bounds.hi.is_finite() {
        lo = raw_lo.clamp(dmin, dmax);
        hi = lo
            .saturating_add(span.min(int_span(dmax, lo)) as i128)
            .min(dmax);
    } else if !bounds.lo.is_finite() && bounds.hi.is_finite() {
        hi = raw_hi.clamp(dmin, dmax);
        lo = hi
            .saturating_sub(span.min(int_span(hi, dmin)) as i128)
            .max(dmin);
    } else {
        return Err(unsatisfiable_bounds());
    }
    if lo > hi {
        return Err(unsatisfiable_bounds());
    }
    Ok((lo, hi))
}

/// Inclusive `u128` sampling range; mirrors `int_range_i128`.
fn int_range_u128(
    specs: &[CheckSpec],
    fallback: (u128, u128),
    min_unique: u128,
) -> PolarsResult<(u128, u128)> {
    let bounds = numeric_bounds(specs);
    if bounds.lo == f64::INFINITY || bounds.hi == f64::NEG_INFINITY {
        return Err(unsatisfiable_bounds());
    }
    // No unsigned value can satisfy an upper bound below zero (-0.0 still
    // admits 0, since -0.0 == 0.0).
    if bounds.hi.is_finite() && bounds.hi.floor() < 0.0 {
        return Err(unsatisfiable_bounds());
    }
    let raw_lo: u128 = if bounds.lo.is_finite() {
        if bounds.lo_inclusive {
            bounds.lo.ceil().max(0.0) as u128
        } else {
            (bounds.lo.floor().max(0.0) as u128).saturating_add(1)
        }
    } else {
        fallback.0
    };
    let raw_hi: u128 = if bounds.hi.is_finite() {
        if bounds.hi_inclusive {
            bounds.hi.floor().min(u128::MAX as f64) as u128
        } else {
            (bounds.hi.ceil().min(u128::MAX as f64) as u128).saturating_sub(1)
        }
    } else {
        fallback.1
    };
    if raw_lo > raw_hi {
        // Contradiction between a real bound and a fallback default:
        // re-derive the fallback side from the bounded side.
        let span = (fallback.1 - fallback.0).max(min_unique).max(1);
        if bounds.lo.is_finite() && !bounds.hi.is_finite() {
            let hi = raw_lo.saturating_add(span.min(u128::MAX - raw_lo));
            return if raw_lo <= hi {
                Ok((raw_lo, hi))
            } else {
                Err(unsatisfiable_bounds())
            };
        } else if !bounds.lo.is_finite() && bounds.hi.is_finite() {
            let lo = raw_hi.saturating_sub(span.min(raw_hi));
            return if lo <= raw_hi {
                Ok((lo, raw_hi))
            } else {
                Err(unsatisfiable_bounds())
            };
        }
        return Err(unsatisfiable_bounds());
    }
    let (mut lo, mut hi) = (raw_lo, raw_hi);
    if min_unique > 0 && (hi - lo) < min_unique {
        if !bounds.lo.is_finite() {
            lo = hi.saturating_sub(min_unique.min(hi));
        } else if !bounds.hi.is_finite() {
            hi = lo.saturating_add(min_unique.min(u128::MAX - lo));
        } else {
            return Err(unsatisfiable_bounds());
        }
        if (hi - lo) < min_unique {
            return Err(unsatisfiable_bounds());
        }
    }
    Ok((lo, hi))
}

fn unsatisfiable_bounds() -> PolarsError {
    PolarsError::ComputeError("Unsatisfiable numeric bounds in schema".into())
}

/// Width of `lo..=hi` as `u128`. Callers must ensure `lo <= hi`; the value
/// is always exact because the widest `i128` span (`2^128 - 1`) fits in
/// `u128`, unlike the plain subtraction which overflows and panics.
fn int_span(hi: i128, lo: i128) -> u128 {
    debug_assert!(lo <= hi);
    (hi as u128).wrapping_sub(lo as u128)
}

/// Distinct values required beyond the first, so `unique()` columns can find
/// room for `n` distinct values (`0` when uniqueness is not needed).
fn int_min_width(specs: &[CheckSpec], n: usize) -> u128 {
    if has_unique(specs) {
        n.saturating_sub(1) as u128
    } else {
        0
    }
}

fn gen_float_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    dtype: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    if let Some(allowed) = one_of_numeric_values(specs) {
        if allowed.is_empty() {
            return Err(PolarsError::ComputeError(
                "Unsatisfiable one_of values for the column in schema".into(),
            ));
        }
        let values: Vec<f64> = (0..n)
            .map(|_| allowed[rng.random_range(0..allowed.len())])
            .collect();
        let float_series = Series::new(name.clone(), values);
        return cast_to_dtype(float_series, dtype);
    }
    let bounds = numeric_bounds(specs);
    let min_width = if has_unique(specs) { n as f64 } else { 0.0 };
    let values: Vec<f64> = (0..n)
        .map(|_| sample_f64_in(rng, &bounds, 0.0, 1.0, min_width))
        .collect::<PolarsResult<_>>()?;
    let float_series = Series::new(name.clone(), values);
    cast_to_dtype(float_series, dtype)
}

fn gen_temporal_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    dtype: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    if one_of_numeric_values(specs).is_some() {
        return Err(PolarsError::ComputeError(
            "Data generation does not support one_of on temporal columns".into(),
        ));
    }
    let (lo, hi) = int_range_i128(
        specs,
        i128::from(i64::MIN),
        i128::from(i64::MAX),
        (1, 10_000),
        int_min_width(specs, n),
    )?;
    let values: Vec<i64> = (0..n).map(|_| rng.random_range(lo..=hi) as i64).collect();
    let int_series = Series::new(name.clone(), values);
    cast_to_dtype(int_series, dtype)
}

fn gen_scalar_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    dtype: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    match dtype {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::Int128
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::UInt128 => gen_int_series(rng, name, dtype, specs, n),
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            gen_float_series(rng, name, dtype, specs, n)
        }
        DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            gen_temporal_series(rng, name, dtype, specs, n)
        }
        DataType::Boolean => {
            if let Some(allowed) = one_of_numeric_values(specs) {
                let has_true = allowed.contains(&1.0);
                let has_false = allowed.contains(&0.0);
                if !has_true && !has_false {
                    return Err(PolarsError::ComputeError(
                        "Unsatisfiable one_of values for the Boolean column in schema".into(),
                    ));
                }
                return match (has_true, has_false) {
                    (true, false) => Ok(Series::new(name.clone(), vec![true; n])),
                    (false, true) => Ok(Series::new(name.clone(), vec![false; n])),
                    _ => Ok(Series::new(
                        name.clone(),
                        (0..n).map(|_| rng.random_bool(0.5)).collect::<Vec<bool>>(),
                    )),
                };
            }
            let values: Vec<bool> = (0..n).map(|_| rng.random_bool(0.5)).collect();
            Ok(Series::new(name.clone(), values))
        }
        DataType::String => {
            let values: Vec<String> = (0..n)
                .map(|row| gen_string_value(rng, specs, row))
                .collect::<PolarsResult<_>>()?;
            Ok(Series::new(name.clone(), values))
        }
        DataType::Null => Ok(Series::new_null(name.clone(), n)),
        other => Err(PolarsError::ComputeError(
            format!("Data generation does not support dtype {other}").into(),
        )),
    }
}

/// Materialization ceiling for sampled list lengths. Open-ended schemas
/// such as `ListLenBetween(1, usize::MAX)` would otherwise draw astronomic
/// lengths and exhaust memory; every sampled length still satisfies the
/// declared bounds.
const MAX_SAMPLE_LIST_LEN: usize = 10_000;

fn gen_list_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    inner: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    let (min_len, max_len) = list_len_bounds(specs);
    if min_len > max_len {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable list length bounds in schema".into(),
        ));
    }
    if min_len > MAX_SAMPLE_LIST_LEN {
        return Err(PolarsError::ComputeError(
            format!("List length {min_len} exceeds generation capacity ({MAX_SAMPLE_LIST_LEN})")
                .into(),
        ));
    }
    let max_len = max_len.min(MAX_SAMPLE_LIST_LEN);
    let mut inners = Vec::with_capacity(n);
    for _ in 0..n {
        let len = if min_len == max_len {
            min_len
        } else {
            rng.random_range(min_len..=max_len)
        };
        // Element checks (beyond length) are not part of a list spec, so
        // inner cells are generated without checks; nested dtypes recurse
        // through `gen_base_series`, which handles any depth.
        let values = gen_base_series(rng, &PlSmallStr::from(""), inner, &[], len)?;
        let values = cast_to_dtype(values, inner)?;
        inners.push(values);
    }
    Ok(Series::new(name.clone(), inners))
}

fn gen_struct_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    fields: &[Field],
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    let mut columns = Vec::with_capacity(fields.len());
    for field in fields {
        let field_specs = struct_field_specs(specs, field.name.as_str());
        columns.push(gen_base_series(
            rng,
            &field.name,
            &field.dtype,
            &field_specs,
            n,
        )?);
    }
    let frame = DataFrame::new(n, columns.into_iter().map(Column::from).collect())?;
    Ok(frame.into_struct(name.clone()).into_series())
}

fn gen_base_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    dtype: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    match dtype {
        DataType::List(inner) => gen_list_series(rng, name, inner, specs, n),
        DataType::Array(inner, width) => {
            let mut list_specs = specs.to_vec();
            list_specs.push(CheckSpec::ListLenBetween(*width, *width));
            gen_list_series(rng, name, inner, &list_specs, n)
                .and_then(|series| cast_to_dtype(series, dtype))
        }
        DataType::Struct(fields) => gen_struct_series(rng, name, fields, specs, n),
        _ => {
            let series = gen_scalar_series(rng, name, dtype, specs, n)?;
            if has_unique(specs) && n > 1 {
                ensure_unique(series, rng, dtype, specs)
            } else {
                Ok(series)
            }
        }
    }
}

/// Whether a dtype has scalar values that can be harvested one by one for
/// uniqueness repair. Nested dtypes keep the legacy whole-column resampling.
fn is_scalar_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Int128
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::UInt128
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
            | DataType::String
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time
            | DataType::Null
    )
}

fn unique_impossible(check: &str) -> PolarsError {
    PolarsError::ComputeError(
        format!("Could not generate unique values for column ({check})").into(),
    )
}

/// Uniqueness repair that preserves column checks.
///
/// Exact small domains resolve immediately: `one_of` pools shuffle-take,
/// and provably impossible demands (`bool` with `n > 2`, pinned single
/// floats, pattern-locked strings) error without sampling. Everything else
/// harvests distinct values in batches with bounded total work, so narrow
/// domains converge instead of relying on whole-column luck.
fn ensure_unique(
    series: Series,
    rng: &mut ChaCha8Rng,
    dtype: &DataType,
    specs: &[CheckSpec],
) -> PolarsResult<Series> {
    let n = series.len();
    let name = series.name().clone();
    if n <= 1 || is_unique(&series) {
        return Ok(series);
    }
    if !is_scalar_dtype(dtype) {
        // Nested values have no per-value sampler: bounded whole-column
        // resampling, unchanged from before.
        let mut current = series;
        for _ in 0..200 {
            if is_unique(&current) {
                return Ok(current);
            }
            current = gen_scalar_series(rng, &name, dtype, specs, n)?;
        }
        return Err(unique_impossible("no unique combination found"));
    }
    if matches!(dtype, DataType::Boolean) && n > 2 {
        return Err(unique_impossible("boolean domain holds 2 values"));
    }
    if is_pinned_float(dtype, specs) && n > 1 {
        return Err(unique_impossible("bounds pin a single value"));
    }
    if matches!(dtype, DataType::String) {
        if let Some(mut pool) = allowed_strings(specs)? {
            // Exact membership domain: shuffle-take, or fail fast.
            if pool.len() < n {
                return Err(unique_impossible("membership domain too small"));
            }
            pool.shuffle(rng);
            pool.truncate(n);
            let out = Series::new(name.clone(), pool);
            return cast_to_dtype(out, dtype);
        }
        if !matches_patterns(specs).is_empty() {
            // Pattern-locked strings render identically on every plain draw,
            // but most regexes admit many values: expand by mutation.
            match distinct_pattern_strings(rng, specs, n)? {
                Some(pool) => {
                    let out = Series::new(name.clone(), pool);
                    return cast_to_dtype(out, dtype);
                }
                None => return Err(unique_impossible("pattern-locked strings cannot vary")),
            }
        }
    }
    // Harvest distinct values in batches; every batch respects the checks,
    // so the assembled column validates by construction.
    let mut seen: HashSet<String> = HashSet::new();
    let mut distinct: Vec<AnyValue<'static>> = Vec::with_capacity(n);
    absorb_distinct(&series, &mut seen, &mut distinct);
    for _ in 0..200 {
        if distinct.len() >= n {
            break;
        }
        let batch = (n - distinct.len()).max(8);
        let fresh = gen_scalar_series(rng, &name, dtype, specs, batch)?;
        absorb_distinct(&fresh, &mut seen, &mut distinct);
    }
    if distinct.len() < n {
        return Err(unique_impossible("no unique combination found"));
    }
    distinct.shuffle(rng);
    distinct.truncate(n);
    let out = Series::from_any_values(name.clone(), &distinct, false)?;
    cast_to_dtype(out, dtype)
}

fn is_unique(series: &Series) -> bool {
    series.n_unique().unwrap_or(0) == series.len()
}

/// Expands pattern-locked strings into `n` distinct values satisfying every
/// regex, by mutating known-good canned seeds. Returns `None` only when no
/// seed exists (generation itself would already have failed); errors when
/// the patterns admit fewer than `n` distinct values.
fn distinct_pattern_strings(
    rng: &mut ChaCha8Rng,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Option<Vec<String>>> {
    let patterns = matches_patterns(specs);
    if patterns.is_empty() {
        return Ok(None);
    }
    let constraints = string_constraints(specs);
    let regexes = patterns
        .iter()
        .map(|pattern| {
            Regex::new(pattern)
                .map_err(|_| PolarsError::ComputeError(format!("Invalid regex: {pattern}").into()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;
    let mut seen: HashSet<String> = HashSet::new();
    let mut pool: Vec<String> = Vec::new();
    for pattern in &patterns {
        if let Some(canned) = canned_match(pattern) {
            if regexes.iter().all(|regex| regex.is_match(&canned))
                && satisfies_string_constraints(&canned, &constraints)
                && seen.insert(canned.clone())
            {
                pool.push(canned);
            }
        }
    }
    if pool.is_empty() {
        return Err(PolarsError::ComputeError(
            "Cannot synthesize a string satisfying every pattern in schema".into(),
        ));
    }
    let mut attempts = 0usize;
    while pool.len() < n && attempts < 20_000 {
        attempts += 1;
        let base = pool[rng.random_range(0..pool.len())].clone();
        let mutated = mutate_ascii_alnum(rng, &base);
        if mutated != base
            && regexes.iter().all(|regex| regex.is_match(&mutated))
            && satisfies_string_constraints(&mutated, &constraints)
            && seen.insert(mutated.clone())
        {
            pool.push(mutated);
        }
    }
    if pool.len() < n {
        return Err(unique_impossible("pattern admits too few distinct values"));
    }
    pool.shuffle(rng);
    pool.truncate(n);
    Ok(Some(pool))
}

/// Substitutes one or two ASCII-alphanumeric characters; anything else
/// (including non-ASCII bytes) is left untouched so output stays valid UTF-8.
fn mutate_ascii_alnum(rng: &mut ChaCha8Rng, base: &str) -> String {
    const ALNUM: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut bytes = base.as_bytes().to_vec();
    let positions: Vec<usize> = bytes
        .iter()
        .enumerate()
        .filter(|(_, byte)| byte.is_ascii_alphanumeric())
        .map(|(index, _)| index)
        .collect();
    if positions.is_empty() {
        return base.to_string();
    }
    for _ in 0..=rng.random_range(0..positions.len().min(2)) {
        let position = positions[rng.random_range(0..positions.len())];
        bytes[position] = ALNUM[rng.random_range(0..ALNUM.len())];
    }
    String::from_utf8(bytes).unwrap_or_else(|_| base.to_string())
}

/// A float dtype is pinned when the bounds collapse to a single value.
fn is_pinned_float(dtype: &DataType, specs: &[CheckSpec]) -> bool {
    if !matches!(
        dtype,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    ) {
        return false;
    }
    let bounds = numeric_bounds(specs);
    bounds.lo == bounds.hi && bounds.lo_inclusive && bounds.hi_inclusive
}

fn absorb_distinct(
    series: &Series,
    seen: &mut HashSet<String>,
    distinct: &mut Vec<AnyValue<'static>>,
) {
    for index in 0..series.len() {
        let value = series.get(index).unwrap().into_static();
        if matches!(value, AnyValue::Null) {
            continue;
        }
        if seen.insert(format!("{value:?}")) {
            distinct.push(value);
        }
    }
}

fn apply_nullable(series: Series, rng: &mut ChaCha8Rng) -> PolarsResult<Series> {
    if series.is_empty() {
        return Ok(series);
    }
    if series.n_unique().unwrap_or(1) == series.len() && series.len() > 1 {
        return Ok(series);
    }
    let mask: Vec<bool> = (0..series.len()).map(|_| rng.random_bool(0.05)).collect();
    if !mask.iter().any(|m| *m) {
        return Ok(series);
    }
    let values: Vec<AnyValue> = (0..series.len())
        .map(|index| {
            if mask[index] {
                AnyValue::Null
            } else {
                series.get(index).unwrap()
            }
        })
        .collect();
    Series::from_any_values(series.name().clone(), &values, false)
}

fn spec_lookup(schema: &Schema) -> HashMap<&str, &ColumnSpec> {
    schema
        .columns
        .iter()
        .map(|spec| (spec.name.as_str(), spec))
        .collect()
}

fn resample_one(
    rng: &mut ChaCha8Rng,
    lookup: &HashMap<&str, &ColumnSpec>,
    name: &str,
) -> PolarsResult<Series> {
    let spec = lookup
        .get(name)
        .copied()
        .ok_or_else(|| PolarsError::ColumnNotFound(format!("Missing column '{name}'").into()))?;
    gen_base_series(
        rng,
        &PlSmallStr::from(name),
        &spec.dtype,
        &specs_of(spec),
        1,
    )
}

fn specs_of(spec: &ColumnSpec) -> Vec<CheckSpec> {
    spec.checks.iter().map(|check| check.spec()).collect()
}

fn missing_sample_column(name: &str) -> PolarsError {
    PolarsError::ColumnNotFound(format!("Table check references missing column '{name}'").into())
}

/// One extracted cell value for cross-column comparison. Numeric values keep
/// their native precision (no `f64` rounding, no string formatting), so
/// temporal and wide-integer comparisons behave like validation.
#[derive(Clone)]
enum Cell {
    Int(Option<i128>),
    UInt(Option<u128>),
    Float(Option<f64>),
    Text(Option<String>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CellKind {
    Int,
    UInt,
    Float,
    Text,
}

fn cell_kind(dtype: &DataType) -> Option<CellKind> {
    match dtype {
        DataType::String => Some(CellKind::Text),
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Some(CellKind::Float),
        DataType::UInt128 => Some(CellKind::UInt),
        dtype if is_supported_numeric_dtype(dtype) => Some(CellKind::Int),
        _ => None,
    }
}

fn unsupported_comparison_column(name: &str, dtype: &DataType) -> PolarsError {
    PolarsError::ComputeError(format!("Cannot compare column '{name}' with dtype {dtype}").into())
}

fn extract_cells(series: &Series, kind: CellKind) -> PolarsResult<Vec<Cell>> {
    match kind {
        CellKind::Int => {
            let base = match series.dtype() {
                DataType::Date
                | DataType::Datetime(_, _)
                | DataType::Duration(_)
                | DataType::Time => series.cast(&DataType::Int64)?,
                _ => series.clone(),
            };
            let wide = base.cast(&DataType::Int128)?;
            let values = wide.i128()?;
            Ok(values.iter().map(Cell::Int).collect())
        }
        CellKind::UInt => {
            let wide = series.cast(&DataType::UInt128)?;
            let values = wide.u128()?;
            Ok(values.iter().map(Cell::UInt).collect())
        }
        CellKind::Float => {
            let wide = series.cast(&DataType::Float64)?;
            let values = wide.f64()?;
            Ok(values.iter().map(Cell::Float).collect())
        }
        CellKind::Text => {
            let string = series.str().map_err(|_| {
                PolarsError::ComputeError("Cannot compare non-string column".into())
            })?;
            Ok(string
                .iter()
                .map(|value| Cell::Text(value.map(str::to_string)))
                .collect())
        }
    }
}

fn cells_hold(op: CompareOp, left: &Cell, right: &Cell) -> bool {
    match (left, right) {
        (Cell::Int(a), Cell::Int(b)) => opts_hold(op, a, b),
        (Cell::UInt(a), Cell::UInt(b)) => opts_hold(op, a, b),
        (Cell::Float(a), Cell::Float(b)) => opts_hold(op, a, b),
        (Cell::Text(a), Cell::Text(b)) => opts_hold(op, a, b),
        _ => false,
    }
}

/// Nulls skip the comparison, mirroring validation semantics.
fn opts_hold<T: PartialEq + PartialOrd>(
    op: CompareOp,
    left: &Option<T>,
    right: &Option<T>,
) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => compare_values(op, left, right),
        _ => true,
    }
}

fn compare_values<T: PartialEq + PartialOrd>(op: CompareOp, left: &T, right: &T) -> bool {
    match op {
        CompareOp::Lt => left < right,
        CompareOp::Le => left <= right,
        CompareOp::Gt => left > right,
        CompareOp::Ge => left >= right,
        CompareOp::Eq => left == right,
        CompareOp::Ne => left != right,
    }
}

/// Replaces one row of a frame column with a single-row candidate series.
fn replace_row(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    name: &str,
    row: usize,
    candidate: &Series,
) -> PolarsResult<()> {
    let current = frame.get(name).ok_or_else(|| missing_sample_column(name))?;
    let n = current.len();
    let values: Vec<AnyValue> = (0..n)
        .map(|index| {
            if index == row {
                candidate.get(0).unwrap()
            } else {
                current.get(index).unwrap()
            }
        })
        .collect();
    let rebuilt = Series::from_any_values(PlSmallStr::from(name), &values, false)?;
    let spec = lookup
        .get(name)
        .copied()
        .ok_or_else(|| missing_sample_column(name))?;
    frame.insert(name.to_string(), cast_to_dtype(rebuilt, &spec.dtype)?);
    Ok(())
}

fn enforce_comparison(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    left: &str,
    op: CompareOp,
    right: &str,
) -> PolarsResult<()> {
    let left_spec = lookup
        .get(left)
        .copied()
        .ok_or_else(|| missing_sample_column(left))?;
    let right_spec = lookup
        .get(right)
        .copied()
        .ok_or_else(|| missing_sample_column(right))?;
    let left_kind = cell_kind(&left_spec.dtype)
        .ok_or_else(|| unsupported_comparison_column(left, &left_spec.dtype))?;
    let right_kind = cell_kind(&right_spec.dtype)
        .ok_or_else(|| unsupported_comparison_column(right, &right_spec.dtype))?;
    // Mixed int/float pairs compare as floats, exactly like validation.
    // Mixed temporal dtypes are rejected like validation: raw ticks from
    // different units are not comparable.
    if is_temporal_dtype(&left_spec.dtype)
        && is_temporal_dtype(&right_spec.dtype)
        && left_spec.dtype != right_spec.dtype
    {
        return Err(PolarsError::ComputeError(
            format!("Cannot satisfy comparison '{left} {op:?} {right}': temporal columns must share a dtype").into(),
        ));
    }
    let kind = match (left_kind, right_kind) {
        (CellKind::Text, CellKind::Text) => CellKind::Text,
        (CellKind::UInt, CellKind::UInt) => CellKind::UInt,
        (CellKind::Int, CellKind::Int) => CellKind::Int,
        (CellKind::Int, CellKind::UInt) | (CellKind::UInt, CellKind::Int) => {
            return enforce_mixed_int(
                frame,
                lookup,
                rng,
                left,
                right,
                op,
                left_kind == CellKind::UInt,
            );
        }
        (left, right) if left != CellKind::Text && right != CellKind::Text => CellKind::Float,
        _ => {
            return Err(PolarsError::ComputeError(
                format!("Cannot satisfy comparison '{left} {op:?} {right}': mismatched kinds")
                    .into(),
            ));
        }
    };

    let n = frame
        .get(left)
        .ok_or_else(|| missing_sample_column(left))?
        .len();
    let mut left_cells = extract_cells(
        frame.get(left).ok_or_else(|| missing_sample_column(left))?,
        kind,
    )?;
    let mut right_cells = extract_cells(
        frame
            .get(right)
            .ok_or_else(|| missing_sample_column(right))?,
        kind,
    )?;
    for row in 0..n {
        if cells_hold(op, &left_cells[row], &right_cells[row]) {
            continue;
        }
        let mut satisfied = false;
        // Joint rejection sampling: redraw both sides from their own
        // distributions so pairs like `start < end` succeed even when the
        // left value sits at the top of its range.
        for _ in 0..200 {
            let left_candidate = resample_one(rng, lookup, left)?;
            let right_candidate = resample_one(rng, lookup, right)?;
            let left_cell = extract_cells(&left_candidate, kind)?
                .into_iter()
                .next()
                .unwrap();
            let right_cell = extract_cells(&right_candidate, kind)?
                .into_iter()
                .next()
                .unwrap();
            if cells_hold(op, &left_cell, &right_cell) {
                replace_row(frame, lookup, left, row, &left_candidate)?;
                replace_row(frame, lookup, right, row, &right_candidate)?;
                left_cells[row] = left_cell;
                right_cells[row] = right_cell;
                satisfied = true;
                break;
            }
        }
        if !satisfied {
            return Err(PolarsError::ComputeError(
                format!("Could not satisfy comparison '{left} {op:?} {right}'").into(),
            ));
        }
    }
    Ok(())
}

/// Enforces a comparison between a signed column and an unsigned one using
/// the exact mixed ordering (mirrors validation). `left_uint` records which
/// side holds the unsigned values so candidates map back correctly.
#[allow(clippy::too_many_arguments)]
fn enforce_mixed_int(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    left: &str,
    right: &str,
    op: CompareOp,
    left_uint: bool,
) -> PolarsResult<()> {
    use std::cmp::Ordering;
    let n = frame
        .get(left)
        .ok_or_else(|| missing_sample_column(left))?
        .len();
    // Each vector holds one side's values; `holds` maps them back to
    // left-vs-right order via `left_uint`.
    let (mut signed_vals, mut unsigned_vals) = if left_uint {
        (
            int128_cells(
                frame
                    .get(right)
                    .ok_or_else(|| missing_sample_column(right))?,
            )?,
            uint128_cells(frame.get(left).ok_or_else(|| missing_sample_column(left))?)?,
        )
    } else {
        (
            int128_cells(frame.get(left).ok_or_else(|| missing_sample_column(left))?)?,
            uint128_cells(
                frame
                    .get(right)
                    .ok_or_else(|| missing_sample_column(right))?,
            )?,
        )
    };
    let holds = |signed: &Option<i128>, unsigned: &Option<u128>| match (signed, unsigned) {
        (Some(a), Some(b)) => {
            let order = cmp_mixed_int_uint(*a, *b);
            let order = if left_uint { order.reverse() } else { order };
            match op {
                CompareOp::Lt => order == Ordering::Less,
                CompareOp::Le => order != Ordering::Greater,
                CompareOp::Gt => order == Ordering::Greater,
                CompareOp::Ge => order != Ordering::Less,
                CompareOp::Eq => order == Ordering::Equal,
                CompareOp::Ne => order != Ordering::Equal,
            }
        }
        _ => true,
    };
    for row in 0..n {
        if holds(&signed_vals[row], &unsigned_vals[row]) {
            continue;
        }
        let mut satisfied = false;
        for _ in 0..200 {
            let left_candidate = resample_one(rng, lookup, left)?;
            let right_candidate = resample_one(rng, lookup, right)?;
            let (signed_candidate, unsigned_candidate) = if left_uint {
                (
                    int128_cells(&right_candidate)?.into_iter().next().unwrap(),
                    uint128_cells(&left_candidate)?.into_iter().next().unwrap(),
                )
            } else {
                (
                    int128_cells(&left_candidate)?.into_iter().next().unwrap(),
                    uint128_cells(&right_candidate)?.into_iter().next().unwrap(),
                )
            };
            if holds(&signed_candidate, &unsigned_candidate) {
                replace_row(frame, lookup, left, row, &left_candidate)?;
                replace_row(frame, lookup, right, row, &right_candidate)?;
                signed_vals[row] = signed_candidate;
                unsigned_vals[row] = unsigned_candidate;
                satisfied = true;
                break;
            }
        }
        if !satisfied {
            return Err(PolarsError::ComputeError(
                format!("Could not satisfy comparison '{left} {op:?} {right}'").into(),
            ));
        }
    }
    Ok(())
}

fn enforce_unique_together(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    columns: &[String],
) -> PolarsResult<()> {
    let first = columns.first().ok_or_else(|| {
        PolarsError::ComputeError("UniqueTogether needs at least one column".into())
    })?;
    for name in columns {
        if !lookup.contains_key(name.as_str()) {
            return Err(missing_sample_column(name));
        }
    }
    let n = frame
        .get(first)
        .ok_or_else(|| missing_sample_column(first))?
        .len();
    let last = columns.last().unwrap().clone();
    for _ in 0..(n * 10 + 10) {
        let selected: Vec<Series> = columns
            .iter()
            .map(|name| {
                frame
                    .get(name)
                    .cloned()
                    .ok_or_else(|| missing_sample_column(name))
            })
            .collect::<PolarsResult<_>>()?;
        let check_frame = DataFrame::new(n, selected.into_iter().map(Column::from).collect())?;
        let duplicated = check_frame.is_duplicated()?;
        if !duplicated.any() {
            return Ok(());
        }
        let rows: Vec<usize> = duplicated
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                if let Some(true) = value {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();
        for row in rows {
            let candidate = resample_one(rng, lookup, &last)?;
            replace_row(frame, lookup, &last, row, &candidate)?;
        }
    }
    Err(PolarsError::ComputeError(
        "Could not generate unique combinations".into(),
    ))
}

/// Enforces a per-row chain order by reusing the pair enforcement: the
/// chain is exactly `c1 (<=|<) c2 (<=|<) c3 ...`. Pairs are repaired
/// jointly, and the fixed-point loop re-runs passes until the frame
/// validates as a whole.
fn enforce_ordered(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    columns: &[String],
    strict: bool,
) -> PolarsResult<()> {
    let op = if strict { CompareOp::Lt } else { CompareOp::Le };
    if columns.len() < 2 {
        return Err(PolarsError::ComputeError(
            "Ordered needs at least two columns".into(),
        ));
    }
    for pair in columns.windows(2) {
        enforce_comparison(frame, lookup, rng, &pair[0], op, &pair[1])?;
    }
    Ok(())
}

/// A summed cell in the widest applicable domain. Mirrors the validation
/// classification: exact `i128`/`u128` when every column shares one
/// integer domain, `f64` when any float or mixed-domain column joins.
#[derive(Clone, Copy)]
enum SumCell {
    Int(Option<i128>),
    UInt(Option<u128>),
    Float(Option<f64>),
}

impl SumCell {
    fn is_null(&self) -> bool {
        matches!(self, Self::Int(None) | Self::UInt(None) | Self::Float(None))
    }
}

fn extract_sum_cells(series: &Series) -> PolarsResult<Vec<SumCell>> {
    if int_domain(series.dtype()) == Some(IntDomain::I128) {
        Ok(int128_cells(series)?
            .into_iter()
            .map(SumCell::Int)
            .collect())
    } else if int_domain(series.dtype()) == Some(IntDomain::U128) {
        Ok(uint128_cells(series)?
            .into_iter()
            .map(SumCell::UInt)
            .collect())
    } else {
        let values = series.cast(&DataType::Float64)?;
        let floats = values
            .f64()
            .map_err(|_| PolarsError::ComputeError("Cannot sum non-numeric column".into()))?;
        Ok(floats.iter().map(SumCell::Float).collect())
    }
}

/// One row's computed sum, or the widest fallback domain when float or
/// mixed-domain columns participate (mirroring `sum_equals_target`).
enum SumValue {
    Signed(i128),
    Unsigned(u128),
    Float(f64),
}

/// Computed sum of one row's operands. All cells are present (nulls are
/// skipped earlier); `None` means the exact arithmetic overflowed.
fn row_sum(cells: &[SumCell]) -> Option<SumValue> {
    let all_unsigned = !cells.is_empty()
        && cells
            .iter()
            .all(|cell| matches!(cell, SumCell::UInt(Some(_))));
    if all_unsigned {
        let mut total: u128 = 0;
        for cell in cells {
            let SumCell::UInt(Some(value)) = cell else {
                unreachable!("all_unsigned checked above");
            };
            total = total.checked_add(*value)?;
        }
        return Some(SumValue::Unsigned(total));
    }
    let any_float = cells
        .iter()
        .any(|cell| matches!(cell, SumCell::Float(Some(_))));
    let mixed_int = cells
        .iter()
        .any(|cell| matches!(cell, SumCell::UInt(Some(_))))
        && cells
            .iter()
            .any(|cell| matches!(cell, SumCell::Int(Some(_))));
    if any_float || mixed_int {
        let mut total = 0.0f64;
        for cell in cells {
            let value = match cell {
                SumCell::Float(Some(v)) => *v,
                SumCell::Int(Some(v)) => *v as f64,
                SumCell::UInt(Some(v)) => *v as f64,
                _ => return None,
            };
            total += value;
        }
        return Some(SumValue::Float(total));
    }
    let mut total: i128 = 0;
    for cell in cells {
        let SumCell::Int(Some(value)) = cell else {
            unreachable!("all remaining cells are signed");
        };
        total = total.checked_add(*value)?;
    }
    Some(SumValue::Signed(total))
}

fn sum_matches_target(sum: &SumValue, target: &SumCell, tolerance: f64) -> bool {
    match (sum, target) {
        (SumValue::Signed(sum), SumCell::Int(Some(target))) => sum == target,
        (SumValue::Unsigned(sum), SumCell::UInt(Some(target))) => sum == target,
        (SumValue::Float(sum), SumCell::Float(Some(target))) => (sum - target).abs() <= tolerance,
        (SumValue::Signed(sum), SumCell::UInt(Some(target))) => {
            *target <= i128::MAX as u128 && *sum == *target as i128
        }
        (SumValue::Unsigned(sum), SumCell::Int(Some(target))) => {
            *sum <= i128::MAX as u128 && *sum as i128 == *target
        }
        (sum, target) => {
            let Some(target_f64) = (match target {
                SumCell::Int(Some(v)) => Some(*v as f64),
                SumCell::UInt(Some(v)) => Some(*v as f64),
                SumCell::Float(Some(v)) => Some(*v),
                _ => None,
            }) else {
                return false;
            };
            let sum_f64 = match sum {
                SumValue::Signed(v) => *v as f64,
                SumValue::Unsigned(v) => *v as f64,
                SumValue::Float(v) => *v,
            };
            (sum_f64 - target_f64).abs() <= tolerance
        }
    }
}

/// The target value that makes the row's sum hold: the operand sum
/// itself, coerced into the target's own domain. `None` marks unsynthesizable repairs (overflow, negative
/// unsigned results); callers fall back to rejection sampling.
#[derive(Clone, Copy, Debug)]
enum CandidateValue {
    Signed(i128),
    Unsigned(u128),
    Float(f64),
}

impl CandidateValue {
    fn as_f64(&self) -> f64 {
        match self {
            Self::Signed(v) => *v as f64,
            Self::Unsigned(v) => *v as f64,
            Self::Float(v) => *v,
        }
    }
}

fn desired_target_value(sum: &SumValue, target: &SumCell) -> Option<CandidateValue> {
    match (sum, target) {
        (SumValue::Signed(sum), SumCell::Int(_)) => Some(CandidateValue::Signed(*sum)),
        (SumValue::Unsigned(sum), SumCell::Int(_)) if *sum <= i128::MAX as u128 => {
            Some(CandidateValue::Signed(*sum as i128))
        }
        (SumValue::Unsigned(sum), SumCell::UInt(_)) => Some(CandidateValue::Unsigned(*sum)),
        (SumValue::Signed(sum), SumCell::UInt(_)) if *sum >= 0 => {
            Some(CandidateValue::Unsigned(*sum as u128))
        }
        (SumValue::Float(sum), SumCell::Float(_)) => Some(CandidateValue::Float(*sum)),
        // A float sum against an integer target rarely repairs exactly; offer
        // the float and let the dtype cast (and rejection fallback) decide.
        (SumValue::Float(sum), _) => Some(CandidateValue::Float(*sum)),
        (SumValue::Signed(sum), _) => Some(CandidateValue::Float(*sum as f64)),
        (SumValue::Unsigned(sum), _) => Some(CandidateValue::Float(*sum as f64)),
    }
}

/// Whether a repaired target value also satisfies the column's own
/// numeric bounds and numeric membership pool.
fn target_accommodates(
    bounds: &NumericBounds,
    one_of_pool: Option<&Vec<f64>>,
    value: &CandidateValue,
) -> bool {
    let candidate = value.as_f64();
    let lo_ok = if bounds.lo_inclusive {
        candidate >= bounds.lo
    } else {
        candidate > bounds.lo
    };
    let hi_ok = if bounds.hi_inclusive {
        candidate <= bounds.hi
    } else {
        candidate < bounds.hi
    };
    if !(lo_ok && hi_ok) {
        return false;
    }
    one_of_pool.is_none_or(|pool| pool.contains(&candidate))
}

fn candidate_series(name: &str, value: CandidateValue) -> PolarsResult<Series> {
    match value {
        CandidateValue::Signed(v) => Ok(Series::new(PlSmallStr::from(name), vec![v])),
        CandidateValue::Float(v) => Ok(Series::new(PlSmallStr::from(name), vec![v])),
        CandidateValue::Unsigned(v) if v <= i128::MAX as u128 => {
            Ok(Series::new(PlSmallStr::from(name), vec![v as i128]))
        }
        CandidateValue::Unsigned(_) => Err(PolarsError::ComputeError(
            "Cannot synthesize a sum target beyond the i128 range".into(),
        )),
    }
}

/// Enforces `target == sum(addends)` per row. The deterministic repair
/// recomputes a violating target cell as the operand sum itself, keeping
/// the column's own numeric constraints
/// (bounds, membership). When that candidate is rejected, joint rejection
/// sampling over every operand gives a bounded second chance before an
/// honest error.
fn enforce_sum(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    addends: &[String],
    target: &str,
    tolerance: Option<f64>,
) -> PolarsResult<()> {
    for name in addends {
        if !lookup.contains_key(name.as_str()) {
            return Err(missing_sample_column(name));
        }
    }
    let target_spec = lookup
        .get(target)
        .copied()
        .ok_or_else(|| missing_sample_column(target))?;
    let tolerance_value = tolerance.unwrap_or(0.0);
    let bounds = numeric_bounds(&specs_of(target_spec));
    let one_of_pool = one_of_numeric_values(&specs_of(target_spec));
    let n = frame
        .get(target)
        .ok_or_else(|| missing_sample_column(target))?
        .len();

    let mut addend_cells: Vec<Vec<SumCell>> = addends
        .iter()
        .map(|name| extract_sum_cells(frame.get(name.as_str()).unwrap()))
        .collect::<PolarsResult<_>>()?;
    let mut target_cells = extract_sum_cells(
        frame
            .get(target)
            .ok_or_else(|| missing_sample_column(target))?,
    )?;

    for row in 0..n {
        let addend_row: Vec<SumCell> = addend_cells.iter().map(|column| column[row]).collect();
        let target_cell = target_cells[row];
        if addend_row.iter().any(SumCell::is_null) || target_cell.is_null() {
            continue; // a null operand skips the row, mirroring validation
        }
        let Some(sum) = row_sum(&addend_row) else {
            continue; // exact arithmetic overflowed with present values
        };
        if sum_matches_target(&sum, &target_cell, tolerance_value) {
            continue;
        }
        // Deterministic repair: recompute the target from its own value.
        let desired = desired_target_value(&sum, &target_cell);
        let repaired =
            desired.filter(|value| target_accommodates(&bounds, one_of_pool.as_ref(), value));
        if let Some(value) = repaired {
            if let Ok(candidate) = candidate_series(target, value) {
                if replace_row(frame, lookup, target, row, &candidate).is_ok() {
                    if let Some(new_cell) = extract_sum_cells(&candidate)?.into_iter().next() {
                        target_cells[row] = new_cell;
                    }
                    continue;
                }
            }
        }
        // Joint rejection sampling: redraw every operand until the sum holds.
        let mut satisfied = false;
        for _ in 0..200 {
            let mut redrawn: Vec<SumCell> = Vec::with_capacity(addends.len() + 1);
            let mut ok = true;
            for name in addends {
                let candidate = resample_one(rng, lookup, name)?;
                ok &= replace_row(frame, lookup, name, row, &candidate).is_ok();
                redrawn.push(extract_sum_cells(&candidate)?.remove(0));
            }
            let candidate = resample_one(rng, lookup, target)?;
            ok &= replace_row(frame, lookup, target, row, &candidate).is_ok();
            if !ok {
                continue;
            }
            let target_value = extract_sum_cells(&candidate)?.remove(0);
            if target_value.is_null() || redrawn.iter().any(SumCell::is_null) {
                continue;
            }
            if let Some(sum) = row_sum(&redrawn) {
                if sum_matches_target(&sum, &target_value, tolerance_value) {
                    addend_cells
                        .iter_mut()
                        .zip(redrawn)
                        .for_each(|(column, cell)| column[row] = cell);
                    target_cells[row] = target_value;
                    satisfied = true;
                    break;
                }
            }
        }
        if !satisfied {
            return Err(PolarsError::ComputeError(
                format!(
                    "Could not satisfy sum '{} == {}' in schema",
                    addends
                        .iter()
                        .map(|name| name.as_str())
                        .collect::<Vec<_>>()
                        .join(" + "),
                    target
                )
                .into(),
            ));
        }
    }
    Ok(())
}

/// Enforces `right - left <= max_gap_ns` on same-dtype temporal columns
/// via joint rejection sampling over the violated row's two cells.
fn enforce_within(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    left: &str,
    right: &str,
    max_gap_ns: i64,
) -> PolarsResult<()> {
    let left_spec = lookup
        .get(left)
        .copied()
        .ok_or_else(|| missing_sample_column(left))?;
    let right_spec = lookup
        .get(right)
        .copied()
        .ok_or_else(|| missing_sample_column(right))?;
    if !is_temporal_dtype(&left_spec.dtype) || !is_temporal_dtype(&right_spec.dtype) {
        return Err(PolarsError::ComputeError(
            format!("Cannot generate gap '{left}' ... '{right}': columns must be temporal").into(),
        ));
    }
    if left_spec.dtype != right_spec.dtype {
        return Err(PolarsError::ComputeError(
            format!(
                "Cannot generate gap '{left}' ... '{right}': temporal columns must share a dtype"
            )
            .into(),
        ));
    }
    let scale = ns_per_tick(&left_spec.dtype);
    let gap = i128::from(max_gap_ns);
    let n = frame
        .get(left)
        .ok_or_else(|| missing_sample_column(left))?
        .len();
    let mut left_cells = int128_cells(frame.get(left).ok_or_else(|| missing_sample_column(left))?)?;
    let mut right_cells = int128_cells(
        frame
            .get(right)
            .ok_or_else(|| missing_sample_column(right))?,
    )?;
    for row in 0..n {
        let within_gap = match (left_cells[row], right_cells[row]) {
            (Some(l), Some(r)) => (r - l).checked_mul(scale).is_some_and(|d| d <= gap),
            _ => true, // a null cell skips the row
        };
        if within_gap {
            continue;
        }
        let mut satisfied = false;
        for _ in 0..200 {
            let left_candidate = resample_one(rng, lookup, left)?;
            let right_candidate = resample_one(rng, lookup, right)?;
            let new_left = int128_cells(&left_candidate)?.remove(0);
            let new_right = int128_cells(&right_candidate)?.remove(0);
            if let (Some(l), Some(r)) = (new_left, new_right) {
                if (r - l).checked_mul(scale).is_some_and(|d| d <= gap) {
                    replace_row(frame, lookup, left, row, &left_candidate)?;
                    replace_row(frame, lookup, right, row, &right_candidate)?;
                    left_cells[row] = new_left;
                    right_cells[row] = new_right;
                    satisfied = true;
                    break;
                }
            }
        }
        if !satisfied {
            return Err(PolarsError::ComputeError(
                format!(
                    "Could not satisfy gap constraint '{right} - {left} <= {max_gap_ns}ns' in schema"
                )
                .into(),
            ));
        }
    }
    Ok(())
}

/// Enforces that a present `left` value requires a present `by` value by
/// replacing violated null cells with fresh non-null draws. Null `left`
/// rows keep their free-form `by` nulls.
fn enforce_implies(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    left: &str,
    by: &str,
) -> PolarsResult<()> {
    for name in [left, by] {
        if !lookup.contains_key(name) {
            return Err(missing_sample_column(name));
        }
    }
    let left_series = frame.get(left).ok_or_else(|| missing_sample_column(left))?;
    let by_series = frame.get(by).ok_or_else(|| missing_sample_column(by))?;
    let left_nulls = left_series.is_null();
    let by_nulls = by_series.is_null();
    for row in 0..by_series.len() {
        // `is_null()` yields `Option<bool>`: `Some(true)` where null.
        if left_nulls.get(row) == Some(false) && by_nulls.get(row) == Some(true) {
            continue;
        }
        let mut satisfied = false;
        for _ in 0..200 {
            let candidate = resample_one(rng, lookup, by)?;
            let value = candidate.get(0).ok();
            if let Some(value) = value {
                if !matches!(value, AnyValue::Null) {
                    replace_row(frame, lookup, by, row, &candidate)?;
                    satisfied = true;
                    break;
                }
            }
        }
        if !satisfied {
            return Err(PolarsError::ComputeError(
                format!("Could not generate a non-null value for '{by}' in schema").into(),
            ));
        }
    }
    Ok(())
}

/// Enforces the functional dependency `a -> b`: rows whose `b` value
/// disagrees with the first `b` mapped from their `a` value are redrawn
/// until they agree with the canonical mapping (or an honest error).
/// Null `a`/`b` cells never participate.
fn enforce_determines(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    a: &str,
    b: &str,
) -> PolarsResult<()> {
    for name in [a, b] {
        if !lookup.contains_key(name) {
            return Err(missing_sample_column(name));
        }
    }
    let a_series = frame.get(a).ok_or_else(|| missing_sample_column(a))?;
    let b_series = frame.get(b).ok_or_else(|| missing_sample_column(b))?;
    let mut first_b: HashMap<String, (String, usize)> = HashMap::new();
    let mut violations: Vec<(usize, String)> = Vec::new();
    for row in 0..b_series.len() {
        let a_value = match a_series.get(row).ok() {
            Some(value) if !matches!(value, AnyValue::Null) => format!("{value:?}"),
            _ => continue,
        };
        let b_value = match b_series.get(row).ok() {
            Some(value) if !matches!(value, AnyValue::Null) => format!("{value:?}"),
            _ => continue,
        };
        match first_b.entry(a_value) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let (seen, _) = entry.get();
                if *seen != b_value {
                    violations.push((row, seen.clone()));
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((b_value, row));
            }
        }
    }
    if violations.is_empty() {
        return Ok(());
    }
    for (row, canonical) in violations {
        let mut satisfied = false;
        for _ in 0..200 {
            let candidate = resample_one(rng, lookup, b)?;
            if let Ok(value) = candidate.get(0) {
                if !matches!(value, AnyValue::Null) && format!("{value:?}") == canonical {
                    replace_row(frame, lookup, b, row, &candidate)?;
                    satisfied = true;
                    break;
                }
            }
        }
        if !satisfied {
            return Err(PolarsError::ComputeError(
                format!("Could not satisfy the functional dependency '{a} -> {b}' in schema")
                    .into(),
            ));
        }
    }
    Ok(())
}

/// Generate a valid `DataFrame` satisfying the schema.
///
/// Strategies are derived from each column's dtype and checks, then
/// cross-column checks are enforced with rejection sampling. Because later
/// repairs (e.g. uniqueness) can disturb earlier ones (e.g. comparisons),
/// enforcement runs to a fixed point: each pass re-applies table checks,
/// repairs single-column uniqueness, and validates the whole frame.
/// Failures (contradictory bounds, unsatisfiable uniqueness) return an error.
pub fn sample(schema: &Schema, n: usize, seed: u64) -> PolarsResult<DataFrame> {
    const MAX_PASSES: usize = 10;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let lookup = spec_lookup(schema);
    let mut frame: HashMap<String, Series> = HashMap::with_capacity(schema.columns.len());

    for spec in &schema.columns {
        let name = PlSmallStr::from(spec.name.as_str());
        let specs = specs_of(spec);
        let series = gen_base_series(&mut rng, &name, &spec.dtype, &specs, n)?;
        // Explicit NotNull forbids nulls even when the column is nullable;
        // sampling nulls anyway would fail the schema on some seeds.
        let series =
            if spec.nullable && !specs.iter().any(|spec| matches!(spec, CheckSpec::NotNull)) {
                apply_nullable(series, &mut rng)?
            } else {
                series
            };
        frame.insert(spec.name.clone(), series);
    }

    for _ in 0..MAX_PASSES {
        for check in &schema.table_checks {
            match check.spec() {
                DataFrameCheckSpec::ColumnComparison { left, op, right } => {
                    enforce_comparison(&mut frame, &lookup, &mut rng, &left, op, &right)?;
                }
                DataFrameCheckSpec::Ordered { columns, strict } => {
                    enforce_ordered(&mut frame, &lookup, &mut rng, &columns, strict)?;
                }
                DataFrameCheckSpec::SumColumns {
                    addends,
                    target,
                    tolerance,
                } => {
                    enforce_sum(&mut frame, &lookup, &mut rng, &addends, &target, tolerance)?;
                }
                DataFrameCheckSpec::Implies { left, by } => {
                    enforce_implies(&mut frame, &lookup, &mut rng, &left, &by)?;
                }
                DataFrameCheckSpec::Determines { a, b } => {
                    enforce_determines(&mut frame, &lookup, &mut rng, &a, &b)?;
                }
                DataFrameCheckSpec::Within {
                    left,
                    right,
                    max_gap_ns,
                } => {
                    enforce_within(&mut frame, &lookup, &mut rng, &left, &right, max_gap_ns)?;
                }
                DataFrameCheckSpec::UniqueTogether { columns } => {
                    enforce_unique_together(&mut frame, &lookup, &mut rng, &columns)?;
                }
            }
        }

        repair_unique_columns(schema, &mut frame, &mut rng)?;

        if schema
            .validate(&frame_to_dataframe(schema, &frame, n)?)
            .is_valid()
        {
            return frame_to_dataframe(schema, &frame, n);
        }
    }

    Err(PolarsError::ComputeError(
        "Could not satisfy all schema constraints while sampling".into(),
    ))
}

/// Re-applies single-column uniqueness after table-check repairs.
fn repair_unique_columns(
    schema: &Schema,
    frame: &mut HashMap<String, Series>,
    rng: &mut ChaCha8Rng,
) -> PolarsResult<()> {
    for spec in &schema.columns {
        let series = frame
            .get(&spec.name)
            .ok_or_else(|| missing_sample_column(&spec.name))?;
        if has_unique(&specs_of(spec)) && series.n_unique().unwrap_or(0) != series.len() {
            let repaired = ensure_unique(series.clone(), rng, &spec.dtype, &specs_of(spec))?;
            frame.insert(spec.name.clone(), repaired);
        }
    }
    Ok(())
}

/// Assembles the working map into a `DataFrame` in schema column order.
fn frame_to_dataframe(
    schema: &Schema,
    frame: &HashMap<String, Series>,
    n: usize,
) -> PolarsResult<DataFrame> {
    let mut columns = Vec::with_capacity(frame.len());
    for spec in &schema.columns {
        columns.push(Column::from(
            frame
                .get(&spec.name)
                .ok_or_else(|| missing_sample_column(&spec.name))?
                .clone(),
        ));
    }
    DataFrame::new(n, columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checks::{GreaterThanF64, ListLenBetween, OneOfString, StrLengthBetween};
    use crate::column::ColumnSpec;
    use crate::dataframe_checks::{ColumnComparison, CompareOp, UniqueTogether};
    use std::collections::HashSet as StdHashSet;

    fn int_spec(name: &str, checks: Vec<Box<dyn crate::checks::Check>>) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            dtype: DataType::Int64,
            nullable: false,
            checks,
        }
    }

    #[test]
    fn samples_respect_numeric_bounds_and_strings() {
        let schema = Schema::new(vec![
            int_spec("age", vec![Box::new(GreaterThanF64 { min_value: 0.0 })]),
            ColumnSpec {
                name: "country".to_string(),
                dtype: DataType::String,
                nullable: false,
                checks: vec![Box::new(OneOfString {
                    allowed_values: ["FR".to_string(), "US".to_string()]
                        .into_iter()
                        .collect::<StdHashSet<_>>(),
                })],
            },
        ]);
        let frame = sample(&schema, 20, 42).unwrap();
        let report = schema.validate(&frame);
        assert!(report.is_valid());
        let again = sample(&schema, 20, 42).unwrap();
        assert_eq!(frame, again);
    }

    #[test]
    fn samples_satisfy_table_checks() {
        let schema = Schema::with_table_checks(
            vec![int_spec("start", vec![]), int_spec("end", vec![])],
            vec![
                Box::new(ColumnComparison {
                    left: "start".to_string(),
                    op: CompareOp::Lt,
                    right: "end".to_string(),
                }),
                Box::new(UniqueTogether {
                    columns: vec!["start".to_string(), "end".to_string()],
                }),
            ],
        );
        let frame = sample(&schema, 10, 7).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn unsatisfiable_bounds_return_an_error() {
        let schema = Schema::new(vec![int_spec(
            "value",
            vec![
                Box::new(GreaterThanF64 { min_value: 10.0 }),
                Box::new(crate::checks::LessThanF64 { max_value: 0.0 }),
            ],
        )]);
        assert!(sample(&schema, 5, 1).is_err());
    }

    #[test]
    fn samples_list_lengths() {
        let schema = Schema::new(vec![ColumnSpec {
            name: "tags".to_string(),
            dtype: DataType::List(Box::new(DataType::Int64)),
            nullable: false,
            checks: vec![Box::new(ListLenBetween {
                min_len: 1,
                max_len: 2,
            })],
        }]);
        let frame = sample(&schema, 10, 3).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn samples_deeply_nested_dtypes() {
        use crate::checks::StrLengthBetween;
        // List(List(Int64)) with an inner Array, and Struct containing a List.
        let deep_list = ColumnSpec {
            name: "matrix".to_string(),
            dtype: DataType::List(Box::new(DataType::Array(Box::new(DataType::Int64), 2))),
            nullable: false,
            checks: vec![Box::new(ListLenBetween {
                min_len: 1,
                max_len: 2,
            })],
        };
        let struct_with_list = ColumnSpec {
            name: "addresses".to_string(),
            dtype: DataType::Struct(vec![
                Field::new("city".into(), DataType::String),
                Field::new("codes".into(), DataType::List(Box::new(DataType::Int64))),
            ]),
            nullable: false,
            checks: vec![],
        };
        let wide_value = ColumnSpec {
            name: "wide".to_string(),
            dtype: DataType::String,
            nullable: false,
            checks: vec![Box::new(StrLengthBetween {
                min_len: 2,
                max_len: 4,
            })],
        };
        let schema = Schema::new(vec![deep_list, struct_with_list, wide_value]);
        let frame = sample(&schema, 8, 11).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn string_intersection_of_one_of_and_matches() {
        use crate::checks::MatchesRegex;
        let schema = Schema::new(vec![ColumnSpec {
            name: "code".to_string(),
            dtype: DataType::String,
            nullable: false,
            checks: vec![
                Box::new(OneOfString {
                    allowed_values: ["FR".to_string(), "12".to_string()]
                        .into_iter()
                        .collect::<StdHashSet<_>>(),
                }),
                Box::new(MatchesRegex {
                    pattern: regex::Regex::new(r"^[A-Z]{2}$").unwrap(),
                }),
            ],
        }]);
        let frame = sample(&schema, 10, 5).unwrap();
        assert!(schema.validate(&frame).is_valid());
        assert!(frame
            .column("code")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .all(|value| value == "FR"));
    }

    #[test]
    fn empty_string_intersection_returns_an_error() {
        use crate::checks::MatchesRegex;
        let schema = Schema::new(vec![ColumnSpec {
            name: "code".to_string(),
            dtype: DataType::String,
            nullable: false,
            checks: vec![
                Box::new(OneOfString {
                    allowed_values: ["xx".to_string()].into_iter().collect::<StdHashSet<_>>(),
                }),
                Box::new(MatchesRegex {
                    pattern: regex::Regex::new(r"^\d+$").unwrap(),
                }),
            ],
        }]);
        assert!(sample(&schema, 5, 5).is_err());
    }

    #[test]
    fn one_sided_bounds_beyond_the_fallback_stay_satisfiable() {
        let schema = Schema::new(vec![int_spec(
            "value",
            vec![Box::new(GreaterThanF64 { min_value: 2000.0 })],
        )]);
        let frame = sample(&schema, 10, 9).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn strict_integer_bounds_are_respected() {
        // gt(5) must never sample 5 itself.
        let schema = Schema::new(vec![int_spec(
            "value",
            vec![Box::new(GreaterThanF64 { min_value: 5.0 })],
        )]);
        for seed in 0..10 {
            let frame = sample(&schema, 50, seed).unwrap();
            assert!(schema.validate(&frame).is_valid());
            assert!(frame
                .column("value")
                .unwrap()
                .i64()
                .unwrap()
                .iter()
                .flatten()
                .all(|value| value > 5));
        }
    }

    #[test]
    fn empty_integer_intersection_returns_an_error() {
        // No integer lies strictly between 0.1 and 0.9.
        let schema = Schema::new(vec![int_spec(
            "value",
            vec![
                Box::new(GreaterThanF64 { min_value: 0.1 }),
                Box::new(crate::checks::LessThanF64 { max_value: 0.9 }),
            ],
        )]);
        assert!(sample(&schema, 5, 11).is_err());
    }

    #[test]
    fn nullable_not_null_columns_sample_without_nulls() {
        use crate::checks::NotNull;
        // Nullable but explicitly NotNull: nulls would fail validation, so
        // sampling must never inject them, on any seed.
        let schema = Schema::new(vec![ColumnSpec {
            name: "value".to_string(),
            dtype: DataType::Int64,
            nullable: true,
            checks: vec![Box::new(NotNull)],
        }]);
        for seed in 0..20 {
            let frame = sample(&schema, 30, seed).unwrap();
            assert!(schema.validate(&frame).is_valid());
            assert_eq!(frame.column("value").unwrap().null_count(), 0);
        }
    }

    #[test]
    fn unique_converges_on_narrow_domains() {
        use crate::checks::{BetweenF64, UniqueValues};
        // between(0, 9) holds exactly 10 integers: an exact fit for n=10.
        let schema = Schema::new(vec![ColumnSpec {
            name: "value".to_string(),
            dtype: DataType::Int64,
            nullable: false,
            checks: vec![
                Box::new(UniqueValues),
                Box::new(BetweenF64 {
                    min_value: 0.0,
                    max_value: 9.0,
                }),
            ],
        }]);
        let frame = sample(&schema, 10, 23).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn unique_pattern_strings_expand_by_mutation() {
        use crate::checks::{MatchesRegex, UniqueValues};
        let schema = Schema::new(vec![ColumnSpec {
            name: "code".to_string(),
            dtype: DataType::String,
            nullable: false,
            checks: vec![
                Box::new(UniqueValues),
                Box::new(MatchesRegex {
                    pattern: regex::Regex::new(r"^[A-Z]{2}$").unwrap(),
                }),
            ],
        }]);
        let frame = sample(&schema, 10, 31).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn single_valued_pattern_unique_errors() {
        use crate::checks::{MatchesRegex, UniqueValues};
        let schema = Schema::new(vec![ColumnSpec {
            name: "code".to_string(),
            dtype: DataType::String,
            nullable: false,
            checks: vec![
                Box::new(UniqueValues),
                Box::new(MatchesRegex {
                    pattern: regex::Regex::new(r"^ab$").unwrap(),
                }),
            ],
        }]);
        assert!(sample(&schema, 3, 37).is_err());
    }

    #[test]
    fn impossible_unique_demands_error_fast() {
        use crate::checks::UniqueValues;
        // Booleans hold 2 values; unique over 3 rows is impossible.
        let schema = Schema::new(vec![ColumnSpec {
            name: "flag".to_string(),
            dtype: DataType::Boolean,
            nullable: false,
            checks: vec![Box::new(UniqueValues)],
        }]);
        assert!(sample(&schema, 3, 29).is_err());
    }

    #[test]
    fn samples_temporal_comparisons() {
        let schema = Schema::with_table_checks(
            vec![
                ColumnSpec {
                    name: "start".to_string(),
                    dtype: DataType::Date,
                    nullable: false,
                    checks: vec![],
                },
                ColumnSpec {
                    name: "end".to_string(),
                    dtype: DataType::Date,
                    nullable: false,
                    checks: vec![],
                },
            ],
            vec![Box::new(ColumnComparison {
                left: "start".to_string(),
                op: CompareOp::Lt,
                right: "end".to_string(),
            })],
        );
        let frame = sample(&schema, 10, 13).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    #[test]
    fn missing_table_column_returns_an_error_instead_of_panicking() {
        let schema = Schema::with_table_checks(
            vec![int_spec("start", vec![])],
            vec![Box::new(ColumnComparison {
                left: "start".to_string(),
                op: CompareOp::Lt,
                right: "ghost".to_string(),
            })],
        );
        assert!(sample(&schema, 5, 17).is_err());
    }

    #[test]
    fn unique_and_comparison_converge_together() {
        use crate::checks::UniqueValues;
        let schema = Schema::with_table_checks(
            vec![
                ColumnSpec {
                    name: "start".to_string(),
                    dtype: DataType::Int64,
                    nullable: false,
                    checks: vec![Box::new(UniqueValues)],
                },
                int_spec("end", vec![]),
            ],
            vec![Box::new(ColumnComparison {
                left: "start".to_string(),
                op: CompareOp::Lt,
                right: "end".to_string(),
            })],
        );
        let frame = sample(&schema, 15, 19).unwrap();
        assert!(schema.validate(&frame).is_valid());
    }

    fn str_spec(name: &str, checks: Vec<Box<dyn crate::checks::Check>>) -> ColumnSpec {
        ColumnSpec {
            name: name.to_string(),
            dtype: DataType::String,
            nullable: false,
            checks,
        }
    }

    #[test]
    fn samples_satisfy_string_constraints() {
        use crate::checks::{NotBlank, StartsWith};
        let schema = Schema::new(vec![str_spec(
            "code",
            vec![
                Box::new(StrLengthBetween {
                    min_len: 3,
                    max_len: 20,
                }),
                Box::new(NotBlank),
                Box::new(StartsWith {
                    prefix: "t-".to_string(),
                }),
            ],
        )]);
        for seed in 0..5 {
            let frame = sample(&schema, 20, seed).unwrap();
            assert!(schema.validate(&frame).is_valid(), "seed {seed} failed");
            let code = frame.column("code").unwrap().str().unwrap();
            for value in code.iter().flatten() {
                assert!(value.starts_with("t-"));
                assert!(!value.trim().is_empty());
                assert!((3..=20).contains(&value.chars().count()));
            }
        }
    }

    #[test]
    fn samples_respect_ends_with() {
        use crate::checks::EndsWith;
        let schema = Schema::new(vec![str_spec(
            "file",
            vec![Box::new(EndsWith {
                suffix: ".txt".to_string(),
            })],
        )]);
        let frame = sample(&schema, 10, 2).unwrap();
        assert!(schema.validate(&frame).is_valid());
        assert!(frame
            .column("file")
            .unwrap()
            .str()
            .unwrap()
            .iter()
            .flatten()
            .all(|value| value.ends_with(".txt")));
    }

    #[test]
    fn prefix_longer_than_max_length_is_unsatisfiable() {
        use crate::checks::StartsWith;
        let schema = Schema::new(vec![str_spec(
            "code",
            vec![
                Box::new(StrLengthBetween {
                    min_len: 0,
                    max_len: 2,
                }),
                Box::new(StartsWith {
                    prefix: "abc".to_string(),
                }),
            ],
        )]);
        assert!(sample(&schema, 5, 1).is_err());
    }

    #[test]
    fn one_of_numeric_samples_allowed_integers_only() {
        use crate::checks::OneOfNumericValues;
        let schema = Schema::new(vec![int_spec(
            "level",
            vec![Box::new(OneOfNumericValues {
                allowed_values: vec![3.0, 5.0],
            })],
        )]);
        let frame = sample(&schema, 30, 8).unwrap();
        assert!(schema.validate(&frame).is_valid());
        let level = frame.column("level").unwrap().i64().unwrap();
        for value in level.iter().flatten() {
            assert!(value == 3 || value == 5);
        }
    }

    #[test]
    fn one_of_numeric_filters_candidates_by_the_column_dtype() {
        use crate::checks::OneOfNumericValues;
        // 500 does not fit Int8, so only 3 can ever be sampled.
        let schema = Schema::new(vec![ColumnSpec {
            name: "small".to_string(),
            dtype: DataType::Int8,
            nullable: false,
            checks: vec![Box::new(OneOfNumericValues {
                allowed_values: vec![3.0, 500.0],
            })],
        }]);
        let frame = sample(&schema, 20, 4).unwrap();
        assert!(schema.validate(&frame).is_valid());
        assert!(frame
            .column("small")
            .unwrap()
            .i8()
            .unwrap()
            .iter()
            .flatten()
            .all(|value| value == 3));
    }

    #[test]
    fn one_of_numeric_fractional_value_on_int_column_errors() {
        use crate::checks::OneOfNumericValues;
        let schema = Schema::new(vec![int_spec(
            "value",
            vec![Box::new(OneOfNumericValues {
                allowed_values: vec![0.5],
            })],
        )]);
        assert!(sample(&schema, 5, 6).is_err());
    }

    #[test]
    fn one_of_numeric_pins_boolean_columns() {
        use crate::checks::OneOfNumericValues;
        let schema = Schema::new(vec![ColumnSpec {
            name: "flag".to_string(),
            dtype: DataType::Boolean,
            nullable: false,
            checks: vec![Box::new(OneOfNumericValues {
                allowed_values: vec![1.0],
            })],
        }]);
        let frame = sample(&schema, 10, 1).unwrap();
        assert!(schema.validate(&frame).is_valid());
        assert!(frame
            .column("flag")
            .unwrap()
            .bool()
            .unwrap()
            .iter()
            .flatten()
            .all(|value| value));
    }

    #[test]
    fn one_of_numeric_on_temporal_column_errors() {
        use crate::checks::OneOfNumericValues;
        let schema = Schema::new(vec![ColumnSpec {
            name: "day".to_string(),
            dtype: DataType::Date,
            nullable: false,
            checks: vec![Box::new(OneOfNumericValues {
                allowed_values: vec![3.0, 5.0],
            })],
        }]);
        assert!(sample(&schema, 5, 2).is_err());
    }

    #[test]
    fn samples_satisfy_sum_columns() {
        let schema = Schema::with_table_checks(
            vec![
                int_spec("a", vec![Box::new(GreaterThanF64 { min_value: 0.0 })]),
                int_spec("b", vec![]),
            ],
            vec![Box::new(crate::dataframe_checks::SumColumns {
                addends: vec!["a".to_string(), "a".to_string()],
                target: "b".to_string(),
                tolerance: None,
            })],
        );
        let frame = sample(&schema, 10, 42).unwrap();
        let report = schema.validate(&frame);
        assert!(
            report.is_valid(),
            "{}",
            report
                .errors
                .iter()
                .map(|error| error.message.clone())
                .collect::<Vec<_>>()
                .join("; ")
        );
    }
}
