use std::collections::HashMap;

use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use regex::Regex;

use crate::checks::{is_supported_numeric_dtype, CheckSpec};
use crate::column::ColumnSpec;
use crate::dataframe_checks::{CompareOp, DataFrameCheckSpec};
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

fn gen_string_value(rng: &mut ChaCha8Rng, specs: &[CheckSpec], row: usize) -> PolarsResult<String> {
    if let Some(allowed) = one_of_values(specs) {
        if allowed.is_empty() {
            return Err(PolarsError::ComputeError(
                "Unsatisfiable one_of with no allowed values".into(),
            ));
        }
        let patterns = matches_patterns(specs);
        if patterns.is_empty() {
            return Ok(allowed[rng.random_range(0..allowed.len())].clone());
        }
        // Both membership and regex constraints apply: sample from their
        // intersection so generated values satisfy the whole schema.
        let regexes = patterns
            .iter()
            .map(|pattern| {
                Regex::new(pattern).map_err(|_| {
                    PolarsError::ComputeError(format!("Invalid regex: {pattern}").into())
                })
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        let matching: Vec<&String> = allowed
            .iter()
            .filter(|value| regexes.iter().all(|regex| regex.is_match(value)))
            .collect();
        if matching.is_empty() {
            return Err(PolarsError::ComputeError(
                "Unsatisfiable one_of and matches constraints in schema".into(),
            ));
        }
        return Ok(matching[rng.random_range(0..matching.len())].clone());
    }
    let patterns = matches_patterns(specs);
    for pattern in &patterns {
        if let Some(canned) = canned_match(pattern) {
            return Ok(canned);
        }
    }
    if patterns.is_empty() {
        Ok(format!("value-{row}"))
    } else {
        Ok("valid".to_string())
    }
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
    // Sample in the integer domain so every value satisfies the bounds
    // exactly; float sampling plus truncation can land outside them.
    let min_width = int_min_width(specs, n);
    if *dtype == DataType::UInt128 {
        let fallback = (0u128, 1_000_000u128);
        let (lo, hi) = int_range_u128(specs, fallback, min_width)?;
        let values: Vec<u128> = (0..n).map(|_| rng.random_range(lo..=hi)).collect();
        return cast_to_dtype(Series::new(name.clone(), values), dtype);
    }
    let (dmin, dmax, fallback) = match dtype {
        DataType::Int8 => (
            i128::from(i8::MIN),
            i128::from(i8::MAX),
            (-100i128, 100i128),
        ),
        DataType::UInt8 => (0, i128::from(u8::MAX), (0, 200)),
        DataType::UInt16 => (0, i128::from(u16::MAX), (0, 1000)),
        DataType::UInt32 => (0, i128::from(u32::MAX), (0, 100_000)),
        DataType::UInt64 => (0, i128::from(u64::MAX), (0, 1_000_000)),
        DataType::Int128 => (i128::MIN, i128::MAX, (0, 1000)),
        _ => (i128::from(i64::MIN), i128::from(i64::MAX), (0, 1000)),
    };
    let (lo, hi) = int_range_i128(specs, dmin, dmax, fallback, min_width)?;
    let values: Vec<i128> = (0..n).map(|_| rng.random_range(lo..=hi)).collect();
    cast_to_dtype(Series::new(name.clone(), values), dtype)
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
        if min_unique > 0 && ((hi - lo) as u128) < min_unique {
            if !bounds.lo.is_finite() {
                let room = ((hi - dmin).max(0)) as u128;
                lo = hi.saturating_sub(min_unique.min(room) as i128).max(dmin);
            } else if !bounds.hi.is_finite() {
                let room = ((dmax - lo).max(0)) as u128;
                hi = lo.saturating_add(min_unique.min(room) as i128).min(dmax);
            } else {
                return Err(unsatisfiable_bounds());
            }
            if ((hi - lo).max(0) as u128) < min_unique {
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
            .saturating_add(span.min((dmax - lo).max(0) as u128) as i128)
            .min(dmax);
    } else if !bounds.lo.is_finite() && bounds.hi.is_finite() {
        hi = raw_hi.clamp(dmin, dmax);
        lo = hi
            .saturating_sub(span.min((hi - dmin).max(0) as u128) as i128)
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

fn gen_list_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    inner: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    if matches!(
        inner,
        DataType::List(_) | DataType::Array(_, _) | DataType::Struct(_)
    ) {
        return Err(PolarsError::ComputeError(
            "Data generation supports one nesting level only".into(),
        ));
    }
    let (min_len, max_len) = list_len_bounds(specs);
    if min_len > max_len {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable list length bounds in schema".into(),
        ));
    }
    let mut inners = Vec::with_capacity(n);
    for _ in 0..n {
        let len = if min_len == max_len {
            min_len
        } else {
            rng.random_range(min_len..=max_len)
        };
        let values = gen_scalar_series(rng, &PlSmallStr::from(""), inner, &[], len)?;
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

/// Uniqueness repair that preserves column checks (e.g. one_of domains).
fn ensure_unique(
    series: Series,
    rng: &mut ChaCha8Rng,
    dtype: &DataType,
    specs: &[CheckSpec],
) -> PolarsResult<Series> {
    let name = series.name().clone();
    let mut current = series;
    for _ in 0..(current.len() * 10 + 10) {
        if current.n_unique().unwrap_or(0) == current.len() {
            return Ok(current);
        }
        current = gen_scalar_series(rng, &name, dtype, specs, current.len())?;
    }
    Err(PolarsError::ComputeError(
        "Could not generate unique values for column".into(),
    ))
}

fn apply_nullable(series: Series, rng: &mut ChaCha8Rng, nullable: bool) -> PolarsResult<Series> {
    if !nullable || series.is_empty() {
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
    let kind = match (left_kind, right_kind) {
        (CellKind::Text, CellKind::Text) => CellKind::Text,
        (CellKind::UInt, CellKind::UInt) => CellKind::UInt,
        (CellKind::Float, _) | (_, CellKind::Float) => CellKind::Float,
        (CellKind::Int, CellKind::Int) => CellKind::Int,
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
        let series = apply_nullable(series, &mut rng, spec.nullable)?;
        frame.insert(spec.name.clone(), series);
    }

    for _ in 0..MAX_PASSES {
        for check in &schema.table_checks {
            match check.spec() {
                DataFrameCheckSpec::ColumnComparison { left, op, right } => {
                    enforce_comparison(&mut frame, &lookup, &mut rng, &left, op, &right)?;
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
    use crate::checks::{GreaterThanF64, ListLenBetween, OneOfString};
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
}
