use polars::prelude::*;
use regex::Regex;
use std::collections::HashSet;

use crate::errors::ValidationError;

#[derive(Clone)]
pub enum CheckSpec {
    NotNull,
    UniqueValues,
    GreaterThan(f64),
    LessThan(f64),
    GreaterThanOrEqual(f64),
    LessThanOrEqual(f64),
    Between(f64, f64),
    Matches(String),
    OneOf(Vec<String>),
    OneOfNumeric(Vec<f64>),
    StrLengthBetween(usize, usize),
    NotBlank,
    StartsWith(String),
    EndsWith(String),
    ListLenBetween(usize, usize),
    ListNotEmpty,
    StructField(String, Vec<CheckSpec>),
}

pub trait Check: Send + Sync {
    fn name(&self) -> &str;
    fn spec(&self) -> CheckSpec;
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError>;
}

/// Checks that no values in the series are null.
pub struct NotNull;

impl Check for NotNull {
    fn name(&self) -> &str {
        "not_null"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::NotNull
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if series.null_count() > 0 {
            let mask = series.is_null();
            let rows = get_true_indices(&mask);

            errors.push(ValidationError::single(
                column_name,
                self.name(),
                rows,
                format!("found {} null value(s)", series.null_count()),
            ));
        }
        errors
    }
}

/// Checks that all values in the series are unique (no duplicates).
pub struct UniqueValues;

impl Check for UniqueValues {
    fn name(&self) -> &str {
        "unique"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::UniqueValues
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(df) = DataFrame::new(series.len(), vec![series.clone().into()]) {
            if let Ok(mask) = df.is_duplicated() {
                if mask.any() {
                    let rows = get_true_indices(&mask);
                    errors.push(ValidationError::single(
                        column_name,
                        self.name(),
                        rows,
                        "contains duplicated values",
                    ));
                }
            }
        }

        errors
    }
}

/// Checks that all numeric values are STRICTLY GREATER than the provided threshold.
pub struct GreaterThanF64 {
    pub min_value: f64,
}

impl Check for GreaterThanF64 {
    fn name(&self) -> &str {
        "gt"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::GreaterThan(self.min_value)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mask = match int_fail_mask(series, IntPredicate::LessOrEqualFloor(self.min_value)) {
            Some(mask) => mask,
            None => numeric_series(series).map(|ca| ca.lt_eq(self.min_value)),
        };
        match mask {
            Ok(mask) if mask.any() => vec![ValidationError::single(
                column_name,
                self.name(),
                get_true_indices(&mask),
                format!("value must be > {}", self.min_value),
            )],
            Ok(_) => Vec::new(),
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            )],
        }
    }
}

/// Checks that all numeric values are STRICTLY LESS than the provided threshold.
pub struct LessThanF64 {
    pub max_value: f64,
}

impl Check for LessThanF64 {
    fn name(&self) -> &str {
        "lt"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::LessThan(self.max_value)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mask = match int_fail_mask(series, IntPredicate::GreaterOrEqualCeil(self.max_value)) {
            Some(mask) => mask,
            None => numeric_series(series).map(|ca| ca.gt_eq(self.max_value)),
        };
        match mask {
            Ok(mask) if mask.any() => vec![ValidationError::single(
                column_name,
                self.name(),
                get_true_indices(&mask),
                format!("value must be < {}", self.max_value),
            )],
            Ok(_) => Vec::new(),
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            )],
        }
    }
}

pub struct GreaterThanOrEqualF64 {
    pub min_value: f64,
}

impl Check for GreaterThanOrEqualF64 {
    fn name(&self) -> &str {
        "gte"
    }

    fn spec(&self) -> CheckSpec {
        CheckSpec::GreaterThanOrEqual(self.min_value)
    }

    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mask = match int_fail_mask(series, IntPredicate::LessThanCeil(self.min_value)) {
            Some(mask) => mask,
            None => numeric_series(series).map(|ca| ca.lt(self.min_value)),
        };
        match mask {
            Ok(mask) if mask.any() => {
                vec![ValidationError::single(
                    column_name,
                    self.name(),
                    get_true_indices(&mask),
                    format!("value must be >= {}", self.min_value),
                )]
            }
            Ok(_) => Vec::new(),
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            )],
        }
    }
}

pub struct LessThanOrEqualF64 {
    pub max_value: f64,
}

impl Check for LessThanOrEqualF64 {
    fn name(&self) -> &str {
        "lte"
    }

    fn spec(&self) -> CheckSpec {
        CheckSpec::LessThanOrEqual(self.max_value)
    }

    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mask = match int_fail_mask(series, IntPredicate::GreaterThanFloor(self.max_value)) {
            Some(mask) => mask,
            None => numeric_series(series).map(|ca| ca.gt(self.max_value)),
        };
        match mask {
            Ok(mask) if mask.any() => {
                vec![ValidationError::single(
                    column_name,
                    self.name(),
                    get_true_indices(&mask),
                    format!("value must be <= {}", self.max_value),
                )]
            }
            Ok(_) => Vec::new(),
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            )],
        }
    }
}

/// Checks that all numeric values are inclusively between `min_value` and `max_value`.
pub struct BetweenF64 {
    pub min_value: f64,
    pub max_value: f64,
}

impl Check for BetweenF64 {
    fn name(&self) -> &str {
        "between"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::Between(self.min_value, self.max_value)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mask = match int_fail_mask(
            series,
            IntPredicate::OutsideCeilFloor(self.min_value, self.max_value),
        ) {
            Some(mask) => mask,
            None => numeric_series(series).map(|ca| {
                let mask_lt = ca.lt(self.min_value);
                let mask_gt = ca.gt(self.max_value);
                mask_lt | mask_gt
            }),
        };
        match mask {
            Ok(mask) if mask.any() => {
                let rows = get_true_indices(&mask);
                vec![ValidationError::single(
                    column_name,
                    self.name(),
                    rows,
                    format!(
                        "value must be between {} and {}",
                        self.min_value, self.max_value
                    ),
                )]
            }
            Ok(_) => Vec::new(),
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            )],
        }
    }
}

/// Checks that string values match a provided Regular Expression.
pub struct MatchesRegex {
    pub pattern: Regex,
}

impl Check for MatchesRegex {
    fn name(&self) -> &str {
        "matches"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::Matches(self.pattern.as_str().to_string())
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();
            let mut sample: Option<&str> = None;

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if !self.pattern.is_match(val) {
                        failed_rows.push(i);
                        if sample.is_none() {
                            sample = Some(val);
                        }
                    }
                }
            }

            if !failed_rows.is_empty() {
                let offender = sample
                    .map(|value| format!(" (first offender: '{}')", truncate_sample(value)))
                    .unwrap_or_default();
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!(
                        "does not match pattern '{}'{offender}",
                        self.pattern.as_str()
                    ),
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "a String"));
        }

        errors
    }
}

/// Checks that all string values are within an allowed set.
pub struct OneOfString {
    pub allowed_values: HashSet<String>,
}

impl Check for OneOfString {
    fn name(&self) -> &str {
        "one_of"
    }
    fn spec(&self) -> CheckSpec {
        let mut values: Vec<String> = self.allowed_values.iter().cloned().collect();
        values.sort_unstable();
        CheckSpec::OneOf(values)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();
            let mut offenders: Vec<&str> = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if !self.allowed_values.contains(val) {
                        failed_rows.push(i);
                        if offenders.len() < SAMPLE_VALUES {
                            offenders.push(val);
                        }
                    }
                }
            }

            if !failed_rows.is_empty() {
                let mut allowed: Vec<&str> =
                    self.allowed_values.iter().map(String::as_str).collect();
                allowed.sort_unstable();
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!(
                        "values {} are not one of {}",
                        render_quoted_list(&offenders, SAMPLE_VALUES),
                        render_quoted_list(&allowed, SAMPLE_ALLOWED),
                    ),
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "a String"));
        }

        errors
    }
}

/// Checks that all numeric/boolean values are within an allowed set.
pub struct OneOfNumericValues {
    pub allowed_values: Vec<f64>,
}

impl OneOfNumericValues {
    /// Exact membership sets derived from the allowed values: integers
    /// within the i128 and u128 domains, plus an f64 pool for float rows.
    /// A value enters an integer set only when the `f64 -> int -> f64`
    /// round-trip is exact, so integer equality never goes through a
    /// lossy comparison.
    fn exact_sets(&self) -> (HashSet<i128>, HashSet<u128>, Vec<f64>) {
        let mut signed = HashSet::with_capacity(self.allowed_values.len());
        let mut unsigned = HashSet::with_capacity(self.allowed_values.len());
        let mut floats: Vec<f64> = self
            .allowed_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        floats.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        floats.dedup();
        for value in &self.allowed_values {
            if let Some(i) = exact_i128_value(*value) {
                signed.insert(i);
            }
            if let Some(u) = exact_u128_value(*value) {
                unsigned.insert(u);
            }
        }
        (signed, unsigned, floats)
    }
}

/// A finite, integral `f64` that is exactly representable as an `i128`.
/// Integral `f64` values strictly inside the domain edge bounds cast
/// exactly; the fractional test plus the edge bounds make edge-of-domain
/// cases (e.g. `2^127`, which saturates) return `None` instead.
pub(crate) fn exact_i128_value(value: f64) -> Option<i128> {
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    if !(-TWO_POW_127..TWO_POW_127).contains(&value) {
        return None;
    }
    Some(value as i128)
}

/// Companion of `exact_i128_value` for the unsigned domain.
pub(crate) fn exact_u128_value(value: f64) -> Option<u128> {
    if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
        return None;
    }
    if value >= TWO_POW_128 {
        return None;
    }
    Some(value as u128)
}

fn int_not_in_rows_i128(values: &Int128Chunked, allowed: &HashSet<i128>) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, cell)| match cell {
            Some(value) if !allowed.contains(&value) => Some(i),
            _ => None,
        })
        .collect()
}

fn int_not_in_rows_u128(values: &UInt128Chunked, allowed: &HashSet<u128>) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, cell)| match cell {
            Some(value) if !allowed.contains(&value) => Some(i),
            _ => None,
        })
        .collect()
}

fn float_not_in_rows(values: &Float64Chunked, allowed: &[f64]) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(i, cell)| match cell {
            Some(value)
                if allowed
                    .binary_search_by(|p| p.partial_cmp(&value).unwrap())
                    .is_err() =>
            {
                Some(i)
            }
            _ => None,
        })
        .collect()
}

impl Check for OneOfNumericValues {
    fn name(&self) -> &str {
        "one_of"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::OneOfNumeric(self.allowed_values.clone())
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let (allow_i, allow_u, allow_f) = self.exact_sets();
        // Booleans compare after widening to Int64, matching the generation
        // side where `[true]` becomes the numeric pool `{1.0}`.
        let bool_wide: Option<Series> = match series.dtype() {
            DataType::Boolean => match series.cast(&DataType::Int64) {
                Ok(wide) => Some(wide),
                Err(_) => {
                    return vec![type_mismatch_error(
                        column_name,
                        self.name(),
                        "numeric, boolean, or temporal",
                    )]
                }
            },
            _ => None,
        };
        let target: &Series = bool_wide.as_ref().unwrap_or(series);
        let rows: PolarsResult<Vec<usize>> = (|| match int_domain(target.dtype()) {
            Some(IntDomain::I128) => Ok(int_not_in_rows_i128(
                target.cast(&DataType::Int128)?.i128()?,
                &allow_i,
            )),
            Some(IntDomain::U128) => Ok(int_not_in_rows_u128(
                target.cast(&DataType::UInt128)?.u128()?,
                &allow_u,
            )),
            None if is_supported_numeric_dtype(target.dtype()) => {
                Ok(float_not_in_rows(&numeric_series(target)?, &allow_f))
            }
            None => Err(PolarsError::ComputeError(
                "expected a membership type".into(),
            )),
        })();
        match rows {
            Ok(rows) if !rows.is_empty() => vec![ValidationError::single(
                column_name,
                self.name(),
                rows,
                format!("values are not one of {}", render_float_list(&allow_f)),
            )],
            Ok(_) => Vec::new(),
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "numeric, boolean, or temporal",
            )],
        }
    }
}

/// Checks that string values are within `[min_len, max_len]` characters.
/// Rows outside the range are collected into one error and nulls are
/// skipped, mirroring the other string checks.
pub struct StrLengthBetween {
    pub min_len: usize,
    pub max_len: usize,
}

impl Check for StrLengthBetween {
    fn name(&self) -> &str {
        "str_length_between"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::StrLengthBetween(self.min_len, self.max_len)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    let len = val.chars().count();
                    if len < self.min_len || len > self.max_len {
                        failed_rows.push(i);
                    }
                }
            }

            if !failed_rows.is_empty() {
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!(
                        "value length must be between {} and {} characters",
                        self.min_len, self.max_len
                    ),
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "a String"));
        }

        errors
    }
}

/// Checks that string values are never empty or whitespace-only. Nulls are
/// skipped; trimming follows Rust's Unicode-aware `str::trim`.
pub struct NotBlank;

impl Check for NotBlank {
    fn name(&self) -> &str {
        "not_blank"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::NotBlank
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if val.trim().is_empty() {
                        failed_rows.push(i);
                    }
                }
            }

            if !failed_rows.is_empty() {
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    "value must not be blank (empty or whitespace-only)",
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "a String"));
        }

        errors
    }
}

/// Checks that string values start with the given prefix.
pub struct StartsWith {
    pub prefix: String,
}

impl Check for StartsWith {
    fn name(&self) -> &str {
        "starts_with"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::StartsWith(self.prefix.clone())
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if !val.starts_with(self.prefix.as_str()) {
                        failed_rows.push(i);
                    }
                }
            }

            if !failed_rows.is_empty() {
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!("value must start with '{}'", self.prefix),
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "a String"));
        }

        errors
    }
}

/// Checks that string values end with the given suffix.
pub struct EndsWith {
    pub suffix: String,
}

impl Check for EndsWith {
    fn name(&self) -> &str {
        "ends_with"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::EndsWith(self.suffix.clone())
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if !val.ends_with(self.suffix.as_str()) {
                        failed_rows.push(i);
                    }
                }
            }

            if !failed_rows.is_empty() {
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!("value must end with '{}'", self.suffix),
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "a String"));
        }

        errors
    }
}

// --- Helper Functions ---

/// Maximum distinct offending values rendered inside an error message.
const SAMPLE_VALUES: usize = 3;
/// Maximum allowed-set values rendered inside an error message.
const SAMPLE_ALLOWED: usize = 8;
/// Maximum characters of a single offender rendered inside an error message.
const SAMPLE_VALUE_CHARS: usize = 40;

/// Renders quoted values as a list, collapsing anything past `cap` into an
/// ellipsis so pathological columns cannot inflate the message.
fn render_quoted_list(values: &[&str], cap: usize) -> String {
    let shown = values
        .iter()
        .take(cap)
        .map(|value| format!("'{value}'"))
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > cap {
        format!("[{shown}, …]")
    } else {
        format!("[{shown}]")
    }
}

/// Renders the numeric allowed pool for `one_of` failure messages, capped
/// the same way as the string variant.
fn render_float_list(values: &[f64]) -> String {
    let shown = values
        .iter()
        .take(SAMPLE_ALLOWED)
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    if values.len() > SAMPLE_ALLOWED {
        format!("[{shown}, …]")
    } else {
        format!("[{shown}]")
    }
}

/// Truncates a value sample to `SAMPLE_VALUE_CHARS` characters with an
/// ellipsis marker, so long cells never dominate a failure message.
fn truncate_sample(value: &str) -> String {
    let mut out = String::new();
    let mut chars = value.chars();
    for _ in 0..SAMPLE_VALUE_CHARS {
        let Some(c) = chars.next() else { break };
        out.push(c);
    }
    if chars.next().is_some() {
        out.push('…');
    }
    out
}

/// Whether a dtype can be coerced to f64 for numeric checks and sampling.
pub(crate) fn is_supported_numeric_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Int8
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
            | DataType::Date
            | DataType::Datetime(_, _)
            | DataType::Duration(_)
            | DataType::Time
    )
}

/// Failure predicate for a numeric check, expressed so it can be evaluated
/// exactly on integer values without rounding through `f64`.
enum IntPredicate {
    /// Fail iff `x <= floor(v)` (strict greater-than check).
    LessOrEqualFloor(f64),
    /// Fail iff `x >= ceil(v)` (strict less-than check).
    GreaterOrEqualCeil(f64),
    /// Fail iff `x < ceil(v)` (greater-than-or-equal check).
    LessThanCeil(f64),
    /// Fail iff `x > floor(v)` (less-than-or-equal check).
    GreaterThanFloor(f64),
    /// Fail iff `x < ceil(lo)` or `x > floor(hi)` (between check).
    OutsideCeilFloor(f64, f64),
}

/// Powers of two bounding the integer domains; exactly representable, so
/// out-of-domain threshold checks below are exact, not approximate.
const TWO_POW_127: f64 = 170141183460469231731687303715884105728.0;
const TWO_POW_128: f64 = 340282366920938463463374607431768211456.0;

/// How a predicate resolves against an integer domain before any casting.
enum Resolved {
    /// Every present value fails.
    AllFail,
    /// No value fails.
    AllPass,
    /// Compare against computed bounds normally.
    Compare,
}

impl IntPredicate {
    fn thresholds_finite(&self) -> bool {
        match *self {
            IntPredicate::LessOrEqualFloor(v)
            | IntPredicate::GreaterOrEqualCeil(v)
            | IntPredicate::LessThanCeil(v)
            | IntPredicate::GreaterThanFloor(v) => v.is_finite(),
            IntPredicate::OutsideCeilFloor(lo, hi) => lo.is_finite() && hi.is_finite(),
        }
    }

    /// Resolves the predicate against the domain edges: `dmin` is the
    /// smallest domain value and `above_max` the smallest f64 strictly
    /// greater than every domain value (`0.0`/`2^128` for `u128`,
    /// `-2^127`/`2^127` for signed). Thresholds outside the domain make the
    /// predicate trivially true or false for every value, which saturating
    /// casts alone would get wrong at the edges.
    fn resolve(&self, dmin: f64, above_max: f64) -> Resolved {
        // `fail iff x <= v` (gt) and `fail iff x < v` (gte) share their
        // low-side behavior. On the high side, `fail iff x >= v` (lt) fails
        // everything at `v == dmin`, but `fail iff x > v` (lte) still
        // passes the boundary value there — strictness matters exactly at
        // the edge, so lte gets its own strict comparison.
        let low_fail_all = |v: f64| v >= above_max;
        let low_fail_none = |v: f64| v < dmin;
        let high_fail_all = |v: f64| v <= dmin;
        let high_fail_none = |v: f64| v >= above_max;
        let lte_fail_all = |v: f64| v < dmin;
        // Unsigned zero edge: `x <= -0.0` still fails for `x == 0`, but any
        // strictly negative threshold is vacuous or total. The generic
        // comparisons below already encode this via `dmin == 0.0`.
        match *self {
            IntPredicate::LessOrEqualFloor(v) => {
                if low_fail_all(v) {
                    Resolved::AllFail
                } else if low_fail_none(v) {
                    Resolved::AllPass
                } else {
                    Resolved::Compare
                }
            }
            IntPredicate::LessThanCeil(v) => {
                if low_fail_all(v) {
                    Resolved::AllFail
                } else if low_fail_none(v) {
                    Resolved::AllPass
                } else {
                    Resolved::Compare
                }
            }
            IntPredicate::GreaterOrEqualCeil(v) => {
                if high_fail_all(v) {
                    Resolved::AllFail
                } else if high_fail_none(v) {
                    Resolved::AllPass
                } else {
                    Resolved::Compare
                }
            }
            IntPredicate::GreaterThanFloor(v) => {
                if lte_fail_all(v) {
                    Resolved::AllFail
                } else if high_fail_none(v) {
                    Resolved::AllPass
                } else {
                    Resolved::Compare
                }
            }
            IntPredicate::OutsideCeilFloor(lo, hi) => {
                // Fail-all if either side fails everything; pass-all only if
                // both sides are vacuous (vacuous sides are harmless below).
                let lo_side = if lo >= above_max {
                    Resolved::AllFail
                } else if lo <= dmin {
                    Resolved::AllPass
                } else {
                    Resolved::Compare
                };
                let hi_side = if hi < dmin {
                    Resolved::AllFail
                } else if hi >= above_max {
                    Resolved::AllPass
                } else {
                    Resolved::Compare
                };
                match (lo_side, hi_side) {
                    (Resolved::AllFail, _) | (_, Resolved::AllFail) => Resolved::AllFail,
                    (Resolved::AllPass, Resolved::AllPass) => Resolved::AllPass,
                    _ => Resolved::Compare,
                }
            }
        }
    }
}

/// Integer comparison domain. Every supported integer width (and temporal
/// types via `Int64`) widens losslessly into one of these two.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum IntDomain {
    I128,
    U128,
}

pub(crate) fn int_domain(dtype: &DataType) -> Option<IntDomain> {
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
        | DataType::Date
        | DataType::Datetime(_, _)
        | DataType::Duration(_)
        | DataType::Time => Some(IntDomain::I128),
        DataType::UInt128 => Some(IntDomain::U128),
        _ => None,
    }
}

/// Evaluates a numeric failure predicate exactly for integer and temporal
/// columns by widening to `i128`/`u128` and comparing against `floor`/`ceil`
/// bounds. Out-of-domain thresholds resolve to trivial all-fail/all-pass
/// masks first, so the saturating `as` casts below only ever see in-domain
/// values. Returns `None` for non-integer dtypes or non-finite thresholds,
/// in which case callers fall back to the `f64` path.
fn int_fail_mask(series: &Series, pred: IntPredicate) -> Option<PolarsResult<BooleanChunked>> {
    if !pred.thresholds_finite() {
        return None;
    }
    let domain = int_domain(series.dtype())?;
    // Thresholds outside the domain decide the whole mask without casting.
    let (dmin, above_max) = match domain {
        IntDomain::I128 => (-TWO_POW_127, TWO_POW_127),
        IntDomain::U128 => (0.0, TWO_POW_128),
    };
    match pred.resolve(dmin, above_max) {
        Resolved::AllFail => {
            // Every present value fails; nulls stay unset and are skipped.
            return Some(Ok(series.is_not_null()));
        }
        Resolved::AllPass => {
            return Some(Ok(BooleanChunked::full(
                series.name().clone(),
                false,
                series.len(),
            )));
        }
        Resolved::Compare => {}
    }
    let base = match series.dtype() {
        DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            series.cast(&DataType::Int64).ok()?
        }
        _ => series.clone(),
    };
    Some((|| match domain {
        IntDomain::I128 => {
            let values = base.cast(&DataType::Int128)?.i128()?.clone();
            Ok(match pred {
                IntPredicate::LessOrEqualFloor(v) => values.lt_eq(v.floor() as i128),
                IntPredicate::GreaterOrEqualCeil(v) => values.gt_eq(v.ceil() as i128),
                IntPredicate::LessThanCeil(v) => values.lt(v.ceil() as i128),
                IntPredicate::GreaterThanFloor(v) => values.gt(v.floor() as i128),
                IntPredicate::OutsideCeilFloor(lo, hi) => {
                    values.lt(lo.ceil() as i128) | values.gt(hi.floor() as i128)
                }
            })
        }
        IntDomain::U128 => {
            let values = base.cast(&DataType::UInt128)?.u128()?.clone();
            Ok(match pred {
                IntPredicate::LessOrEqualFloor(v) => values.lt_eq(v.floor() as u128),
                IntPredicate::GreaterOrEqualCeil(v) => values.gt_eq(v.ceil() as u128),
                IntPredicate::LessThanCeil(v) => values.lt(v.ceil() as u128),
                IntPredicate::GreaterThanFloor(v) => values.gt(v.floor() as u128),
                IntPredicate::OutsideCeilFloor(lo, hi) => {
                    values.lt(lo.ceil() as u128) | values.gt(hi.floor() as u128)
                }
            })
        }
    })())
}

/// Casts supported scalar numeric columns once so checks can use Polars' native
/// vectorized comparisons instead of converting values through Python.
pub(crate) fn numeric_series(series: &Series) -> PolarsResult<Float64Chunked> {
    if !is_supported_numeric_dtype(series.dtype()) {
        return Err(PolarsError::ComputeError("expected a numeric type".into()));
    }

    let cast_dtype = match series.dtype() {
        DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            DataType::Int64
        }
        _ => DataType::Float64,
    };
    series
        .cast(&cast_dtype)?
        .cast(&DataType::Float64)?
        .f64()
        .cloned()
}

/// Retrieves the raw row indexing where the BooleanChunked mask answers true.
pub(crate) fn get_true_indices(mask: &BooleanChunked) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, b)| if let Some(true) = b { Some(i) } else { None })
        .collect()
}

/// Returns a boilerplate error for failed typecasts.
pub(crate) fn type_mismatch_error(
    column: &str,
    check: &str,
    expected_type: &str,
) -> ValidationError {
    ValidationError::single(
        column,
        check,
        vec![],
        format!("column is not {expected_type}"),
    )
}

/// Build a boxed check from its serializable spec. Used for nested checks
/// (struct fields) so recursion stays in one place.
pub fn check_from_spec(spec: &CheckSpec) -> PolarsResult<Box<dyn Check>> {
    let check: Box<dyn Check> = match spec {
        CheckSpec::NotNull => Box::new(NotNull),
        CheckSpec::UniqueValues => Box::new(UniqueValues),
        CheckSpec::GreaterThan(v) => Box::new(GreaterThanF64 { min_value: *v }),
        CheckSpec::LessThan(v) => Box::new(LessThanF64 { max_value: *v }),
        CheckSpec::GreaterThanOrEqual(v) => Box::new(GreaterThanOrEqualF64 { min_value: *v }),
        CheckSpec::LessThanOrEqual(v) => Box::new(LessThanOrEqualF64 { max_value: *v }),
        CheckSpec::Between(lo, hi) => Box::new(BetweenF64 {
            min_value: *lo,
            max_value: *hi,
        }),
        CheckSpec::Matches(pattern) => Box::new(MatchesRegex {
            pattern: Regex::new(pattern).map_err(|_| {
                PolarsError::ComputeError(format!("invalid regex: {pattern}").into())
            })?,
        }),
        CheckSpec::OneOf(values) => Box::new(OneOfString {
            allowed_values: values.iter().cloned().collect(),
        }),
        CheckSpec::OneOfNumeric(values) => Box::new(OneOfNumericValues {
            allowed_values: values.clone(),
        }),
        CheckSpec::StrLengthBetween(min_len, max_len) => Box::new(StrLengthBetween {
            min_len: *min_len,
            max_len: *max_len,
        }),
        CheckSpec::NotBlank => Box::new(NotBlank),
        CheckSpec::StartsWith(prefix) => Box::new(StartsWith {
            prefix: prefix.clone(),
        }),
        CheckSpec::EndsWith(suffix) => Box::new(EndsWith {
            suffix: suffix.clone(),
        }),
        CheckSpec::ListLenBetween(lo, hi) => Box::new(ListLenBetween {
            min_len: *lo,
            max_len: *hi,
        }),
        CheckSpec::ListNotEmpty => Box::new(ListNotEmpty),
        CheckSpec::StructField(field, inner) => {
            let inner_checks = inner
                .iter()
                .map(check_from_spec)
                .collect::<PolarsResult<Vec<_>>>()?;
            Box::new(StructFieldCheck {
                field: field.clone(),
                inner_checks,
            })
        }
    };
    Ok(check)
}

/// Checks that list/array columns have a length within `[min_len, max_len]`.
pub struct ListLenBetween {
    pub min_len: usize,
    pub max_len: usize,
}

impl Check for ListLenBetween {
    fn name(&self) -> &str {
        "list_len_between"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::ListLenBetween(self.min_len, self.max_len)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        match nested_lengths(series) {
            Ok(lengths) => {
                let failed: Vec<usize> = lengths
                    .iter()
                    .enumerate()
                    .filter_map(|(i, len)| match len {
                        None => None,
                        Some(n) if *n < self.min_len || *n > self.max_len => Some(i),
                        _ => None,
                    })
                    .collect();
                if failed.is_empty() {
                    Vec::new()
                } else {
                    vec![ValidationError::single(
                        column_name,
                        self.name(),
                        failed,
                        format!(
                            "list length must be between {} and {}",
                            self.min_len, self.max_len
                        ),
                    )]
                }
            }
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a List or Array",
            )],
        }
    }
}

/// Checks that list/array columns are never empty (nulls are skipped).
pub struct ListNotEmpty;

impl Check for ListNotEmpty {
    fn name(&self) -> &str {
        "list_not_empty"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::ListNotEmpty
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        match nested_lengths(series) {
            Ok(lengths) => {
                let failed: Vec<usize> = lengths
                    .iter()
                    .enumerate()
                    .filter_map(|(i, len)| match len {
                        Some(0) => Some(i),
                        _ => None,
                    })
                    .collect();
                if failed.is_empty() {
                    Vec::new()
                } else {
                    vec![ValidationError::single(
                        column_name,
                        self.name(),
                        failed,
                        "list must not be empty",
                    )]
                }
            }
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "a List or Array",
            )],
        }
    }
}

/// Checks struct subfields with regular single-column checks.
/// The struct column `column.field` is extracted, then each inner check runs
/// against that sub-series. Failures are reported against the parent column.
pub struct StructFieldCheck {
    pub field: String,
    pub inner_checks: Vec<Box<dyn Check>>,
}

impl Check for StructFieldCheck {
    fn name(&self) -> &str {
        "struct_field"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::StructField(
            self.field.clone(),
            self.inner_checks.iter().map(|c| c.spec()).collect(),
        )
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let struct_ca = match series.struct_() {
            Ok(ca) => ca,
            Err(_) => {
                return vec![type_mismatch_error(column_name, self.name(), "a Struct")];
            }
        };
        let field_series = match struct_ca.field_by_name(self.field.as_str()) {
            Ok(s) => s,
            Err(_) => {
                return vec![ValidationError::single(
                    column_name,
                    self.name(),
                    vec![],
                    format!(
                        "Struct column '{}' is missing field '{}'",
                        column_name, self.field
                    ),
                )];
            }
        };
        let mut errors = Vec::new();
        for check in &self.inner_checks {
            for mut error in check.validate(column_name, &field_series) {
                error.message = format!("Field '{}': {}", self.field, error.message);
                errors.push(error);
            }
        }
        errors
    }
}

/// Row lengths for List and Array columns; `None` marks a null row.
fn nested_lengths(series: &Series) -> PolarsResult<Vec<Option<usize>>> {
    match series.dtype() {
        DataType::List(_) => {
            let ca = series.list()?;
            Ok(ca.iter().map(|opt| opt.map(|s| s.len())).collect())
        }
        DataType::Array(_, _) => {
            let ca = series.array()?;
            Ok(ca.iter().map(|opt| opt.map(|s| s.len())).collect())
        }
        _ => Err(PolarsError::ComputeError("expected List or Array".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unique_values_check_reports_duplicates() {
        let series = Series::new("value".into(), [1i32, 2, 2, 3]);
        let check = UniqueValues;

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1, 2]);
    }

    #[test]
    fn greater_than_supports_integer_and_float_columns() {
        let integer = Series::new("value".into(), [1i32, 2, 3]);
        let float = Series::new("value".into(), [1.0f32, 2.0, 3.0]);
        let check = GreaterThanF64 { min_value: 1.5 };

        assert_eq!(check.validate("value", &integer)[0].rows, vec![0]);
        assert_eq!(check.validate("value", &float)[0].rows, vec![0]);
    }

    #[test]
    fn numeric_check_rejects_string_columns() {
        let series = Series::new("value".into(), ["1", "2"]);
        let check = GreaterThanF64 { min_value: 0.0 };

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("column is not a numeric type"));
    }

    #[test]
    fn wide_integers_compare_exactly_above_2_to_53() {
        // 2^53 + 1 is not representable as f64; the old float path merged it
        // with 2^53 and reported the wrong row.
        let two_to_53: u64 = 1 << 53;
        let series = Series::new("value".into(), [two_to_53, two_to_53 + 1]);
        let check = GreaterThanF64 {
            min_value: two_to_53 as f64,
        };

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0]);
    }

    #[test]
    fn i128_extremes_compare_exactly() {
        let series = Series::new("value".into(), [i128::MIN, 0, i128::MAX]);
        let check = LessThanF64 { max_value: 0.0 };

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1, 2]);
    }

    #[test]
    fn u128_saturating_thresholds_do_not_panic() {
        let series = Series::new("value".into(), [u128::MAX - 1, u128::MAX]);
        // (u128::MAX - 1) as f64 rounds up to 2^128, which saturates back to
        // u128::MAX, so every value fails the strict greater-than check.
        let check = GreaterThanF64 {
            min_value: (u128::MAX - 1) as f64,
        };

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn non_finite_thresholds_keep_float_semantics() {
        let series = Series::new("value".into(), [1i64, 2, 3]);
        let gt_inf = GreaterThanF64 {
            min_value: f64::INFINITY,
        };
        let gt_neg_inf = GreaterThanF64 {
            min_value: f64::NEG_INFINITY,
        };

        assert_eq!(gt_inf.validate("value", &series)[0].rows, vec![0, 1, 2]);
        assert!(gt_neg_inf.validate("value", &series).is_empty());
    }

    #[test]
    fn lte_at_domain_minimum_passes_the_boundary_value() {
        // lte(0) on unsigned: 0 itself satisfies x <= 0, only larger fail.
        let series = Series::new("value".into(), [0u128, 1]);
        let check = LessThanOrEqualF64 { max_value: 0.0 };

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
    }

    #[test]
    fn unsigned_negative_thresholds_resolve_trivially() {
        // No u128 is <= -0.5, so gt(-0.5) passes everything...
        let series = Series::new("value".into(), [0u128, 1]);
        let pass = GreaterThanF64 { min_value: -0.5 };
        assert!(pass.validate("value", &series).is_empty());

        // ...while lte(-0.5) fails everything (0 is not <= -0.5 either).
        let fail = LessThanOrEqualF64 { max_value: -0.5 };
        let errors = fail.validate("value", &series);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn signed_out_of_range_thresholds_resolve_trivially() {
        let series = Series::new("value".into(), [i64::MIN, 0, i64::MAX]);
        // Nothing exceeds 2^127.
        let gt_huge = GreaterThanF64 {
            min_value: 170141183460469231731687303715884105728.0,
        };
        assert_eq!(gt_huge.validate("value", &series)[0].rows, vec![0, 1, 2]);
        // Every i64 is >= -2^127, so lt(-2^127) fails everything.
        let lt_huge_neg = LessThanF64 {
            max_value: -170141183460469231731687303715884105728.0,
        };
        assert_eq!(
            lt_huge_neg.validate("value", &series)[0].rows,
            vec![0, 1, 2]
        );
    }

    #[test]
    fn between_with_empty_integer_intersection_fails_all_rows() {
        // No integer lies in (0.1, 0.9): every value must fail.
        let series = Series::new("value".into(), [0i64, 1]);
        let check = BetweenF64 {
            min_value: 0.1,
            max_value: 0.9,
        };

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn list_len_between_reports_out_of_range_rows() {
        let inner_a = Series::new("".into(), [1i64, 2]);
        let inner_b = Series::new("".into(), [1i64]);
        let inner_c = Series::new("".into(), [0i64, 0, 0, 0]);
        let series = Series::new("tags".into(), [inner_a, inner_b, inner_c]);
        assert!(matches!(series.dtype(), DataType::List(_)));

        let check = ListLenBetween {
            min_len: 1,
            max_len: 2,
        };

        let errors = check.validate("tags", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![2]);
    }

    #[test]
    fn list_len_rejects_scalar_columns() {
        let series = Series::new("value".into(), [1i64, 2]);
        let check = ListNotEmpty;

        let errors = check.validate("value", &series);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("a List or Array"));
    }

    #[test]
    fn struct_field_check_validates_subfield() {
        let city = Series::new("city".into(), ["Paris", "X"]);
        let zip = Series::new("zip".into(), ["75001", "bad"]);
        let df = DataFrame::new(2, vec![city.into(), zip.into()]).unwrap();
        let address = df.into_struct("address".into()).into_series();
        let check = StructFieldCheck {
            field: "zip".to_string(),
            inner_checks: vec![
                check_from_spec(&CheckSpec::Matches(r"^\d{5}$".to_string())).unwrap()
            ],
        };

        let errors = check.validate("address", &address);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
        assert!(errors[0].message.contains("zip"));
    }

    #[test]
    fn str_length_between_reports_short_and_long_rows() {
        let series = Series::new(
            "code".into(),
            [Some("abc"), Some("a"), Some("toolongvalue"), None],
        );
        let check = StrLengthBetween {
            min_len: 2,
            max_len: 5,
        };

        let errors = check.validate("code", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1, 2]);
    }

    fn string_series(values: &[&str]) -> Series {
        Series::new("s".into(), values.to_vec())
    }

    fn collect_rows(check: &dyn Check, series: &Series) -> Vec<usize> {
        let errors = check.validate("s", series);
        assert_eq!(errors.len(), 1);
        errors[0].rows.clone()
    }

    #[test]
    fn not_blank_flags_empty_and_whitespace_only() {
        let series = string_series(&["ok", "", "   ", "\t\n"]);
        let rows = collect_rows(&NotBlank, &series);
        assert_eq!(rows, vec![1, 2, 3]);
    }

    #[test]
    fn not_blank_skips_nulls() {
        let series = Series::new("s".into(), [Some("a"), None, Some("  ")]);
        let rows = collect_rows(&NotBlank, &series);
        assert_eq!(rows, vec![2]);
    }

    #[test]
    fn not_blank_rejects_non_string_columns() {
        let series = Series::new("s".into(), [1i64, 2]);
        let errors = NotBlank.validate("s", &series);
        assert!(errors[0].message.contains("column is not a String"));
    }

    #[test]
    fn starts_with_and_ends_with_report_prefix_suffix_rows() {
        let series = string_series(&["tag-1", "other", "tag-2"]);
        let prefix = StartsWith {
            prefix: "tag-".to_string(),
        };
        let suffix = EndsWith {
            suffix: "-1".to_string(),
        };

        assert_eq!(collect_rows(&prefix, &series), vec![1]);
        assert_eq!(collect_rows(&suffix, &series), vec![1, 2]);
    }

    #[test]
    fn str_length_between_counts_characters_not_bytes() {
        let series = string_series(&["ééé", "aaaa"]);
        // é is one character but two bytes; a 3-char minimum passes despite
        // a 6-byte length.
        let check = StrLengthBetween {
            min_len: 3,
            max_len: 4,
        };
        let errors = check.validate("s", &series);
        assert!(errors.is_empty());
    }

    #[test]
    fn one_of_numeric_matches_integers_exactly() {
        // 2^54 is representable in f64, but 2^54 - 1 is not: it rounds UP to
        // 2^54. A float-equality implementation would wrongly accept the
        // row; the integer path reports it.
        let base: i64 = 1 << 54;
        let series = Series::new("n".into(), [base - 1, base]);
        let check = OneOfNumericValues {
            allowed_values: vec![base as f64],
        };

        let errors = check.validate("n", &series);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0]);
    }

    #[test]
    fn one_of_numeric_supports_floats_bools_and_nulls() {
        let floats = Series::new("f".into(), [0.5f64, 0.7, 1.5]);
        let bools = Series::new("b".into(), [true, false, true]);
        let nullable = Series::new("n".into(), [Some(2i32), None, Some(4), Some(7)]);

        let float_check = OneOfNumericValues {
            allowed_values: vec![0.5, 1.5],
        };
        let bool_check = OneOfNumericValues {
            allowed_values: vec![1.0],
        };
        let int_check = OneOfNumericValues {
            allowed_values: vec![2.0, 4.0],
        };

        assert_eq!(collect_rows(&float_check, &floats), vec![1]);
        assert_eq!(collect_rows(&bool_check, &bools), vec![1]);
        assert_eq!(collect_rows(&int_check, &nullable), vec![3]);
    }

    #[test]
    fn one_of_numeric_rejects_string_columns() {
        let series = string_series(&["1", "2"]);
        let check = OneOfNumericValues {
            allowed_values: vec![1.0],
        };

        let errors = check.validate("s", &series);

        assert!(errors[0].message.contains("numeric, boolean"));
    }

    #[test]
    fn one_of_numeric_empty_allowed_set_fails_every_value() {
        let series = Series::new("n".into(), [1i64, 2]);
        let check = OneOfNumericValues {
            allowed_values: vec![],
        };

        let errors = check.validate("n", &series);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn one_of_numeric_values_are_exactly_representable_only() {
        // 2^127 is outside the i128 domain, so no integer row can equal it;
        // it must not be rounded into the member set.
        let allowed = vec![170141183460469231731687303715884105728.0];
        let series = Series::new("n".into(), [i128::MAX]);
        let check = OneOfNumericValues {
            allowed_values: allowed,
        };

        let errors = check.validate("n", &series);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0]);
    }
}
