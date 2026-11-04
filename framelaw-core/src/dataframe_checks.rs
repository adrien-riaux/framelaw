use polars::prelude::*;

use std::collections::{HashMap, HashSet};

use crate::checks::{get_false_indices, get_true_indices, int_domain, numeric_series, IntDomain};
use crate::errors::ValidationError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CompareOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Eq => "==",
            Self::Ne => "!=",
        }
    }
}

#[derive(Clone, Debug)]
pub enum DataFrameCheckSpec {
    ColumnComparison {
        left: String,
        op: CompareOp,
        right: String,
    },
    Ordered {
        columns: Vec<String>,
        strict: bool,
    },
    SumColumns {
        addends: Vec<String>,
        target: String,
        tolerance: Option<f64>,
    },
    Implies {
        left: String,
        by: String,
    },
    Determines {
        a: String,
        b: String,
    },
    Within {
        left: String,
        right: String,
        max_gap_ns: i64,
    },
    UniqueTogether {
        columns: Vec<String>,
    },
}

pub trait DataFrameCheck: Send + Sync {
    fn name(&self) -> &str;
    fn spec(&self) -> DataFrameCheckSpec;
    fn validate(&self, df: &DataFrame) -> Vec<ValidationError>;
}

fn missing_column_error(requested: &[String]) -> ValidationError {
    ValidationError::multi(
        requested.to_vec(),
        "column_exists",
        vec![],
        "column is missing from the DataFrame",
    )
}

/// Whether a dtype stores temporal ticks (whose raw magnitude depends on the
/// unit, so mixed temporal dtypes must never be compared directly).
pub(crate) fn is_temporal_dtype(dtype: &DataType) -> bool {
    matches!(
        dtype,
        DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time
    )
}

/// Extracts signed values exactly (`i128` domain; temporal via `Int64`),
/// as an owned chunked array rather than a materialized `Option<i128>`
/// vector: the widening costs one cheap column cast, not one heap cell per
/// row.
pub(crate) fn int128_ca(series: &Series) -> PolarsResult<Int128Chunked> {
    let base = match series.dtype() {
        DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            series.cast(&DataType::Int64)?
        }
        _ => series.clone(),
    };
    Ok(base.cast(&DataType::Int128)?.i128()?.clone())
}

/// Extracts unsigned values exactly (`u128` domain) as a chunked array.
pub(crate) fn uint128_ca(series: &Series) -> PolarsResult<UInt128Chunked> {
    Ok(series.cast(&DataType::UInt128)?.u128()?.clone())
}

/// Materialized signed cells; the sampling generator works value by value,
/// where the single-cell read stays natural.
pub(crate) fn int128_cells(series: &Series) -> PolarsResult<Vec<Option<i128>>> {
    Ok(int128_ca(series)?.iter().collect())
}

/// Materialized unsigned cells; see `int128_cells`.
pub(crate) fn uint128_cells(series: &Series) -> PolarsResult<Vec<Option<u128>>> {
    Ok(uint128_ca(series)?.iter().collect())
}

/// Total order between a signed and an unsigned value with no rounding:
/// negatives sort below everything, values above `i128::MAX` above everything.
pub(crate) fn cmp_mixed_int_uint(signed: i128, unsigned: u128) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if signed < 0 || unsigned > i128::MAX as u128 {
        Ordering::Less
    } else {
        (signed as u128).cmp(&unsigned)
    }
}

/// Chunked exact-value extraction for a mixed signed/unsigned comparison.
type MixedPairValues = (Int128Chunked, UInt128Chunked);

/// Declarative comparison between two columns, e.g. `start < end`.
/// Nulls on either side skip the row; use nullable=false for strictness.
pub struct ColumnComparison {
    pub left: String,
    pub op: CompareOp,
    pub right: String,
}

impl ColumnComparison {
    /// Exact integer-pair comparison in the widened `i128`/`u128` domain.
    /// Temporal columns convert through `Int64` first. Pairs of native
    /// `Int8`–`Int64` columns (and identical-dtype temporals) compare
    /// through `i64` directly: the values lose nothing in the conversion,
    /// so results are identical while using the CPU's native width. Nulls
    /// propagate as null and are skipped by the caller, mirroring the
    /// float path.
    fn validate_int_pair(
        &self,
        left: &Series,
        right: &Series,
        unsigned: bool,
    ) -> Vec<ValidationError> {
        /// Any signed int dtype within `i64`, or a temporal dtype (whose
        /// ticks are `i64` by construction; mixed temporal units were
        /// rejected by the caller already).
        fn is_narrow(series: &Series) -> bool {
            matches!(
                series.dtype(),
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            ) || is_temporal_dtype(series.dtype())
        }
        /// Lossless widening of a narrow column to a shared `i64` domain.
        fn narrow_int64(series: &Series) -> PolarsResult<Int64Chunked> {
            Ok(series.cast(&DataType::Int64)?.i64()?.clone())
        }
        let mask: PolarsResult<BooleanChunked> = (|| {
            if unsigned {
                let l = left.cast(&DataType::UInt128)?.u128()?.clone();
                let r = right.cast(&DataType::UInt128)?.u128()?.clone();
                Ok(match self.op {
                    CompareOp::Lt => l.lt(&r),
                    CompareOp::Le => l.lt_eq(&r),
                    CompareOp::Gt => l.gt(&r),
                    CompareOp::Ge => l.gt_eq(&r),
                    CompareOp::Eq => l.equal(&r),
                    CompareOp::Ne => l.not_equal(&r),
                })
            } else if is_narrow(left) && is_narrow(right) {
                let l = narrow_int64(left)?;
                let r = narrow_int64(right)?;
                Ok(match self.op {
                    CompareOp::Lt => l.lt(&r),
                    CompareOp::Le => l.lt_eq(&r),
                    CompareOp::Gt => l.gt(&r),
                    CompareOp::Ge => l.gt_eq(&r),
                    CompareOp::Eq => l.equal(&r),
                    CompareOp::Ne => l.not_equal(&r),
                })
            } else {
                let l = left.cast(&DataType::Int128)?.i128()?.clone();
                let r = right.cast(&DataType::Int128)?.i128()?.clone();
                Ok(match self.op {
                    CompareOp::Lt => l.lt(&r),
                    CompareOp::Le => l.lt_eq(&r),
                    CompareOp::Gt => l.gt(&r),
                    CompareOp::Ge => l.gt_eq(&r),
                    CompareOp::Eq => l.equal(&r),
                    CompareOp::Ne => l.not_equal(&r),
                })
            }
        })();
        match mask {
            Ok(mask) => {
                // Rows where the comparison does not hold fail; nulls stay
                // unset and are skipped, mirroring the float path.
                let failed: Vec<usize> = get_false_indices(&mask);
                comparison_result(self, failed)
            }
            Err(error) => vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "Failed to compare '{}' and '{}': {error}",
                    self.left, self.right
                ),
            )],
        }
    }

    /// Exact comparison for mixed signed/unsigned pairs, which share no
    /// lossless primitive domain. `left_signed` records which side holds
    /// the signed values so the operator direction stays correct.
    fn validate_mixed_pair(
        &self,
        left: &Series,
        right: &Series,
        left_signed: bool,
    ) -> Vec<ValidationError> {
        use std::cmp::Ordering;
        let extracted: PolarsResult<MixedPairValues> = (|| {
            if left_signed {
                Ok((int128_ca(left)?, uint128_ca(right)?))
            } else {
                Ok((int128_ca(right)?, uint128_ca(left)?))
            }
        })();
        match extracted {
            Ok((signed_vals, unsigned_vals)) => {
                let failed: Vec<usize> = signed_vals
                    .iter()
                    .zip(unsigned_vals.iter())
                    .enumerate()
                    .filter_map(|(index, pair)| match pair {
                        (Some(a), Some(b)) => {
                            let order = cmp_mixed_int_uint(a, b);
                            // `order` is signed-vs-unsigned; flip it when the
                            // unsigned side is on the left.
                            let order = if left_signed { order } else { order.reverse() };
                            let holds = match self.op {
                                CompareOp::Lt => order == Ordering::Less,
                                CompareOp::Le => order != Ordering::Greater,
                                CompareOp::Gt => order == Ordering::Greater,
                                CompareOp::Ge => order != Ordering::Less,
                                CompareOp::Eq => order == Ordering::Equal,
                                CompareOp::Ne => order != Ordering::Equal,
                            };
                            if holds {
                                None
                            } else {
                                Some(index)
                            }
                        }
                        _ => None,
                    })
                    .collect();
                comparison_result(self, failed)
            }
            Err(error) => vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "Failed to compare '{}' and '{}': {error}",
                    self.left, self.right
                ),
            )],
        }
    }
}

impl DataFrameCheck for ColumnComparison {
    fn name(&self) -> &str {
        match self.op {
            CompareOp::Lt => "lt_column",
            CompareOp::Le => "lte_column",
            CompareOp::Gt => "gt_column",
            CompareOp::Ge => "gte_column",
            CompareOp::Eq => "eq_columns",
            CompareOp::Ne => "ne_columns",
        }
    }
    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::ColumnComparison {
            left: self.left.clone(),
            op: self.op,
            right: self.right.clone(),
        }
    }
    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        // Borrows only: no column materialization happens on this path.
        let left = match df.column(&self.left) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.left))],
        };
        let right = match df.column(&self.right) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.right))],
        };

        // String comparison path.
        if let (Ok(left_str), Ok(right_str)) = (left.str(), right.str()) {
            let mut failed = Vec::new();
            for (i, (l, r)) in left_str.iter().zip(right_str.iter()).enumerate() {
                let (Some(l), Some(r)) = (l, r) else { continue };
                let ok = match self.op {
                    CompareOp::Lt => l < r,
                    CompareOp::Le => l <= r,
                    CompareOp::Gt => l > r,
                    CompareOp::Ge => l >= r,
                    CompareOp::Eq => l == r,
                    CompareOp::Ne => l != r,
                };
                if !ok {
                    failed.push(i);
                }
            }
            return comparison_result(self, failed);
        }

        // Temporal columns carry their unit in the raw ticks (days vs
        // microseconds vs ...), so mixed temporal dtypes would compare
        // unrelated magnitudes. Require identical temporal dtypes.
        if is_temporal_dtype(left.dtype())
            && is_temporal_dtype(right.dtype())
            && left.dtype() != right.dtype()
        {
            return vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "Cannot compare '{}' ({}) and '{}' ({}): temporal columns must share a dtype",
                    self.left,
                    left.dtype(),
                    self.right,
                    right.dtype(),
                ),
            )];
        }

        // Numeric comparison path (ints, floats, dates, datetimes).
        // Integer pairs compare exactly in the i128/u128 domain so wide
        // values above 2^53 keep their distinctness; any pair involving a
        // float keeps the historical f64 semantics.
        if int_domain(left.dtype()) == Some(IntDomain::I128)
            && int_domain(right.dtype()) == Some(IntDomain::I128)
        {
            return self.validate_int_pair(left, right, false);
        }
        if int_domain(left.dtype()) == Some(IntDomain::U128)
            && int_domain(right.dtype()) == Some(IntDomain::U128)
        {
            return self.validate_int_pair(left, right, true);
        }
        // Mixed signed/unsigned pairs share no lossless primitive domain;
        // compare exactly value by value instead of rounding through f64.
        if int_domain(left.dtype()) == Some(IntDomain::I128)
            && int_domain(right.dtype()) == Some(IntDomain::U128)
        {
            return self.validate_mixed_pair(left, right, true);
        }
        if int_domain(left.dtype()) == Some(IntDomain::U128)
            && int_domain(right.dtype()) == Some(IntDomain::I128)
        {
            return self.validate_mixed_pair(left, right, false);
        }
        let (Ok(left_num), Ok(right_num)) = (numeric_series(left), numeric_series(right)) else {
            return vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "Cannot compare '{}' and '{}': need two numeric or two string columns",
                    self.left, self.right
                ),
            )];
        };
        // Vectorized comparison in Polars' native float domain; nulls on
        // either side propagate to a null mask cell and are skipped by the
        // index extraction. Polars' kernels use total equality/order (NaN
        // == NaN), but the historical scalar semantics were Rust's `f64`
        // ops (any NaN involvement fails N lt/le/gt/ge/eq comparisons and
        // passes `ne`), so NaN rows are re-mapped afterwards to keep every
        // prior verdict.
        let nan_involved = (&left_num.is_nan()) | (&right_num.is_nan());
        let compare_mask = match self.op {
            CompareOp::Lt => left_num.lt(&right_num),
            CompareOp::Le => left_num.lt_eq(&right_num),
            CompareOp::Gt => left_num.gt(&right_num),
            CompareOp::Ge => left_num.gt_eq(&right_num),
            // Exact float equality is intentional: cross-column checks
            // assert identical stored values, not approximate math.
            CompareOp::Eq => left_num.equal(&right_num),
            CompareOp::Ne => left_num.not_equal(&right_num),
        };
        let fail_mask = match self.op {
            CompareOp::Ne => (&!compare_mask) & &!nan_involved,
            _ => (&!compare_mask) | &nan_involved,
        };
        comparison_result(self, get_true_indices(&fail_mask))
    }
}

fn comparison_result(check: &ColumnComparison, failed: Vec<usize>) -> Vec<ValidationError> {
    if failed.is_empty() {
        Vec::new()
    } else {
        vec![ValidationError::multi(
            vec![check.left.clone(), check.right.clone()],
            check.name(),
            failed.clone(),
            format!(
                "condition '{} {} {}' failed for {} row(s)",
                check.left,
                check.op.as_str(),
                check.right,
                failed.len(),
            ),
        )]
    }
}

/// Checks that columns form a per-row chain `c1 <= c2 <= ... <= cn`
/// (`<` everywhere when `strict`). All columns must be strings or share
/// numeric comparability: identical temporal dtypes are required between
/// temporal neighbors, integers compare exactly in the widened
/// `i128`/`u128` domain, and any float column downgrades comparisons to
/// `f64`. Nulls on either side of a pair skip that pair.
pub struct Ordered {
    pub columns: Vec<String>,
    pub strict: bool,
}

/// The per-column value pool used by the chain comparison. Owned chunked
/// arrays: widening (ints to `i128`, floats to `f64`) is a cheap column
/// cast, so no per-row cell vector is materialized, and text pools borrow
/// the original string data through the chunked instead of cloning every
/// cell into a heap `String`.
enum ChainVals {
    Signed(Int128Chunked),
    Unsigned(UInt128Chunked),
    Float(Float64Chunked),
    Text(StringChunked),
}

/// Classified pool of one column, choosing the widest exact domain with
/// the same fallback rules as `ColumnComparison` (floats downgrade the
/// whole comparison; text is uncomparable against numbers).
fn classify_chain_column(name: &str, series: &Series) -> PolarsResult<ChainVals> {
    if series.dtype() == &DataType::String {
        let text = series
            .str()
            .map_err(|_| PolarsError::ComputeError("expected a string column".into()))?;
        return Ok(ChainVals::Text(text.clone()));
    }
    if int_domain(series.dtype()) == Some(IntDomain::I128) {
        return Ok(ChainVals::Signed(int128_ca(series)?));
    }
    if int_domain(series.dtype()) == Some(IntDomain::U128) {
        return Ok(ChainVals::Unsigned(uint128_ca(series)?));
    }
    let floats = numeric_series(series).map_err(|_| {
        PolarsError::ComputeError(format!("Column '{name}' is not string or numeric").into())
    })?;
    Ok(ChainVals::Float(floats))
}

/// Row fails the pair when the chain relation does not hold: strict chains
/// require `a < b` everywhere, non-strict `a <= b`. The negation form is
/// deliberate: it keeps non-comparable values (NaN) failing exactly like
/// the old cell loop.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn chain_pair_fails<T: PartialOrd>(strict: bool, a: T, b: T) -> bool {
    if strict {
        !(a < b)
    } else {
        !(a <= b)
    }
}

/// Builds a Boolean mask over paired present values; `None` on either side
/// (null or unmatched length) skips the row.
fn pair_mask<T, U, F, I, J>(left: I, right: J, holds: F) -> BooleanChunked
where
    I: IntoIterator<Item = Option<T>>,
    J: IntoIterator<Item = Option<U>>,
    F: Fn(T, U) -> bool,
{
    IntoIterator::into_iter(left)
        .zip(IntoIterator::into_iter(right))
        .map(|(a, b)| match (a, b) {
            (Some(a), Some(b)) => Some(holds(a, b)),
            _ => None,
        })
        .collect::<BooleanChunked>()
}

/// Exact total-order comparison between signed and unsigned cell values.
fn mixed_pair_holds(strict: bool, signed: i128, unsigned: u128, unsigned_on_left: bool) -> bool {
    use std::cmp::Ordering;
    let order = cmp_mixed_int_uint(signed, unsigned);
    let order = if unsigned_on_left {
        order.reverse()
    } else {
        order
    };
    if strict {
        order == Ordering::Less
    } else {
        order != Ordering::Greater
    }
}

/// Failing mask of one adjacent pair in the ordered chain. `Err` marks an
/// unsupported domain pair (the caller already rejects text/numeric
/// mixing, so this is defensive).
fn pair_fail_mask(strict: bool, left: &ChainVals, right: &ChainVals) -> Result<BooleanChunked, ()> {
    use ChainVals::*;
    match (left, right) {
        (Signed(l), Signed(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a, b)
        })),
        (Unsigned(l), Unsigned(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a, b)
        })),
        (Float(l), Float(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a, b)
        })),
        (Text(l), Text(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a, b)
        })),
        (Signed(l), Unsigned(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            !mixed_pair_holds(strict, a, b, false)
        })),
        (Unsigned(l), Signed(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            !mixed_pair_holds(strict, b, a, true)
        })),
        (Float(l), Signed(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a, b as f64)
        })),
        (Signed(l), Float(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a as f64, b)
        })),
        (Float(l), Unsigned(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a, b as f64)
        })),
        (Unsigned(l), Float(r)) => Ok(pair_mask(l.iter(), r.iter(), |a, b| {
            chain_pair_fails(strict, a as f64, b)
        })),
        _ => Err(()),
    }
}

impl DataFrameCheck for Ordered {
    fn name(&self) -> &str {
        "ordered"
    }

    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::Ordered {
            columns: self.columns.clone(),
            strict: self.strict,
        }
    }

    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        if self.columns.len() < 2 {
            return vec![ValidationError::multi(
                self.columns.clone(),
                self.name(),
                vec![],
                "Ordered needs at least two columns".to_string(),
            )];
        }
        let mut pools = Vec::with_capacity(self.columns.len());
        let mut missing: Vec<String> = Vec::new();
        for name in &self.columns {
            match df.column(name) {
                Ok(column) => match classify_chain_column(name, column.as_materialized_series()) {
                    Ok(pool) => pools.push(pool),
                    Err(_) => {
                        return vec![ValidationError::multi(
                            self.columns.clone(),
                            self.name(),
                            vec![],
                            format!("Cannot order '{name}': columns must be strings or numeric"),
                        )]
                    }
                },
                Err(_) => missing.push(name.clone()),
            }
        }
        if !missing.is_empty() {
            return vec![missing_column_error(&missing)];
        }
        // Text columns can never be ordered against numeric ones, so the
        // chain kind must not mix string and numeric neighbors.
        let text_neighbor = pools.windows(2).any(|pair| {
            matches!(pair, [ChainVals::Text(_), _] | [_, ChainVals::Text(_)])
                && !matches!(pair, [ChainVals::Text(_), ChainVals::Text(_)])
        });
        if text_neighbor {
            return vec![ValidationError::multi(
                self.columns.clone(),
                self.name(),
                vec![],
                "Cannot order: string and numeric columns cannot be mixed".to_string(),
            )];
        }
        // Temporal neighbors with different dtypes carry raw ticks from
        // different units, same rule as ColumnComparison.
        let dtypes: Vec<DataType> = self
            .columns
            .iter()
            .map(|name| df.column(name).expect("checked above").dtype().clone())
            .collect();
        for pair in dtypes.windows(2) {
            if is_temporal_dtype(&pair[0]) && is_temporal_dtype(&pair[1]) && pair[0] != pair[1] {
                return vec![ValidationError::multi(
                    self.columns.clone(),
                    self.name(),
                    vec![],
                    format!(
                        "Cannot order {}: temporal columns must share a dtype",
                        self.columns.join(", ")
                    ),
                )];
            }
        }
        let symbols = self
            .columns
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>()
            .join(if self.strict { " < " } else { " <= " });
        // One failing mask per adjacent pair, merged while walking the
        // chain: a row fails when any adjacent pair is violated. This
        // replaces the per-row window loop with per-pair passes over the
        // data (better locality, no per-cell materialization).
        let mut failed_mask: Option<BooleanChunked> = None;
        for pair in pools.windows(2) {
            let partial = match pair_fail_mask(self.strict, &pair[0], &pair[1]) {
                Ok(mask) => mask,
                Err(()) => {
                    return vec![ValidationError::multi(
                        self.columns.clone(),
                        self.name(),
                        vec![],
                        format!(
                            "Cannot order {}: unsupported column types",
                            self.columns.join(", ")
                        ),
                    )]
                }
            };
            failed_mask = Some(match failed_mask {
                None => partial,
                Some(merged) => &merged | &partial,
            });
        }
        let Some(failed_mask) = failed_mask else {
            // A single-column chain has no pair to violate.
            return Vec::new();
        };
        let failed = get_true_indices(&failed_mask);
        if failed.is_empty() {
            Vec::new()
        } else {
            vec![ValidationError::multi(
                self.columns.clone(),
                self.name(),
                failed.clone(),
                format!("ordering '{}' failed for {} row(s)", symbols, failed.len()),
            )]
        }
    }
}

/// Checks that `target` equals the sum of `addends` per row
/// (`c = a + b + ...`). All-integer schemas compare exactly in the widened
/// integer domain; any float column downgrades to `f64` where `tolerance`
/// (default: exact equality) bounds the allowed absolute difference.
/// Nulls on any operand skip the row.
pub struct SumColumns {
    pub addends: Vec<String>,
    pub target: String,
    pub tolerance: Option<f64>,
}

impl DataFrameCheck for SumColumns {
    fn name(&self) -> &str {
        "sum_columns"
    }

    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::SumColumns {
            addends: self.addends.clone(),
            target: self.target.clone(),
            tolerance: self.tolerance,
        }
    }

    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let mut columns = self.addends.clone();
        columns.push(self.target.clone());
        let mut pools = Vec::with_capacity(columns.len());
        for name in &columns {
            let series = match df.column(name) {
                Ok(column) => column.as_materialized_series(),
                Err(_) => return vec![missing_column_error(std::slice::from_ref(name))],
            };
            if series.dtype() == &DataType::String {
                return vec![ValidationError::multi(
                    columns.clone(),
                    self.name(),
                    vec![],
                    format!("Cannot sum '{name}': need numeric columns"),
                )];
            }
            let pool = if int_domain(series.dtype()) == Some(IntDomain::I128) {
                match int128_ca(series) {
                    Ok(values) => ChainVals::Signed(values),
                    Err(error) => {
                        return vec![ValidationError::multi(
                            columns.clone(),
                            self.name(),
                            vec![],
                            format!("Failed to extract values from '{name}': {error}"),
                        )]
                    }
                }
            } else if int_domain(series.dtype()) == Some(IntDomain::U128) {
                match uint128_ca(series) {
                    Ok(values) => ChainVals::Unsigned(values),
                    Err(error) => {
                        return vec![ValidationError::multi(
                            columns.clone(),
                            self.name(),
                            vec![],
                            format!("Failed to extract values from '{name}': {error}"),
                        )]
                    }
                }
            } else {
                match numeric_series(series) {
                    Ok(values) => ChainVals::Float(values),
                    Err(_) => {
                        return vec![ValidationError::multi(
                            columns.clone(),
                            self.name(),
                            vec![],
                            format!("Cannot sum '{name}': need numeric columns"),
                        )]
                    }
                }
            };
            pools.push(pool);
        }
        let target_idx = pools.len() - 1;
        let addend_count = self.addends.len();
        let tolerance = self.tolerance.unwrap_or(0.0);
        // The widest applicable sum domain depends only on the columns,
        // not on the row: resolve it once instead of per row.
        let sum_kind = sum_kind(&pools[..addend_count]);
        let height = df.height();
        let mut failed = Vec::new();
        for row in 0..height {
            let sum = match sum_cells(&pools[..addend_count], sum_kind, row) {
                SumStatus::Skip => continue,
                SumStatus::Overflow => {
                    failed.push(row);
                    continue;
                }
                SumStatus::Value(sum) => sum,
            };
            let holds = match target_cell(&pools[target_idx], row) {
                Some(target) => sum_equals_target(sum, target, tolerance),
                None => true, // a null target skips the row
            };
            if !holds {
                failed.push(row);
            }
        }
        if failed.is_empty() {
            Vec::new()
        } else {
            vec![ValidationError::multi(
                columns,
                self.name(),
                failed.clone(),
                format!(
                    "sum '{}' must equal '{}' failed for {} row(s)",
                    self.addends
                        .iter()
                        .map(|name| name.as_str())
                        .collect::<Vec<_>>()
                        .join(" + "),
                    self.target,
                    failed.len(),
                ),
            )]
        }
    }
}

/// Sum of the target row's operands in the widest applicable domain.
enum SumResult {
    Signed(i128),
    Unsigned(u128),
    Float(f64),
}

enum SumStatus {
    /// A null operand skips the row.
    Skip,
    /// The checked arithmetic overflowed; no sum can hold, so the row fails.
    Overflow,
    Value(SumResult),
}

/// The widest applicable sum domain, resolved once per check instead of
/// once per row.
enum SumKind {
    UnsignedOnly,
    FloatDowngrade,
    SignedOnly,
}

impl Copy for SumKind {}

impl Clone for SumKind {
    fn clone(&self) -> Self {
        *self
    }
}

fn sum_kind(pools: &[ChainVals]) -> SumKind {
    let all_unsigned = !pools.is_empty()
        && pools
            .iter()
            .all(|pool| matches!(pool, ChainVals::Unsigned(_)));
    if all_unsigned {
        return SumKind::UnsignedOnly;
    }
    let any_float = pools.iter().any(|pool| matches!(pool, ChainVals::Float(_)));
    let mixed_int = pools
        .iter()
        .any(|pool| matches!(pool, ChainVals::Unsigned(_)))
        && pools
            .iter()
            .any(|pool| matches!(pool, ChainVals::Signed(_)));
    if any_float || mixed_int {
        SumKind::FloatDowngrade
    } else {
        SumKind::SignedOnly
    }
}

/// Sums the operands of one row when they are all present, keeping
/// integers exact with `i128`/`u128` checked arithmetic and defaulting to
/// `f64` whenever a float column participates (or the two integer domains
/// mix, mirroring the `ColumnComparison` fallback).
fn sum_cells(pools: &[ChainVals], kind: SumKind, row: usize) -> SumStatus {
    match kind {
        SumKind::UnsignedOnly => {
            let mut total: u128 = 0;
            for pool in pools {
                let ChainVals::Unsigned(cell) = pool else {
                    unreachable!("sum_kind checked above");
                };
                let Some(value) = cell.get(row) else {
                    return SumStatus::Skip;
                };
                let Some(next) = total.checked_add(value) else {
                    return SumStatus::Overflow;
                };
                total = next;
            }
            SumStatus::Value(SumResult::Unsigned(total))
        }
        SumKind::FloatDowngrade => {
            let mut total = 0.0f64;
            for pool in pools {
                let cell = match pool {
                    ChainVals::Float(cell) => cell.get(row),
                    ChainVals::Signed(cell) => cell.get(row).map(|v| v as f64),
                    ChainVals::Unsigned(cell) => cell.get(row).map(|v| v as f64),
                    ChainVals::Text(_) => return SumStatus::Overflow,
                };
                let Some(cell) = cell else {
                    return SumStatus::Skip;
                };
                total += cell;
            }
            SumStatus::Value(SumResult::Float(total))
        }
        SumKind::SignedOnly => {
            let mut total: i128 = 0;
            for pool in pools {
                let ChainVals::Signed(cell) = pool else {
                    unreachable!("all remaining pools are signed");
                };
                let Some(value) = cell.get(row) else {
                    return SumStatus::Skip;
                };
                let Some(next) = total.checked_add(value) else {
                    return SumStatus::Overflow;
                };
                total = next;
            }
            SumStatus::Value(SumResult::Signed(total))
        }
    }
}

/// The target value of one row in its own domain.
fn target_cell(pool: &ChainVals, row: usize) -> Option<SumResult> {
    match pool {
        ChainVals::Signed(cell) => cell.get(row).map(SumResult::Signed),
        ChainVals::Unsigned(cell) => cell.get(row).map(SumResult::Unsigned),
        ChainVals::Float(cell) => cell.get(row).map(SumResult::Float),
        ChainVals::Text(_) => None,
    }
}

/// Whether a computed sum matches the target. Integer-vs-integer
/// comparisons stay in the exact domain; any float folds to `f64` where
/// `tolerance` bounds the allowed absolute difference. Signed-vs-unsigned
/// integer mixes compare exactly when the sum fits `i128`, otherwise the
/// row fails (matching total-order semantics, no rounding).
fn sum_equals_target(sum: SumResult, target: SumResult, tolerance: f64) -> bool {
    match (sum, target) {
        (SumResult::Signed(sum), SumResult::Signed(target)) => sum == target,
        (SumResult::Unsigned(sum), SumResult::Unsigned(target)) => sum == target,
        (SumResult::Float(sum), SumResult::Float(target)) => (sum - target).abs() <= tolerance,
        (SumResult::Signed(sum), SumResult::Unsigned(target)) => {
            target <= i128::MAX as u128 && sum == target as i128
        }
        (SumResult::Unsigned(sum), SumResult::Signed(target)) => {
            sum <= i128::MAX as u128 && sum as i128 == target
        }
        (sum, target) => {
            let as_f64 = |result: &SumResult| match *result {
                SumResult::Signed(v) => v as f64,
                SumResult::Unsigned(v) => v as f64,
                SumResult::Float(v) => v,
            };
            (as_f64(&sum) - as_f64(&target)).abs() <= tolerance
        }
    }
}

/// Checks that every non-null value in `left` also appears in `by` for that
/// row: a null `left` skips the row, a null `by` implies a missing mapping.
pub struct Implies {
    pub left: String,
    pub by: String,
}

impl DataFrameCheck for Implies {
    fn name(&self) -> &str {
        "implies"
    }

    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::Implies {
            left: self.left.clone(),
            by: self.by.clone(),
        }
    }

    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let left = match df.column(&self.left) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.left))],
        };
        let by = match df.column(&self.by) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.by))],
        };
        // A null `left` skips the row; a null `by` under a present `left`
        // fails. `is_null` masks carry no validity bitmap themselves, so
        // the AND composes null-free.
        let failed_mask = left.is_not_null() & by.is_null();
        let failed: Vec<usize> = get_true_indices(&failed_mask);
        if failed.is_empty() {
            Vec::new()
        } else {
            vec![ValidationError::multi(
                vec![self.left.clone(), self.by.clone()],
                self.name(),
                failed.clone(),
                format!(
                    "where '{}' is present, '{}' must not be null ({} row(s))",
                    self.left,
                    self.by,
                    failed.len(),
                ),
            )]
        }
    }
}

/// Functional dependency: each distinct non-null value in `a` must map to
/// exactly one distinct non-null value in `b`. Row with a null `a` or a
/// null `b` does not participate. Every participating row of a conflicting
/// group (first row plus deviants) is reported.
pub struct Determines {
    pub a: String,
    pub b: String,
}

impl DataFrameCheck for Determines {
    fn name(&self) -> &str {
        "determines"
    }

    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::Determines {
            a: self.a.clone(),
            b: self.b.clone(),
        }
    }

    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let a = match df.column(&self.a) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.a))],
        };
        let b = match df.column(&self.b) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.b))],
        };
        // Only rows where both cells participate (neither is null) take
        // part in the functional dependency, so sorting starts from the
        // participating subset with its original row numbers attached.
        let failed = determines_rows(a, b);
        if failed.is_empty() {
            Vec::new()
        } else {
            vec![ValidationError::multi(
                vec![self.a.clone(), self.b.clone()],
                self.name(),
                failed.clone(),
                format!(
                    "each distinct '{}' value must map to exactly one '{}' value ({} conflicting row(s))",
                    self.a, self.b, failed.len(),
                ),
            )]
        }
    }
}

/// Conflicting rows for `b = f(a)`: scalar fallback. Compares the `Debug`
/// representations of the cells, which works for every dtype, and mirrors
/// the historical implementation (a null on either side skips the row; the
/// first occurrence of each `a` value binds its `b`; deviants and their
/// binding row are reported).
fn determines_rows_scalar(a: &Series, b: &Series) -> Vec<usize> {
    let mut first_b: HashMap<String, (String, usize)> = HashMap::new();
    let mut first_row_reported: HashSet<usize> = HashSet::new();
    let mut failed: Vec<usize> = Vec::new();
    for row in 0..a.len() {
        let a_cell = match a.get(row).ok() {
            Some(value) if !matches!(value, AnyValue::Null) => format!("{value:?}"),
            _ => continue, // a null (or unreadable) `a` does not participate
        };
        let b_cell = match b.get(row).ok() {
            Some(value) if !matches!(value, AnyValue::Null) => format!("{value:?}"),
            _ => continue, // a null `b` does not participate
        };
        match first_b.entry(a_cell) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let (seen, first_row) = entry.get();
                if *seen != b_cell {
                    failed.push(row);
                    if first_row_reported.insert(*first_row) {
                        failed.push(*first_row);
                    }
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert((b_cell, row));
            }
        }
    }
    failed.sort_unstable();
    failed.dedup();
    failed
}

/// Vectorized conflict detection without the lazy engine. Participants
/// are filtered eagerly, then stably sorted by `a` alone (participants
/// start row-ascending, so that equals sorting by `(a, row)`); the
/// per-group reference index is filled forward in a small boolean scan and
/// the actual value comparisons run through Polars's chunked kernels.
fn determines_rows_vectorized(a: &Series, b: &Series) -> PolarsResult<Vec<usize>> {
    let height = a.len();
    let mask = (&a.is_not_null()) & (&b.is_not_null());
    let rows_full = Series::new(
        PlSmallStr::from_static("__row"),
        (0..height as i64).collect::<Vec<i64>>(),
    );
    let a_f = a.filter(&mask)?;
    let b_f = b.filter(&mask)?;
    let rows_f = rows_full.filter(&mask)?;

    let n_sorted = a_f.len();
    if n_sorted == 0 {
        return Ok(Vec::new());
    }

    // Stable ascending sort keeps the participants of one `a` group in
    // ascending row order, so the first member is the binding row.
    // Single threaded because the check already runs inside the parallel
    // validation loop.
    let order = a_f.arg_sort(SortOptions {
        descending: false,
        nulls_last: false,
        multithreaded: false,
        maintain_order: true,
        limit: None,
    });
    let a_s = a_f.take(&order)?;
    let b_s = b_f.take(&order)?;
    let rows_s = rows_f.take(&order)?;
    let row_ids = rows_s.i64()?;

    // Group boundaries: row i (for i >= 1) belongs to the previous row's
    // group iff its `a` equals the previous `a`. Null cells cannot appear
    // because only participating rows are kept.
    let same_prev = a_s
        .slice(1, n_sorted - 1)
        .equal(&a_s.slice(0, n_sorted - 1))?;

    // Pass 1: fill forward the per-group reference index.
    let mut ref_idx: Vec<u32> = Vec::with_capacity(n_sorted);
    let mut current: u32 = 0;
    let mut same_values = same_prev
        .downcast_iter()
        .flat_map(|chunk| chunk.values_iter());
    for position in 0..n_sorted {
        let same: bool = if position == 0 {
            false
        } else {
            same_values.next().unwrap_or(false)
        };
        if !same {
            current = position as u32;
        }
        ref_idx.push(current);
    }

    // The reference `b` of each position's group, brought in by gather so
    // the comparison itself stays a chunked kernel call.
    let ref_idx_ca = IdxCa::from_vec(PlSmallStr::from_static("__ref"), ref_idx);
    let ref_b = b_s.take(&ref_idx_ca)?;
    let deviant = b_s.not_equal(&ref_b)?;

    // Pass 2: deviant rows and their binding row become the failed set.
    let mut failed: Vec<usize> = Vec::new();
    let mut same_values = same_prev
        .downcast_iter()
        .flat_map(|chunk| chunk.values_iter());
    let mut deviant_values = deviant
        .downcast_iter()
        .flat_map(|chunk| chunk.values_iter());
    for position in 0..n_sorted {
        let same: bool = if position == 0 {
            false
        } else {
            same_values.next().unwrap_or(false)
        };
        let deviant = deviant_values.next().unwrap_or(false);
        if same && deviant {
            let deviant_row = row_ids.get(position).unwrap_or(0);
            failed.push(deviant_row as usize);
            let binding = ref_idx_ca.get(position).unwrap_or(0) as usize;
            let binding_row = row_ids.get(binding).unwrap_or(0);
            failed.push(binding_row as usize);
        }
    }
    failed.sort_unstable();
    failed.dedup();
    Ok(failed)
}

/// Conflicting rows for `b = f(a)`; nested and binary dtypes (and any
/// engine failure) fall back to the scalar path, which supports every
/// dtype.
fn determines_rows(a: &Series, b: &Series) -> Vec<usize> {
    fn comparable(series: &Series) -> bool {
        !matches!(
            series.dtype(),
            DataType::List(_) | DataType::Array(_, _) | DataType::Struct(_) | DataType::Binary
        )
    }
    if comparable(a) && comparable(b) {
        if let Ok(rows) = determines_rows_vectorized(a, b) {
            return rows;
        }
    }
    determines_rows_scalar(a, b)
}

/// Checks that `right - left` at most `max_gap_ns`. Both columns must be
/// temporal with identical dtype; temporal ticks scale to nanoseconds via
/// the shared dtype's time unit (`Date` ticks are days). Negative gaps are
/// allowed and simply pass when the difference is even more negative.
pub struct Within {
    pub left: String,
    pub right: String,
    pub max_gap_ns: i64,
}

/// Nanoseconds per stored tick of a temporal dtype.
pub(crate) fn ns_per_tick(dtype: &DataType) -> i128 {
    match dtype {
        DataType::Date => 86_400_000_000_000,
        DataType::Datetime(unit, _) | DataType::Duration(unit) => match unit {
            TimeUnit::Nanoseconds => 1,
            TimeUnit::Microseconds => 1_000,
            TimeUnit::Milliseconds => 1_000_000,
        },
        DataType::Time => 1,
        _ => 1,
    }
}

impl DataFrameCheck for Within {
    fn name(&self) -> &str {
        "within"
    }

    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::Within {
            left: self.left.clone(),
            right: self.right.clone(),
            max_gap_ns: self.max_gap_ns,
        }
    }

    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let left = match df.column(&self.left) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.left))],
        };
        let right = match df.column(&self.right) {
            Ok(c) => c.as_materialized_series(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.right))],
        };
        if !is_temporal_dtype(left.dtype()) || !is_temporal_dtype(right.dtype()) {
            return vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "'{}' and '{}' must both be temporal columns",
                    self.left, self.right
                ),
            )];
        }
        if left.dtype() != right.dtype() {
            return vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "Cannot gap '{}' ({}) and '{}' ({}): temporal columns must share a dtype",
                    self.left,
                    left.dtype(),
                    self.right,
                    right.dtype(),
                ),
            )];
        }
        let scale = ns_per_tick(left.dtype());
        let left_ticks = match int128_ca(left) {
            Ok(values) => values,
            Err(error) => {
                return vec![ValidationError::multi(
                    vec![self.left.clone(), self.right.clone()],
                    self.name(),
                    vec![],
                    format!("Failed to extract ticks from '{}': {error}", self.left),
                )]
            }
        };
        let right_ticks = match int128_ca(right) {
            Ok(values) => values,
            Err(error) => {
                return vec![ValidationError::multi(
                    vec![self.left.clone(), self.right.clone()],
                    self.name(),
                    vec![],
                    format!("Failed to extract ticks from '{}': {error}", self.right),
                )]
            }
        };
        // The bound is divided once instead of every row scaling its own
        // difference. Failing iff `difference * scale > gap_ns` is exactly
        // failing iff `difference > gap_ns.div_euclid(scale)` (floor
        // division), given tick differences are integers — so the
        // per-row multiplication (and the old overflow silent-row-skip)
        // disappear without changing any verdict.
        let cap = i128::from(self.max_gap_ns).div_euclid(scale);
        let failed: Vec<usize> = left_ticks
            .iter()
            .zip(right_ticks.iter())
            .enumerate()
            .filter_map(|(row, (l, r))| match (l, r) {
                // Both cells are widened from temporal `i64`/`i32` ticks,
                // so the subtraction cannot overflow `i128`.
                (Some(l), Some(r)) if r - l > cap => Some(row),
                _ => None,
            })
            .collect();
        if failed.is_empty() {
            Vec::new()
        } else {
            vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                failed.clone(),
                format!(
                    "gap from '{}' to '{}' exceeds {}ns ({} row(s))",
                    self.left,
                    self.right,
                    self.max_gap_ns,
                    failed.len(),
                ),
            )]
        }
    }
}

/// Checks that the combination of columns is unique across rows.
pub struct UniqueTogether {
    pub columns: Vec<String>,
}

impl DataFrameCheck for UniqueTogether {
    fn name(&self) -> &str {
        "unique_together"
    }
    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::UniqueTogether {
            columns: self.columns.clone(),
        }
    }
    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let mut selected = Vec::with_capacity(self.columns.len());
        for name in &self.columns {
            match df.column(name) {
                Ok(c) => selected.push(c.as_materialized_series().clone()),
                Err(_) => return vec![missing_column_error(std::slice::from_ref(name))],
            }
        }
        let height = df.height();
        let frame = match DataFrame::new(height, selected.into_iter().map(Column::from).collect()) {
            Ok(frame) => frame,
            Err(error) => {
                return vec![ValidationError::multi(
                    self.columns.clone(),
                    self.name(),
                    vec![],
                    format!("UniqueTogether failed to build comparison frame: {error}"),
                )];
            }
        };
        let mask = match frame.is_duplicated() {
            Ok(mask) => mask,
            Err(error) => {
                return vec![ValidationError::multi(
                    self.columns.clone(),
                    self.name(),
                    vec![],
                    format!("UniqueTogether failed: {error}"),
                )];
            }
        };
        if mask.any() {
            let rows = get_true_indices(&mask);
            vec![ValidationError::multi(
                self.columns.clone(),
                self.name(),
                rows.clone(),
                format!(
                    "duplicate ({}) combinations in {} row(s)",
                    self.columns.join(", "),
                    rows.len(),
                ),
            )]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> DataFrame {
        DataFrame::new(
            3,
            vec![
                Series::new("start".into(), [1i64, 2, 3]).into(),
                Series::new("end".into(), [2i64, 2, 4]).into(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn comparison_reports_failing_rows() {
        let check = ColumnComparison {
            left: "start".to_string(),
            op: CompareOp::Lt,
            right: "end".to_string(),
        };
        let errors = check.validate(&frame());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
        assert_eq!(
            errors[0].columns,
            vec!["start".to_string(), "end".to_string()]
        );
    }

    #[test]
    fn comparison_skips_nulls_and_reports_missing() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [Some(1i64), None]).into(),
                Series::new("b".into(), [Some(0i64), Some(5)]).into(),
            ],
        )
        .unwrap();
        let check = ColumnComparison {
            left: "a".to_string(),
            op: CompareOp::Gt,
            right: "b".to_string(),
        };
        let errors = check.validate(&df);
        assert!(errors.is_empty());

        let missing = ColumnComparison {
            left: "a".to_string(),
            op: CompareOp::Gt,
            right: "nope".to_string(),
        };
        assert_eq!(missing.validate(&df)[0].check, "column_exists");
    }

    #[test]
    fn unique_together_reports_duplicates() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), ["x", "x", "y"]).into(),
                Series::new("b".into(), [1i64, 1, 2]).into(),
            ],
        )
        .unwrap();
        let check = UniqueTogether {
            columns: vec!["a".to_string(), "b".to_string()],
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn mixed_temporal_units_are_rejected() {
        // 1970-01-03 as days (2) vs a microsecond timestamp: raw ticks are
        // unrelated magnitudes, so the comparison must refuse, not guess.
        let dates = Series::new("day".into(), [2i32])
            .cast(&DataType::Date)
            .unwrap();
        let instants = Series::new("instant".into(), [2i64])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        let df = DataFrame::new(1, vec![dates.into(), instants.into()]).unwrap();
        let check = ColumnComparison {
            left: "day".to_string(),
            op: CompareOp::Lt,
            right: "instant".to_string(),
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must share a dtype"));
    }

    #[test]
    fn mixed_i128_u128_compares_exactly() {
        // 2^127 - 1 < 2^127: distinct in integers, equal in f64.
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [i128::MAX, i128::MAX]).into(),
                Series::new("b".into(), [1u128 << 127, 1u128 << 127]).into(),
            ],
        )
        .unwrap();
        let check = ColumnComparison {
            left: "a".to_string(),
            op: CompareOp::Lt,
            right: "b".to_string(),
        };
        assert!(check.validate(&df).is_empty());

        let flipped = ColumnComparison {
            left: "b".to_string(),
            op: CompareOp::Lt,
            right: "a".to_string(),
        };
        let errors = flipped.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn wide_integers_compare_exactly_across_columns() {
        let two_to_53: u64 = 1 << 53;
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [two_to_53, two_to_53]).into(),
                Series::new("b".into(), [two_to_53 + 1, two_to_53]).into(),
            ],
        )
        .unwrap();
        let check = ColumnComparison {
            left: "a".to_string(),
            op: CompareOp::Lt,
            right: "b".to_string(),
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
    }

    #[test]
    fn ordered_chain_reports_violating_rows() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), [1i64, 3, 2]).into(),
                Series::new("b".into(), [2i64, 2, 4]).into(),
                Series::new("c".into(), [3i64, 5, 7]).into(),
            ],
        )
        .unwrap();
        let check = Ordered {
            columns: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            strict: false,
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
        // Strict mode also rejects the equal row.
        let strict = Ordered {
            columns: check.columns.clone(),
            strict: true,
        };
        let errors = strict.validate(&df);
        assert_eq!(errors[0].rows, vec![1]); // a=3, b=2: 3 < 2 fails
    }

    #[test]
    fn ordered_requires_all_columns_of_one_kind() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [1i64, 2]).into(),
                Series::new("b".into(), ["x", "y"]).into(),
            ],
        )
        .unwrap();
        let check = Ordered {
            columns: vec!["a".to_string(), "b".to_string()],
            strict: false,
        };
        let errors = check.validate(&df);
        assert!(errors[0].message.contains("cannot be mixed"));
    }

    #[test]
    fn ordered_mixed_int_float_downgrades_to_f64() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [1i64, 2]).into(),
                Series::new("b".into(), [1.5f64, 0.5]).into(),
            ],
        )
        .unwrap();
        let check = Ordered {
            columns: vec!["a".to_string(), "b".to_string()],
            strict: false,
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
    }

    #[test]
    fn ordered_accepts_string_chain() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), ["a", "b", "b"]).into(),
                Series::new("b".into(), ["b", "b", "a"]).into(),
            ],
        )
        .unwrap();
        let check = Ordered {
            columns: vec!["a".to_string(), "b".to_string()],
            strict: false,
        };
        let errors = check.validate(&df);
        assert_eq!(errors[0].rows, vec![2]);
    }

    #[test]
    fn sum_columns_reports_mismatched_rows_and_skips_nulls() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), [Some(1i64), Some(2), None]).into(),
                Series::new("b".into(), [Some(2i64), Some(3), Some(4)]).into(),
                Series::new("c".into(), [Some(3i64), Some(9), Some(7)]).into(),
            ],
        )
        .unwrap();
        let check = SumColumns {
            addends: vec!["a".to_string(), "b".to_string()],
            target: "c".to_string(),
            tolerance: None,
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
    }

    #[test]
    fn sum_columns_float_tolerance() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [0.1f64, 1.5]).into(),
                Series::new("b".into(), [0.2f64, 1.5]).into(),
                Series::new("c".into(), [0.3f64, 10.0]).into(),
            ],
        )
        .unwrap();
        let check = SumColumns {
            addends: vec!["a".to_string(), "b".to_string()],
            target: "c".to_string(),
            tolerance: Some(1e-9),
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
    }

    #[test]
    fn implies_reports_missing_mandatory_values() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("left".into(), [Some(1i64), None, Some(3)]).into(),
                Series::new("by".into(), [Some("x"), None, None]).into(),
            ],
        )
        .unwrap();
        let check = Implies {
            left: "left".to_string(),
            by: "by".to_string(),
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![2]);
    }

    #[test]
    fn determines_reports_conflicting_groups() {
        let df = DataFrame::new(
            4,
            vec![
                Series::new("a".into(), [Some("x"), Some("x"), Some("y"), None]).into(),
                Series::new("b".into(), [1i64, 2, 3, 9]).into(),
            ],
        )
        .unwrap();
        let check = Determines {
            a: "a".to_string(),
            b: "b".to_string(),
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }

    #[test]
    fn within_bounds_temporal_difference() {
        let left = Series::new("d".into(), [0i32, 0, 1])
            .cast(&DataType::Date)
            .unwrap();
        let right = Series::new("e".into(), [2i32, 20, 3])
            .cast(&DataType::Date)
            .unwrap();
        let df = DataFrame::new(3, vec![left.into(), right.into()]).unwrap();
        let check = Within {
            left: "d".to_string(),
            right: "e".to_string(),
            // At most 3 days is exactly 3 * 86400 seconds in nanoseconds.
            max_gap_ns: 3i64 * 86_400_000_000_000,
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);

        let invalid = Within {
            left: "d".to_string(),
            right: "missing".to_string(),
            max_gap_ns: 1,
        };
        assert_eq!(invalid.validate(&df)[0].check, "column_exists");
    }

    #[test]
    fn within_rejects_non_temporal_and_mismatched_dtypes() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [1i64, 2]).into(),
                Series::new("b".into(), [2i64, 3]).into(),
            ],
        )
        .unwrap();
        let check = Within {
            left: "a".to_string(),
            right: "b".to_string(),
            max_gap_ns: 1_000,
        };
        let errors = check.validate(&df);
        assert!(errors[0].message.contains("temporal"));
    }
}
