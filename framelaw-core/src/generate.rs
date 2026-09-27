use std::collections::HashMap;

use polars::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
) -> PolarsResult<f64> {
    if bounds.is_empty() {
        return Err(PolarsError::ComputeError(
            "Unsatisfiable numeric bounds in schema".into(),
        ));
    }
    let lo = if bounds.lo.is_finite() {
        if bounds.lo_inclusive {
            bounds.lo
        } else {
            bounds.lo.next_up()
        }
    } else {
        fallback_lo
    };
    let hi = if bounds.hi.is_finite() {
        if bounds.hi_inclusive {
            bounds.hi
        } else {
            bounds.hi.next_down()
        }
    } else {
        fallback_hi
    };
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
        return Ok(allowed[rng.random_range(0..allowed.len())].clone());
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
    let bounds = numeric_bounds(specs);
    let (fallback_lo, fallback_hi) = match dtype {
        DataType::UInt8 => (0.0, 200.0),
        DataType::UInt16 => (0.0, 1000.0),
        DataType::UInt32 => (0.0, 100_000.0),
        DataType::UInt64 | DataType::UInt128 => (0.0, 1_000_000.0),
        DataType::Int8 => (-100.0, 100.0),
        _ => (0.0, 1000.0),
    };
    let values: Vec<f64> = (0..n)
        .map(|_| sample_f64_in(rng, &bounds, fallback_lo, fallback_hi))
        .collect::<PolarsResult<_>>()?;
    let float_series = Series::new(name.clone(), values);
    cast_to_dtype(float_series, dtype)
}

fn gen_float_series(
    rng: &mut ChaCha8Rng,
    name: &PlSmallStr,
    dtype: &DataType,
    specs: &[CheckSpec],
    n: usize,
) -> PolarsResult<Series> {
    let bounds = numeric_bounds(specs);
    let values: Vec<f64> = (0..n)
        .map(|_| sample_f64_in(rng, &bounds, 0.0, 1.0))
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
    let bounds = numeric_bounds(specs);
    let values: Vec<i64> = (0..n)
        .map(|_| sample_f64_in(rng, &bounds, 1.0, 10_000.0).map(|v| v.round() as i64))
        .collect::<PolarsResult<_>>()?;
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

fn comparison_holds(op: CompareOp, left: &str, right: &str, numeric: bool) -> bool {
    if numeric {
        let (Ok(left), Ok(right)) = (left.parse::<f64>(), right.parse::<f64>()) else {
            return false;
        };
        return compare_values(op, left, right);
    }
    compare_values(op, left, right)
}

fn compare_values<T: PartialEq + PartialOrd>(op: CompareOp, left: T, right: T) -> bool {
    match op {
        CompareOp::Lt => left < right,
        CompareOp::Le => left <= right,
        CompareOp::Gt => left > right,
        CompareOp::Ge => left >= right,
        CompareOp::Eq => left == right,
        CompareOp::Ne => left != right,
    }
}

fn enforce_comparison(
    frame: &mut HashMap<String, Series>,
    lookup: &HashMap<&str, &ColumnSpec>,
    rng: &mut ChaCha8Rng,
    left: &str,
    op: CompareOp,
    right: &str,
) -> PolarsResult<()> {
    let n = frame
        .get(left)
        .map(|series| series.len())
        .unwrap_or_else(|| frame.get(right).map(|series| series.len()).unwrap_or(0));
    let left_is_numeric = lookup
        .get(left)
        .map(|spec| is_supported_numeric_dtype(&spec.dtype))
        .unwrap_or(false);
    let right_is_numeric = lookup
        .get(right)
        .map(|spec| is_supported_numeric_dtype(&spec.dtype))
        .unwrap_or(false);
    if left_is_numeric != right_is_numeric {
        return Err(PolarsError::ComputeError(
            format!("Cannot satisfy comparison '{left} {op:?} {right}': mismatched kinds").into(),
        ));
    }
    let numeric = left_is_numeric;

    for row in 0..n {
        let left_value = frame[left]
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default();
        let right_value = frame[right]
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default();
        if comparison_holds(op, &left_value, &right_value, numeric) {
            continue;
        }
        let mut satisfied = false;
        // Joint rejection sampling: redraw both sides from their own
        // distributions so pairs like `start < end` succeed even when the
        // left value sits at the top of its range.
        for _ in 0..200 {
            let left_candidate = resample_one(rng, lookup, left)?;
            let right_candidate = resample_one(rng, lookup, right)?;
            let left_string = left_candidate
                .get(0)
                .map(|v| v.to_string())
                .unwrap_or_default();
            let right_string = right_candidate
                .get(0)
                .map(|v| v.to_string())
                .unwrap_or_default();
            if comparison_holds(op, &left_string, &right_string, numeric) {
                for (name, candidate) in [(left, left_candidate), (right, right_candidate)] {
                    let values: Vec<AnyValue> = (0..n)
                        .map(|index| {
                            if index == row {
                                candidate.get(0).unwrap()
                            } else {
                                frame[name].get(index).unwrap()
                            }
                        })
                        .collect();
                    let rebuilt = Series::from_any_values(PlSmallStr::from(name), &values, false)?;
                    let cast = cast_to_dtype(rebuilt, &lookup[name].dtype)?;
                    frame.insert(name.to_string(), cast);
                }
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
    let n = frame
        .get(&columns[0])
        .map(|series| series.len())
        .unwrap_or(0);
    let last = columns.last().unwrap().clone();
    for _ in 0..(n * 10 + 10) {
        let selected: Vec<Series> = columns.iter().map(|name| frame[name].clone()).collect();
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
            let candidate_value = candidate.get(0).unwrap();
            let values: Vec<AnyValue> = (0..n)
                .map(|index| {
                    if index == row {
                        candidate_value.clone()
                    } else {
                        frame[&last].get(index).unwrap()
                    }
                })
                .collect();
            let rebuilt = Series::from_any_values(PlSmallStr::from(last.as_str()), &values, false)?;
            let cast = cast_to_dtype(rebuilt, &lookup[last.as_str()].dtype)?;
            frame.insert(last.clone(), cast);
        }
    }
    Err(PolarsError::ComputeError(
        "Could not generate unique combinations".into(),
    ))
}

/// Generate a valid `DataFrame` satisfying the schema.
///
/// Strategies are derived from each column's dtype and checks, then
/// cross-column checks are enforced with rejection sampling. Failures
/// (contradictory bounds, unsatisfiable uniqueness) return an error.
pub fn sample(schema: &Schema, n: usize, seed: u64) -> PolarsResult<DataFrame> {
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

    // Re-validate uniqueness for single-column unique checks after table repair.
    for spec in &schema.columns {
        let series = &frame[&spec.name];
        if has_unique(&specs_of(spec)) && series.n_unique().unwrap_or(0) != series.len() {
            let repaired = ensure_unique(series.clone(), &mut rng, &spec.dtype, &specs_of(spec))?;
            frame.insert(spec.name.clone(), repaired);
        }
    }

    let mut columns = Vec::with_capacity(frame.len());
    for spec in &schema.columns {
        columns.push(Column::from(frame.remove(&spec.name).unwrap()));
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
}
