use polars::prelude::*;
use regex::Regex;
use std::collections::HashSet;

use crate::checks::{
    Check, GreaterThanF64, LessThanF64, ListLenBetween, MatchesRegex, NotNull, OneOfString,
    UniqueValues,
};
use crate::column::ColumnSpec;
use crate::schema::Schema;
use rayon::prelude::*;

/// Infer a `Schema` from an existing Polars `DataFrame`.
/// Computes stats concurrently across all columns using Rayon.
pub fn infer_schema(df: &DataFrame) -> Schema {
    let dataframe_columns = df.columns();
    let columns: Vec<ColumnSpec> = dataframe_columns
        .par_iter()
        .map(|column| infer_series_schema(column.as_materialized_series()))
        .collect();

    Schema::new(columns)
}

fn infer_series_schema(series: &Series) -> ColumnSpec {
    let name = series.name().to_string();
    let dtype = series.dtype().clone();
    let null_count = series.null_count();
    let nullable = null_count > 0;

    let mut checks: Vec<Box<dyn Check>> = Vec::new();

    // 1. Nullability
    if !nullable {
        checks.push(Box::new(NotNull));
    }

    let len = series.len();
    if len > 0 {
        // 2. Uniqueness
        let n_unique = series.n_unique().unwrap_or(0);
        if n_unique == len && null_count == 0 {
            checks.push(Box::new(UniqueValues));
        }

        // 3. Numeric Bounds (Min/Max heuristics)
        if dtype.is_numeric() && supports_lossless_float_bounds(&dtype) {
            if let Ok(cast_s) = series.cast(&DataType::Float64) {
                if let Ok(f64_ca) = cast_s.f64() {
                    // Min bound
                    if let Some(min_val) = f64_ca.min() {
                        checks.push(Box::new(GreaterThanF64 {
                            min_value: min_val - 0.0001,
                        }));
                    }
                    // Max bound
                    if let Some(max_val) = f64_ca.max() {
                        checks.push(Box::new(LessThanF64 {
                            max_value: max_val + 0.0001,
                        }));
                    }
                }
            }
        }

        // 4. Nested List/Array lengths: observed min/max as starting bounds.
        if matches!(dtype, DataType::List(_) | DataType::Array(_, _)) {
            if let Some((min_len, max_len)) = nested_len_bounds(series) {
                checks.push(Box::new(ListLenBetween { min_len, max_len }));
            }
        }

        // 5. Strings: Regex patterns and Categories (OneOfString)
        if let Ok(str_ca) = series.str() {
            // Heuristic for Enum / Categorical: at most 20 values and 5% of rows.
            if n_unique > 0 && n_unique <= 20 && n_unique * 20 <= len {
                let mut unique_vals = HashSet::new();
                for val in str_ca.iter().flatten() {
                    unique_vals.insert(val.to_string());
                }
                checks.push(Box::new(OneOfString {
                    allowed_values: unique_vals,
                }));
            } else if n_unique > 0 {
                // Infer a format only when every non-null value matches it.
                let patterns = [
                    (
                        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
                        "uuid",
                    ),
                    (r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", "email"),
                    (r"^https?://[^\s/$.?#].[^\s]*$", "url"),
                    (
                        r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)?$",
                        "date",
                    ),
                ];

                for (pattern, _) in patterns {
                    let regex = Regex::new(pattern).expect("inference regexes are valid");
                    if str_ca.iter().flatten().all(|value| regex.is_match(value)) {
                        checks.push(Box::new(MatchesRegex { pattern: regex }));
                        break;
                    }
                }
            }
        }
    }

    ColumnSpec {
        name,
        dtype,
        nullable,
        checks,
    }
}

fn supports_lossless_float_bounds(dtype: &DataType) -> bool {
    !matches!(
        dtype,
        DataType::Int64 | DataType::UInt64 | DataType::Int128 | DataType::UInt128
    )
}

fn nested_len_bounds(series: &Series) -> Option<(usize, usize)> {
    let lengths: Vec<usize> = match series.dtype() {
        DataType::List(_) => series
            .list()
            .ok()?
            .iter()
            .flatten()
            .map(|s| s.len())
            .collect(),
        DataType::Array(_, _) => series
            .array()
            .ok()?
            .iter()
            .flatten()
            .map(|s| s.len())
            .collect(),
        _ => return None,
    };
    Some((*lengths.iter().min()?, *lengths.iter().max()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checks::CheckSpec;

    fn inferred_specs(schema: &Schema, column: &str) -> Vec<CheckSpec> {
        schema
            .columns
            .iter()
            .find(|spec| spec.name == column)
            .expect("column should be inferred")
            .checks
            .iter()
            .map(|check| check.spec())
            .collect()
    }

    #[test]
    fn infers_all_documented_string_patterns() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("email".into(), ["a@example.com", "b@example.com"]).into(),
                Series::new(
                    "uuid".into(),
                    [
                        "123e4567-e89b-12d3-a456-426614174000",
                        "123e4567-e89b-12d3-a456-426614174001",
                    ],
                )
                .into(),
                Series::new(
                    "url".into(),
                    ["https://example.com/a", "http://example.org/b"],
                )
                .into(),
                Series::new("date".into(), ["2026-09-26", "2026-09-27T12:30:00"]).into(),
            ],
        )
        .unwrap();

        let schema = infer_schema(&df);

        for name in ["email", "uuid", "url", "date"] {
            assert!(inferred_specs(&schema, name)
                .iter()
                .any(|spec| matches!(spec, CheckSpec::Matches(_))));
        }
    }

    #[test]
    fn infers_categories_only_with_documented_thresholds() {
        let allowed = (0..40)
            .map(|index| if index % 2 == 0 { "a" } else { "b" })
            .collect::<Vec<_>>();
        let too_many_rows = (0..40)
            .map(|index| format!("value-{}", index % 20))
            .collect::<Vec<_>>();
        let df = DataFrame::new(
            40,
            vec![
                Series::new("allowed".into(), allowed).into(),
                Series::new("not_allowed".into(), too_many_rows).into(),
            ],
        )
        .unwrap();

        let schema = infer_schema(&df);

        assert!(inferred_specs(&schema, "allowed")
            .iter()
            .any(|spec| matches!(spec, CheckSpec::OneOf(values) if values == &vec!["a".to_string(), "b".to_string()])));
        assert!(!inferred_specs(&schema, "not_allowed")
            .iter()
            .any(|spec| matches!(spec, CheckSpec::OneOf(_))));
    }

    #[test]
    fn avoids_bounds_for_potentially_lossy_64_bit_integers() {
        let df = DataFrame::new(
            2,
            vec![Series::new("value".into(), [u64::MAX - 1, u64::MAX]).into()],
        )
        .unwrap();

        let specs = inferred_specs(&infer_schema(&df), "value");

        assert!(!specs
            .iter()
            .any(|spec| { matches!(spec, CheckSpec::GreaterThan(_) | CheckSpec::LessThan(_)) }));
    }

    #[test]
    fn empty_and_all_null_columns_have_no_unsafe_checks() {
        let all_null = DataFrame::new(
            2,
            vec![Series::new("all_null".into(), [None::<i32>, None]).into()],
        )
        .unwrap();
        let empty = DataFrame::new(
            0,
            vec![Series::new("empty".into(), Vec::<i32>::new()).into()],
        )
        .unwrap();

        assert!(inferred_specs(&infer_schema(&all_null), "all_null").is_empty());
        assert!(inferred_specs(&infer_schema(&empty), "empty")
            .iter()
            .any(|spec| matches!(spec, CheckSpec::NotNull)));
    }
}
