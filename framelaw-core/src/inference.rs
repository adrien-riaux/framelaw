use polars::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use std::collections::HashSet;

use crate::column::ColumnSpec;
use crate::schema::Schema;
use crate::checks::{
    Check, GreaterThanF64, LessThanF64, MatchesRegex, NotNull, OneOfString, UniqueValues,
};

/// Infer a `Schema` from an existing Polars `DataFrame`.
/// Computes stats concurrently across all columns using Rayon.
pub fn infer_schema(df: &DataFrame) -> Schema {
    let columns: Vec<ColumnSpec> = df
        .get_columns()
        .par_iter()
        .map(|series| infer_series_schema(series))
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
        if dtype.is_numeric() {
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

        // 4. Strings: Regex patterns and Categories (OneOfString)
        if let Ok(str_ca) = series.str() {
            // Heuristic for Enum / Categorical: If few unique values relative to length
            if n_unique > 0 && n_unique <= 10 && len > n_unique {
                let mut unique_vals = HashSet::new();
                for val in str_ca.into_iter().flatten() {
                    unique_vals.insert(val.to_string());
                }
                checks.push(Box::new(OneOfString {
                    allowed_values: unique_vals,
                }));
            } else if n_unique > 0 {
                // Heuristics for Strings formats using regex patterns.
                // We'll perform a quick scan on a sample of non-null values.
                let sample_size = std::cmp::min(100, len);
                let sample: Vec<&str> = str_ca.into_iter().flatten().take(sample_size).collect();

                if !sample.is_empty() {
                    let is_uuid = sample.iter().all(|s| s.len() == 36 && s.contains('-')); // basic UUID heuristic
                    if is_uuid {
                        checks.push(Box::new(MatchesRegex { pattern: Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$").unwrap() }));
                    } else {
                        let is_email = sample.iter().all(|s| s.contains('@') && s.contains('.'));
                        if is_email {
                            checks.push(Box::new(MatchesRegex { pattern: Regex::new(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$").unwrap() }));
                        }
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
