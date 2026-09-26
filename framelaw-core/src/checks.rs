use polars::prelude::*;
use regex::Regex;
use std::collections::HashSet;

use crate::errors::ValidationError;

pub trait Check: Send + Sync {
    fn name(&self) -> &str;
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError>;
}

/// Checks that no values in the series are null.
pub struct NotNull;

impl Check for NotNull {
    fn name(&self) -> &str {
        "NotNull"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if series.null_count() > 0 {
            let mask = series.is_null();
            let rows = get_true_indices(&mask);

            errors.push(ValidationError {
                column: column_name.to_string(),
                check: self.name().to_string(),
                rows,
                message: format!(
                    "Found {} null values in '{}'",
                    series.null_count(),
                    column_name
                ),
            });
        }
        errors
    }
}

/// Checks that all values in the series are unique (no duplicates).
pub struct UniqueValues;

impl Check for UniqueValues {
    fn name(&self) -> &str {
        "UniqueValues"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(df) = DataFrame::new(series.len(), vec![series.clone().into()]) {
            if let Ok(mask) = df.is_duplicated() {
                if mask.any() {
                    let rows = get_true_indices(&mask);
                    errors.push(ValidationError {
                        column: column_name.to_string(),
                        check: self.name().to_string(),
                        rows,
                        message: format!("Found duplicated values in '{}'", column_name),
                    });
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
        "GreaterThanF64"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(mask) = numeric_series(series).map(|ca| ca.lt_eq(self.min_value)) {
            if mask.any() {
                let rows = get_true_indices(&mask);
                errors.push(ValidationError {
                    column: column_name.to_string(),
                    check: self.name().to_string(),
                    rows,
                    message: format!("Values in '{}' must be > {}", column_name, self.min_value),
                });
            }
        } else {
            errors.push(type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            ));
        }

        errors
    }
}

/// Checks that all numeric values are STRICTLY LESS than the provided threshold.
pub struct LessThanF64 {
    pub max_value: f64,
}

impl Check for LessThanF64 {
    fn name(&self) -> &str {
        "LessThanF64"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(mask) = numeric_series(series).map(|ca| ca.gt_eq(self.max_value)) {
            if mask.any() {
                let rows = get_true_indices(&mask);
                errors.push(ValidationError {
                    column: column_name.to_string(),
                    check: self.name().to_string(),
                    rows,
                    message: format!("Values in '{}' must be < {}", column_name, self.max_value),
                });
            }
        } else {
            errors.push(type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            ));
        }

        errors
    }
}

/// Checks that all numeric values are inclusively between `min_value` and `max_value`.
pub struct BetweenF64 {
    pub min_value: f64,
    pub max_value: f64,
}

impl Check for BetweenF64 {
    fn name(&self) -> &str {
        "BetweenF64"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = numeric_series(series) {
            let mask_lt = ca.lt(self.min_value);
            let mask_gt = ca.gt(self.max_value);

            let combined_mask = mask_lt | mask_gt;
            if combined_mask.any() {
                let rows = get_true_indices(&combined_mask);
                errors.push(ValidationError {
                    column: column_name.to_string(),
                    check: self.name().to_string(),
                    rows,
                    message: format!(
                        "Values in '{}' must be between {} and {}",
                        column_name, self.min_value, self.max_value
                    ),
                });
            }
        } else {
            errors.push(type_mismatch_error(
                column_name,
                self.name(),
                "a numeric type",
            ));
        }

        errors
    }
}

/// Checks that string values match a provided Regular Expression.
pub struct MatchesRegex {
    pub pattern: Regex,
}

impl Check for MatchesRegex {
    fn name(&self) -> &str {
        "MatchesRegex"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if !self.pattern.is_match(val) {
                        failed_rows.push(i);
                    }
                }
            }

            if !failed_rows.is_empty() {
                errors.push(ValidationError {
                    column: column_name.to_string(),
                    check: self.name().to_string(),
                    rows: failed_rows,
                    message: format!(
                        "Values in '{}' do not match pattern '{}'",
                        column_name,
                        self.pattern.as_str()
                    ),
                });
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "String"));
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
        "OneOfString"
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = series.str() {
            let mut failed_rows = Vec::new();

            for (i, val_opt) in ca.iter().enumerate() {
                if let Some(val) = val_opt {
                    if !self.allowed_values.contains(val) {
                        failed_rows.push(i);
                    }
                }
            }

            if !failed_rows.is_empty() {
                errors.push(ValidationError {
                    column: column_name.to_string(),
                    check: self.name().to_string(),
                    rows: failed_rows,
                    message: format!(
                        "Values in '{}' are not part of the allowed set",
                        column_name
                    ),
                });
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "String"));
        }

        errors
    }
}

// --- Helper Functions ---

/// Casts supported scalar numeric columns once so checks can use Polars' native
/// vectorized comparisons instead of converting values through Python.
fn numeric_series(series: &Series) -> PolarsResult<Float64Chunked> {
    if !matches!(
        series.dtype(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    ) {
        return Err(PolarsError::ComputeError("expected a numeric type".into()));
    }

    series.cast(&DataType::Float64)?.f64().cloned()
}

/// Retrieves the raw row indexing where the BooleanChunked mask answers true.
fn get_true_indices(mask: &BooleanChunked) -> Vec<usize> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, b)| if let Some(true) = b { Some(i) } else { None })
        .collect()
}

/// Returns a boilerplate error for failed typecasts.
fn type_mismatch_error(column: &str, check: &str, expected_type: &str) -> ValidationError {
    ValidationError {
        column: column.to_string(),
        check: check.to_string(),
        rows: vec![],
        message: format!("Failed to dynamically cast column to {}", expected_type),
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
        assert!(errors[0].message.contains("numeric type"));
    }
}
