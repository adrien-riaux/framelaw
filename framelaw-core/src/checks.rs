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
        "NotNull"
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
                format!(
                    "Found {} null values in '{}'",
                    series.null_count(),
                    column_name
                ),
            ));
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
                        format!("Found duplicated values in '{}'", column_name),
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
        "GreaterThanF64"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::GreaterThan(self.min_value)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(mask) = numeric_series(series).map(|ca| ca.lt_eq(self.min_value)) {
            if mask.any() {
                let rows = get_true_indices(&mask);
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    rows,
                    format!("Values in '{}' must be > {}", column_name, self.min_value),
                ));
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
    fn spec(&self) -> CheckSpec {
        CheckSpec::LessThan(self.max_value)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(mask) = numeric_series(series).map(|ca| ca.gt_eq(self.max_value)) {
            if mask.any() {
                let rows = get_true_indices(&mask);
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    rows,
                    format!("Values in '{}' must be < {}", column_name, self.max_value),
                ));
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

pub struct GreaterThanOrEqualF64 {
    pub min_value: f64,
}

impl Check for GreaterThanOrEqualF64 {
    fn name(&self) -> &str {
        "GreaterThanOrEqualF64"
    }

    fn spec(&self) -> CheckSpec {
        CheckSpec::GreaterThanOrEqual(self.min_value)
    }

    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        match numeric_series(series) {
            Ok(values) => {
                let mask = values.lt(self.min_value);
                if mask.any() {
                    vec![ValidationError::single(
                        column_name,
                        self.name(),
                        get_true_indices(&mask),
                        format!("Values in '{}' must be >= {}", column_name, self.min_value),
                    )]
                } else {
                    Vec::new()
                }
            }
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
        "LessThanOrEqualF64"
    }

    fn spec(&self) -> CheckSpec {
        CheckSpec::LessThanOrEqual(self.max_value)
    }

    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        match numeric_series(series) {
            Ok(values) => {
                let mask = values.gt(self.max_value);
                if mask.any() {
                    vec![ValidationError::single(
                        column_name,
                        self.name(),
                        get_true_indices(&mask),
                        format!("Values in '{}' must be <= {}", column_name, self.max_value),
                    )]
                } else {
                    Vec::new()
                }
            }
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
        "BetweenF64"
    }
    fn spec(&self) -> CheckSpec {
        CheckSpec::Between(self.min_value, self.max_value)
    }
    fn validate(&self, column_name: &str, series: &Series) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        if let Ok(ca) = numeric_series(series) {
            let mask_lt = ca.lt(self.min_value);
            let mask_gt = ca.gt(self.max_value);

            let combined_mask = mask_lt | mask_gt;
            if combined_mask.any() {
                let rows = get_true_indices(&combined_mask);
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    rows,
                    format!(
                        "Values in '{}' must be between {} and {}",
                        column_name, self.min_value, self.max_value
                    ),
                ));
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
    fn spec(&self) -> CheckSpec {
        CheckSpec::Matches(self.pattern.as_str().to_string())
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
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!(
                        "Values in '{}' do not match pattern '{}'",
                        column_name,
                        self.pattern.as_str()
                    ),
                ));
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
    fn spec(&self) -> CheckSpec {
        let mut values: Vec<String> = self.allowed_values.iter().cloned().collect();
        values.sort_unstable();
        CheckSpec::OneOf(values)
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
                errors.push(ValidationError::single(
                    column_name,
                    self.name(),
                    failed_rows,
                    format!(
                        "Values in '{}' are not part of the allowed set",
                        column_name
                    ),
                ));
            }
        } else {
            errors.push(type_mismatch_error(column_name, self.name(), "String"));
        }

        errors
    }
}

// --- Helper Functions ---

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
        format!("Failed to dynamically cast column to {}", expected_type),
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
        "ListLenBetween"
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
                            "Lists in '{}' must have length between {} and {}",
                            column_name, self.min_len, self.max_len
                        ),
                    )]
                }
            }
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "List or Array",
            )],
        }
    }
}

/// Checks that list/array columns are never empty (nulls are skipped).
pub struct ListNotEmpty;

impl Check for ListNotEmpty {
    fn name(&self) -> &str {
        "ListNotEmpty"
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
                        format!("Lists in '{}' must not be empty", column_name),
                    )]
                }
            }
            Err(_) => vec![type_mismatch_error(
                column_name,
                self.name(),
                "List or Array",
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
        "StructField"
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
                return vec![type_mismatch_error(column_name, self.name(), "Struct")];
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
        assert!(errors[0].message.contains("numeric type"));
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
        assert!(errors[0].message.contains("List or Array"));
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
}
