use polars::prelude::{DataFrame, DataType};

use crate::checks::Check;
use crate::errors::ValidationError;

pub struct ColumnSpec {
    pub name: String,
    pub dtype: DataType,
    pub nullable: bool,
    pub checks: Vec<Box<dyn Check>>,
}

impl ColumnSpec {
    pub fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        let series = match df.column(&self.name) {
            Ok(column) => column.as_materialized_series(),
            Err(_) => {
                errors.push(ValidationError::single(
                    &self.name,
                    "ColumnExists",
                    vec![],
                    format!("Column '{}' is missing from DataFrame", self.name),
                ));
                return errors;
            }
        };

        let null_column_is_compatible = self.nullable && *series.dtype() == DataType::Null;
        if !dtype_matches(&self.dtype, series.dtype()) && !null_column_is_compatible {
            errors.push(ValidationError::single(
                &self.name,
                "Dtype",
                vec![],
                format!(
                    "Column '{}' has dtype {}, expected {}",
                    self.name,
                    series.dtype(),
                    self.dtype
                ),
            ));
            return errors;
        }

        if !self.nullable && series.null_count() > 0 {
            let rows = series
                .is_null()
                .iter()
                .enumerate()
                .filter_map(|(index, value)| value.filter(|is_null| *is_null).map(|_| index))
                .collect();

            errors.push(ValidationError::single(
                &self.name,
                "NotNull",
                rows,
                format!(
                    "Found {} null values in '{}'",
                    series.null_count(),
                    self.name
                ),
            ));
        }

        if null_column_is_compatible {
            return errors;
        }

        // Run all individual checks against the series
        for check in &self.checks {
            if !self.nullable && check.name() == "NotNull" {
                continue;
            }

            let mut check_errors = check.validate(&self.name, series);
            errors.append(&mut check_errors);
        }

        errors
    }
}

/// Structural dtype comparison supporting nested List/Struct/Array.
/// Scalar dtypes require exact equality; nested dtypes recurse into
/// inner types so `List(Int32)` does not silently accept `List(String)`.
pub fn dtype_matches(expected: &DataType, actual: &DataType) -> bool {
    match (expected, actual) {
        (DataType::List(expected_inner), DataType::List(actual_inner)) => {
            dtype_matches(expected_inner, actual_inner)
        }
        (
            DataType::Array(expected_inner, expected_width),
            DataType::Array(actual_inner, actual_width),
        ) => expected_width == actual_width && dtype_matches(expected_inner, actual_inner),
        (DataType::Struct(expected_fields), DataType::Struct(actual_fields)) => {
            expected_fields.len() == actual_fields.len()
                && expected_fields
                    .iter()
                    .zip(actual_fields.iter())
                    .all(|(e, a)| e.name == a.name && dtype_matches(&e.dtype, &a.dtype))
        }
        (expected, actual) => expected == actual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checks::NotNull;
    use polars::prelude::{NamedFrom, Series};

    #[test]
    fn reports_dtype_mismatch_without_running_checks() {
        let df = DataFrame::new(2, vec![Series::new("age".into(), [18i32, 21]).into()]).unwrap();
        let column = ColumnSpec {
            name: "age".to_string(),
            dtype: DataType::Int64,
            nullable: true,
            checks: vec![Box::new(NotNull)],
        };

        let errors = column.validate(&df);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].check, "Dtype");
        assert!(errors[0].rows.is_empty());
    }

    #[test]
    fn nullable_false_reports_null_rows_once() {
        let df = DataFrame::new(
            3,
            vec![Series::new("age".into(), [Some(18i32), None, Some(21)]).into()],
        )
        .unwrap();
        let column = ColumnSpec {
            name: "age".to_string(),
            dtype: DataType::Int32,
            nullable: false,
            checks: vec![Box::new(NotNull)],
        };

        let errors = column.validate(&df);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].check, "NotNull");
        assert_eq!(errors[0].rows, vec![1]);
    }

    #[test]
    fn nested_dtypes_match_structurally() {
        let list_i64 = DataType::List(Box::new(DataType::Int64));
        let list_str = DataType::List(Box::new(DataType::String));
        assert!(dtype_matches(&list_i64, &list_i64));
        assert!(!dtype_matches(&list_i64, &list_str));
        assert!(!dtype_matches(&list_i64, &DataType::Int64));

        let array_a = DataType::Array(Box::new(DataType::Int64), 2);
        let array_b = DataType::Array(Box::new(DataType::Int64), 3);
        assert!(dtype_matches(&array_a, &array_a));
        assert!(!dtype_matches(&array_a, &array_b));
    }
}
