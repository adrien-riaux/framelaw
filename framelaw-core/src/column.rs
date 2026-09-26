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
                errors.push(ValidationError {
                    column: self.name.clone(),
                    check: "ColumnExists".to_string(),
                    rows: vec![],
                    message: format!("Column '{}' is missing from DataFrame", self.name),
                });
                return errors;
            }
        };

        if self.dtype != *series.dtype() {
            errors.push(ValidationError {
                column: self.name.clone(),
                check: "Dtype".to_string(),
                rows: vec![],
                message: format!(
                    "Column '{}' has dtype {}, expected {}",
                    self.name,
                    series.dtype(),
                    self.dtype
                ),
            });
            return errors;
        }

        if !self.nullable && series.null_count() > 0 {
            let rows = series
                .is_null()
                .iter()
                .enumerate()
                .filter_map(|(index, value)| value.filter(|is_null| *is_null).map(|_| index))
                .collect();

            errors.push(ValidationError {
                column: self.name.clone(),
                check: "NotNull".to_string(),
                rows,
                message: format!(
                    "Found {} null values in '{}'",
                    series.null_count(),
                    self.name
                ),
            });
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
}
