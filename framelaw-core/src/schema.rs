use crate::column::ColumnSpec;
use crate::dataframe_checks::DataFrameCheck;
use crate::errors::{ValidationError, ValidationReport};
use polars::prelude::DataFrame;
use rayon::prelude::*;

pub struct Schema {
    pub columns: Vec<ColumnSpec>,
    pub table_checks: Vec<Box<dyn DataFrameCheck>>,
}

impl Schema {
    pub fn new(columns: Vec<ColumnSpec>) -> Self {
        Self {
            columns,
            table_checks: Vec::new(),
        }
    }

    pub fn with_table_checks(
        columns: Vec<ColumnSpec>,
        table_checks: Vec<Box<dyn DataFrameCheck>>,
    ) -> Self {
        Self {
            columns,
            table_checks,
        }
    }

    pub fn validate(&self, df: &DataFrame) -> ValidationReport {
        let mut errors: Vec<ValidationError> = self
            .columns
            .par_iter()
            .flat_map(|col_spec| col_spec.validate(df))
            .collect();

        let mut table_errors: Vec<ValidationError> = self
            .table_checks
            .par_iter()
            .flat_map(|check| check.validate(df))
            .collect();

        errors.append(&mut table_errors);
        ValidationReport { errors }
    }
}
