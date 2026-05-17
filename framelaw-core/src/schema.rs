use crate::column::ColumnSpec;
use crate::errors::{ValidationError, ValidationReport};
use polars::prelude::DataFrame;
use rayon::prelude::*;

pub struct Schema {
    pub columns: Vec<ColumnSpec>,
}

impl Schema {
    pub fn new(columns: Vec<ColumnSpec>) -> Self {
        Self { columns }
    }

    pub fn validate(&self, df: &DataFrame) -> ValidationReport {
        let errors: Vec<ValidationError> = self
            .columns
            .par_iter()
            .flat_map(|col_spec| col_spec.validate(df))
            .collect();

        ValidationReport { errors }
    }
}
