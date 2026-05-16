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
            Ok(s) => s,
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

        // Run all individual checks against the series
        for check in &self.checks {
            let mut check_errors = check.validate(&self.name, series);
            errors.append(&mut check_errors);
        }

        errors
    }
}
