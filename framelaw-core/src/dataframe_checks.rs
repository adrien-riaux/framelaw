use polars::prelude::*;

use crate::checks::{get_true_indices, numeric_series};
use crate::errors::ValidationError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl CompareOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Lt => "<",
            Self::Le => "<=",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Eq => "==",
            Self::Ne => "!=",
        }
    }
}

#[derive(Clone, Debug)]
pub enum DataFrameCheckSpec {
    ColumnComparison {
        left: String,
        op: CompareOp,
        right: String,
    },
    UniqueTogether {
        columns: Vec<String>,
    },
}

pub trait DataFrameCheck: Send + Sync {
    fn name(&self) -> &str;
    fn spec(&self) -> DataFrameCheckSpec;
    fn validate(&self, df: &DataFrame) -> Vec<ValidationError>;
}

fn missing_column_error(requested: &[String]) -> ValidationError {
    ValidationError::multi(
        requested.to_vec(),
        "ColumnExists",
        vec![],
        format!("Missing column(s): {}", requested.join(", ")),
    )
}

/// Declarative comparison between two columns, e.g. `start < end`.
/// Nulls on either side skip the row; use nullable=false for strictness.
pub struct ColumnComparison {
    pub left: String,
    pub op: CompareOp,
    pub right: String,
}

impl DataFrameCheck for ColumnComparison {
    fn name(&self) -> &str {
        "ColumnComparison"
    }
    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::ColumnComparison {
            left: self.left.clone(),
            op: self.op,
            right: self.right.clone(),
        }
    }
    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let left = match df.column(&self.left) {
            Ok(c) => c.as_materialized_series().clone(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.left))],
        };
        let right = match df.column(&self.right) {
            Ok(c) => c.as_materialized_series().clone(),
            Err(_) => return vec![missing_column_error(std::slice::from_ref(&self.right))],
        };

        // String comparison path.
        if let (Ok(left_str), Ok(right_str)) = (left.str(), right.str()) {
            let mut failed = Vec::new();
            for (i, (l, r)) in left_str.iter().zip(right_str.iter()).enumerate() {
                let (Some(l), Some(r)) = (l, r) else { continue };
                let ok = match self.op {
                    CompareOp::Lt => l < r,
                    CompareOp::Le => l <= r,
                    CompareOp::Gt => l > r,
                    CompareOp::Ge => l >= r,
                    CompareOp::Eq => l == r,
                    CompareOp::Ne => l != r,
                };
                if !ok {
                    failed.push(i);
                }
            }
            return comparison_result(self, failed);
        }

        // Numeric comparison path (ints, floats, dates, datetimes).
        let (Ok(left_num), Ok(right_num)) = (numeric_series(&left), numeric_series(&right)) else {
            return vec![ValidationError::multi(
                vec![self.left.clone(), self.right.clone()],
                self.name(),
                vec![],
                format!(
                    "Cannot compare '{}' and '{}': need two numeric or two string columns",
                    self.left, self.right
                ),
            )];
        };
        let left_vals: Vec<Option<f64>> = left_num.iter().collect();
        let right_vals: Vec<Option<f64>> = right_num.iter().collect();
        let mut failed = Vec::new();
        for (i, (l, r)) in left_vals.iter().zip(right_vals.iter()).enumerate() {
            let (Some(l), Some(r)) = (l, r) else { continue };
            let ok = match self.op {
                CompareOp::Lt => l < r,
                CompareOp::Le => l <= r,
                CompareOp::Gt => l > r,
                CompareOp::Ge => l >= r,
                // Exact float equality is intentional: cross-column checks
                // assert identical stored values, not approximate math.
                CompareOp::Eq => l == r,
                CompareOp::Ne => l != r,
            };
            if !ok {
                failed.push(i);
            }
        }
        comparison_result(self, failed)
    }
}

fn comparison_result(check: &ColumnComparison, failed: Vec<usize>) -> Vec<ValidationError> {
    if failed.is_empty() {
        Vec::new()
    } else {
        vec![ValidationError::multi(
            vec![check.left.clone(), check.right.clone()],
            check.name(),
            failed.clone(),
            format!(
                "Condition '{} {} {}' failed for {} row(s) at {:?}",
                check.left,
                check.op.as_str(),
                check.right,
                failed.len(),
                failed
            ),
        )]
    }
}

/// Checks that the combination of columns is unique across rows.
pub struct UniqueTogether {
    pub columns: Vec<String>,
}

impl DataFrameCheck for UniqueTogether {
    fn name(&self) -> &str {
        "UniqueTogether"
    }
    fn spec(&self) -> DataFrameCheckSpec {
        DataFrameCheckSpec::UniqueTogether {
            columns: self.columns.clone(),
        }
    }
    fn validate(&self, df: &DataFrame) -> Vec<ValidationError> {
        let mut selected = Vec::with_capacity(self.columns.len());
        for name in &self.columns {
            match df.column(name) {
                Ok(c) => selected.push(c.as_materialized_series().clone()),
                Err(_) => return vec![missing_column_error(std::slice::from_ref(name))],
            }
        }
        let height = df.height();
        let frame = match DataFrame::new(height, selected.into_iter().map(Column::from).collect()) {
            Ok(frame) => frame,
            Err(error) => {
                return vec![ValidationError::multi(
                    self.columns.clone(),
                    self.name(),
                    vec![],
                    format!("UniqueTogether failed to build comparison frame: {error}"),
                )];
            }
        };
        let mask = match frame.is_duplicated() {
            Ok(mask) => mask,
            Err(error) => {
                return vec![ValidationError::multi(
                    self.columns.clone(),
                    self.name(),
                    vec![],
                    format!("UniqueTogether failed: {error}"),
                )];
            }
        };
        if mask.any() {
            let rows = get_true_indices(&mask);
            vec![ValidationError::multi(
                self.columns.clone(),
                self.name(),
                rows.clone(),
                format!(
                    "Duplicate combinations of ({}) in {} row(s) at {:?}",
                    self.columns.join(", "),
                    rows.len(),
                    rows
                ),
            )]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> DataFrame {
        DataFrame::new(
            3,
            vec![
                Series::new("start".into(), [1i64, 2, 3]).into(),
                Series::new("end".into(), [2i64, 2, 4]).into(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn comparison_reports_failing_rows() {
        let check = ColumnComparison {
            left: "start".to_string(),
            op: CompareOp::Lt,
            right: "end".to_string(),
        };
        let errors = check.validate(&frame());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![1]);
        assert_eq!(
            errors[0].columns,
            vec!["start".to_string(), "end".to_string()]
        );
    }

    #[test]
    fn comparison_skips_nulls_and_reports_missing() {
        let df = DataFrame::new(
            2,
            vec![
                Series::new("a".into(), [Some(1i64), None]).into(),
                Series::new("b".into(), [Some(0i64), Some(5)]).into(),
            ],
        )
        .unwrap();
        let check = ColumnComparison {
            left: "a".to_string(),
            op: CompareOp::Gt,
            right: "b".to_string(),
        };
        let errors = check.validate(&df);
        assert!(errors.is_empty());

        let missing = ColumnComparison {
            left: "a".to_string(),
            op: CompareOp::Gt,
            right: "nope".to_string(),
        };
        assert_eq!(missing.validate(&df)[0].check, "ColumnExists");
    }

    #[test]
    fn unique_together_reports_duplicates() {
        let df = DataFrame::new(
            3,
            vec![
                Series::new("a".into(), ["x", "x", "y"]).into(),
                Series::new("b".into(), [1i64, 1, 2]).into(),
            ],
        )
        .unwrap();
        let check = UniqueTogether {
            columns: vec!["a".to_string(), "b".to_string()],
        };
        let errors = check.validate(&df);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].rows, vec![0, 1]);
    }
}
