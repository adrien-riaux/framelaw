#[derive(Clone)]
pub struct ValidationError {
    pub columns: Vec<String>,
    pub check: String,
    pub rows: Vec<usize>,
    pub message: String,
}

impl ValidationError {
    pub fn single(
        column: impl Into<String>,
        check: impl Into<String>,
        rows: Vec<usize>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            columns: vec![column.into()],
            check: check.into(),
            rows,
            message: message.into(),
        }
    }

    pub fn multi(
        columns: Vec<String>,
        check: impl Into<String>,
        rows: Vec<usize>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            columns,
            check: check.into(),
            rows,
            message: message.into(),
        }
    }

    pub fn primary_column(&self) -> &str {
        self.columns.first().map(String::as_str).unwrap_or("")
    }
}

pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
}

/// Row indices shown per error line before the rest collapses into
/// `+N more`; the full list stays available on each `ValidationError`.
const MAX_REPORTED_ROWS: usize = 5;

/// Renders row indices for a summary line: `rows 1, 3`, or
/// `rows 1, 3 +2 more` when indices were hidden. Empty lists (schema-level
/// errors) render as `schema`.
fn render_rows(rows: &[usize]) -> String {
    if rows.is_empty() {
        return "schema".to_string();
    }
    let shown: Vec<String> = rows
        .iter()
        .take(MAX_REPORTED_ROWS)
        .map(|row| row.to_string())
        .collect();
    let mut text = format!("rows {}", shown.join(", "));
    if rows.len() > MAX_REPORTED_ROWS {
        text.push_str(&format!(" +{} more", rows.len() - MAX_REPORTED_ROWS));
    }
    text
}

/// Formats a failure report as one header plus one line per error.
/// Location, row list, message, and public check name are separated so
/// none of them has to repeat the others.
pub fn format_summary(errors: &[ValidationError]) -> String {
    let count = errors.len();
    let header = format!(
        "Validation failed with {} error{}",
        count,
        if count == 1 { "" } else { "s" }
    );
    let lines: Vec<String> = errors
        .iter()
        .map(|error| {
            let location = match error.columns.as_slice() {
                [] => "Schema".to_string(),
                [single] => format!("Column '{single}'"),
                many => format!("Columns '{}'", many.join("', '")),
            };
            format!(
                "- {} ({}): {} [{}]",
                location,
                render_rows(&error.rows),
                error.message,
                error.check
            )
        })
        .collect();
    std::iter::once(header.to_string())
        .chain(lines)
        .collect::<Vec<_>>()
        .join("\n")
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn summary(&self) -> String {
        format_summary(&self.errors)
    }

    pub fn raise_if_invalid(&self) -> Result<(), String> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(self.summary())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_declares_the_error_count() {
        let report = ValidationReport {
            errors: vec![ValidationError::single(
                "age",
                "gt",
                vec![2, 5],
                "value must be > 0",
            )],
        };
        assert_eq!(
            report.summary(),
            "Validation failed with 1 error\n\
             - Column 'age' (rows 2, 5): value must be > 0 [gt]"
        );
    }

    #[test]
    fn plural_header_and_multi_column_locations() {
        let report = ValidationReport {
            errors: vec![
                ValidationError::single("age", "gt", vec![0], "value must be > 0"),
                ValidationError::multi(
                    vec!["start".to_string(), "end".to_string()],
                    "lt_column",
                    vec![1],
                    "condition 'start < end' failed for 1 row(s)",
                ),
            ],
        };
        assert_eq!(
            report.summary(),
            "Validation failed with 2 errors\n\
             - Column 'age' (rows 0): value must be > 0 [gt]\n\
             - Columns 'start', 'end' (rows 1): condition 'start < end' failed for 1 row(s) [lt_column]"
        );
    }

    #[test]
    fn row_lists_collapse_beyond_the_cap() {
        let rows: Vec<usize> = (0..10).collect();
        let report = ValidationReport {
            errors: vec![ValidationError::single(
                "age",
                "not_null",
                rows,
                "found 10 null value(s)",
            )],
        };
        assert!(report.summary().starts_with(
            "Validation failed with 1 error\n\
             - Column 'age' (rows 0, 1, 2, 3, 4 +5 more)"
        ));
    }

    #[test]
    fn schema_level_errors_omit_the_row_list() {
        let report = ValidationReport {
            errors: vec![ValidationError::single(
                "age",
                "dtype",
                vec![],
                "column has dtype String, expected Int64",
            )],
        };
        assert_eq!(
            report.summary(),
            "Validation failed with 1 error\n\
             - Column 'age' (schema): column has dtype String, expected Int64 [dtype]"
        );
    }
}
