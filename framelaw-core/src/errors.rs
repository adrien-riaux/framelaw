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

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn summary(&self) -> String {
        self.errors
            .iter()
            .map(|error| {
                let location = if error.columns.len() > 1 {
                    format!("Columns '{}'", error.columns.join("', '"))
                } else {
                    format!("Column '{}'", error.primary_column())
                };
                format!(
                    "{}: rows {:?}, check '{}': {}",
                    location, error.rows, error.check, error.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn raise_if_invalid(&self) -> Result<(), String> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(format!("Validation failed:\n{}", self.summary()))
        }
    }
}
