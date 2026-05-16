pub struct ValidationError {
    pub column: String,
    pub check: String,
    pub rows: Vec<usize>,
    pub message: String,
}

pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}
