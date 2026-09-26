use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use pyo3_polars::{PyDataFrame, PyDataType};
use regex::Regex;

use framelaw_core::checks::{
    BetweenF64, Check, GreaterThanF64, LessThanF64, MatchesRegex, NotNull, OneOfString,
    UniqueValues,
};
use framelaw_core::column::ColumnSpec;
use framelaw_core::schema::Schema;

#[derive(Clone)]
enum CheckType {
    NotNull,
    UniqueValues,
    GreaterThan(f64),
    LessThan(f64),
    BetweenF64(f64, f64),
    MatchesRegex(String),
    OneOfString(Vec<String>),
}

#[pyclass(module = "framelaw", name = "Check", from_py_object)]
#[derive(Clone)]
pub struct PyCheck {
    inner: CheckType,
}

impl PyCheck {
    fn into_check(self) -> PyResult<Box<dyn Check>> {
        Ok(match self.inner {
            CheckType::NotNull => Box::new(NotNull),
            CheckType::UniqueValues => Box::new(UniqueValues),
            CheckType::GreaterThan(v) => Box::new(GreaterThanF64 { min_value: v }),
            CheckType::LessThan(v) => Box::new(LessThanF64 { max_value: v }),
            CheckType::BetweenF64(a, b) => Box::new(BetweenF64 {
                min_value: a,
                max_value: b,
            }),
            CheckType::MatchesRegex(s) => Box::new(MatchesRegex {
                pattern: Regex::new(&s).map_err(|error| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid regular expression: {error}"
                    ))
                })?,
            }),
            CheckType::OneOfString(v) => Box::new(OneOfString {
                allowed_values: v.into_iter().collect(),
            }),
        })
    }
}

#[pyfunction]
pub fn not_null() -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::NotNull,
    })
}

#[pyfunction]
pub fn unique_values() -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::UniqueValues,
    })
}

#[pyfunction]
pub fn gt(value: f64) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::GreaterThan(value),
    })
}

#[pyfunction]
pub fn lt(value: f64) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::LessThan(value),
    })
}

#[pyfunction]
pub fn between(min: f64, max: f64) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::BetweenF64(min, max),
    })
}

#[pyfunction]
pub fn matches_regex(pattern: String) -> PyResult<PyCheck> {
    Regex::new(&pattern).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid regular expression: {error}"))
    })?;

    Ok(PyCheck {
        inner: CheckType::MatchesRegex(pattern),
    })
}

#[pyfunction]
pub fn is_in(allowed: Vec<String>) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::OneOfString(allowed),
    })
}

#[pyfunction(name = "matches")]
fn matches_check(pattern: String) -> PyResult<PyCheck> {
    matches_regex(pattern)
}

#[pyfunction(name = "one_of")]
fn one_of(allowed: Vec<String>) -> PyResult<PyCheck> {
    is_in(allowed)
}

#[pyfunction(name = "unique")]
fn unique() -> PyResult<PyCheck> {
    unique_values()
}

#[pyclass(module = "framelaw", name = "Column", skip_from_py_object)]
pub struct PyColumn {
    #[pyo3(get)]
    dtype: Py<PyAny>,
    #[pyo3(get)]
    nullable: bool,
    #[pyo3(get)]
    checks: Vec<PyCheck>,
}

#[pymethods]
impl PyColumn {
    #[new]
    #[pyo3(signature = (dtype, nullable = true, checks = None))]
    fn new(dtype: Py<PyAny>, nullable: bool, checks: Option<Vec<PyCheck>>) -> Self {
        PyColumn {
            dtype,
            nullable,
            checks: checks.unwrap_or_default(),
        }
    }
}

#[pyclass(module = "framelaw", name = "Schema")]
pub struct PySchema {
    columns: Vec<(String, PyColumn)>,
}

#[pyclass(module = "framelaw", name = "ValidationError", skip_from_py_object)]
#[derive(Clone)]
pub struct PyValidationError {
    #[pyo3(get)]
    column: String,
    #[pyo3(get)]
    check: String,
    #[pyo3(get)]
    rows: Vec<usize>,
    #[pyo3(get)]
    message: String,
}

#[pyclass(module = "framelaw", name = "ValidationReport")]
pub struct PyValidationReport {
    #[pyo3(get)]
    errors: Vec<PyValidationError>,
}

#[pymethods]
impl PyValidationReport {
    fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    fn summary(&self) -> String {
        self.errors
            .iter()
            .map(|error| {
                format!(
                    "Column '{}': rows {:?}, check '{}': {}",
                    error.column, error.rows, error.check, error.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl From<framelaw_core::errors::ValidationReport> for PyValidationReport {
    fn from(report: framelaw_core::errors::ValidationReport) -> Self {
        Self {
            errors: report
                .errors
                .into_iter()
                .map(|error| PyValidationError {
                    column: error.column,
                    check: error.check,
                    rows: error.rows,
                    message: error.message,
                })
                .collect(),
        }
    }
}

#[pymethods]
impl PySchema {
    #[new]
    fn new(columns: Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cols = Vec::new();
        for (k, v) in columns.iter() {
            let name = k.extract::<String>()?;
            let entry = v.cast::<PyAny>()?;

            let dtype = entry.getattr("dtype")?;
            let nullable = entry.getattr("nullable")?.extract::<bool>()?;
            let checks = match entry.getattr("checks") {
                Ok(value) => {
                    if value.is_none() {
                        Vec::new()
                    } else {
                        value.extract::<Vec<PyCheck>>()?
                    }
                }
                Err(_) => Vec::new(),
            };

            cols.push((
                name,
                PyColumn {
                    dtype: dtype.unbind(),
                    nullable,
                    checks,
                },
            ));
        }
        Ok(PySchema { columns: cols })
    }

    #[pyo3(signature = (df, raise_on_error = true))]
    fn validate(
        &self,
        py: Python<'_>,
        df: PyDataFrame,
        raise_on_error: bool,
    ) -> PyResult<Option<PyValidationReport>> {
        let mut specs = Vec::new();
        for (name, col) in &self.columns {
            let checks: Vec<Box<dyn Check>> = col
                .checks
                .iter()
                .cloned()
                .map(|check| check.into_check())
                .collect::<PyResult<_>>()?;

            let dtype = col.dtype.bind(py).extract::<PyDataType>()?.0;

            specs.push(ColumnSpec {
                name: name.clone(),
                dtype,
                nullable: col.nullable,
                checks,
            });
        }

        let rust_schema = Schema::new(specs);
        let polars_df = df.into();
        let report = rust_schema.validate(&polars_df);

        if report.is_valid() {
            Ok(None)
        } else if raise_on_error {
            use pyo3::exceptions::PyValueError;
            let report = PyValidationReport::from(report);
            Err(PyValueError::new_err(format!(
                "Validation failed:\n{}",
                report.summary()
            )))
        } else {
            Ok(Some(PyValidationReport::from(report)))
        }
    }
}

#[pymodule]
fn _framelaw(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCheck>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PySchema>()?;
    m.add_class::<PyValidationError>()?;
    m.add_class::<PyValidationReport>()?;

    m.add_function(wrap_pyfunction!(not_null, m)?)?;
    m.add_function(wrap_pyfunction!(unique_values, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(between, m)?)?;
    m.add_function(wrap_pyfunction!(matches_regex, m)?)?;
    m.add_function(wrap_pyfunction!(is_in, m)?)?;
    m.add_function(wrap_pyfunction!(matches_check, m)?)?;
    m.add_function(wrap_pyfunction!(one_of, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;

    Ok(())
}
