use framelaw_core::checks::{
    BetweenF64, Check, GreaterThanF64, LessThanF64, MatchesRegex, NotNull, OneOfString,
    UniqueValues,
};
use framelaw_core::column::ColumnSpec;
use framelaw_core::schema::Schema;
use polars::datatypes::DataType;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_polars::PyDataFrame;

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

#[pyclass(module = "framelaw", name = "Check")]
#[derive(Clone)]
pub struct PyCheck {
    inner: CheckType,
}

impl PyCheck {
    fn into_check(self) -> Box<dyn Check> {
        match self.inner {
            CheckType::NotNull => Box::new(NotNull),
            CheckType::UniqueValues => Box::new(UniqueValues),
            CheckType::GreaterThan(v) => Box::new(GreaterThanF64(v)),
            CheckType::LessThan(v) => Box::new(LessThanF64(v)),
            CheckType::BetweenF64(a, b) => Box::new(BetweenF64 {
                min_value: a,
                max_value: b,
            }),
            CheckType::MatchesRegex(s) => Box::new(MatchesRegex::new(&s).unwrap()),
            CheckType::OneOfString(v) => Box::new(OneOfString {
                allowed_values: v.into_iter().collect(),
            }),
        }
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

#[pyclass(module = "framelaw", name = "Column")]
#[derive(Clone)]
pub struct PyColumn {
    dtype: PyObject,
    nullable: bool,
    checks: Vec<PyCheck>,
}

#[pymethods]
impl PyColumn {
    #[new]
    #[pyo3(signature = (dtype, nullable=true, checks=None))]
    fn new(dtype: PyObject, nullable: bool, checks: Option<Vec<PyCheck>>) -> Self {
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

#[pymethods]
impl PySchema {
    #[new]
    fn new(columns: Bound<'_, PyDict>) -> PyResult<Self> {
        let mut cols = Vec::new();
        for (k, v) in columns.iter() {
            let name = k.extract::<String>()?;
            let col = v.extract::<PyColumn>()?;
            cols.push((name, col));
        }
        Ok(PySchema { columns: cols })
    }

    fn validate(&self, df: PyDataFrame) -> PyResult<()> {
        let mut specs = Vec::new();
        for (name, col) in &self.columns {
            let checks: Vec<Box<dyn Check>> =
                col.checks.iter().map(|c| c.clone().into_check()).collect();
            specs.push(ColumnSpec {
                name: name.clone(),
                dtype: DataType::Unknown(polars::datatypes::UnknownKind::Any),
                nullable: col.nullable,
                checks,
            });
        }
        let rust_schema = Schema::new(specs);
        let polars_df = df.into(); // PyDataFrame into polars::DataFrame
        let report = rust_schema.validate(&polars_df);

        if report.is_valid {
            Ok(())
        } else {
            use pyo3::exceptions::PyValueError;
            let mut msgs = Vec::new();
            for f in report.failures {
                msgs.push(format!(
                    "Column {}: row {}, error: {}",
                    f.column_name, f.row_index, f.message
                ));
            }
            Err(PyValueError::new_err(format!(
                "Validation failed:\n{}",
                msgs.join("\n")
            )))
        }
    }
}

#[pymodule]
fn framelaw(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCheck>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PySchema>()?;

    m.add_function(wrap_pyfunction!(not_null, m)?)?;
    m.add_function(wrap_pyfunction!(unique_values, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(between, m)?)?;
    m.add_function(wrap_pyfunction!(matches_regex, m)?)?;
    m.add_function(wrap_pyfunction!(is_in, m)?)?;

    Ok(())
}
