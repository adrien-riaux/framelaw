use polars::prelude::{DataFrame, NamedFrom, Series};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};
use pyo3_polars::{PyDataFrame, PyDataType};
use regex::Regex;
use serde::{Deserialize, Serialize};

use framelaw_core::checks::{
    BetweenF64, Check, CheckSpec, GreaterThanF64, GreaterThanOrEqualF64, LessThanF64,
    LessThanOrEqualF64, ListLenBetween, ListNotEmpty, MatchesRegex, NotNull, OneOfString,
    StructFieldCheck, UniqueValues,
};
use framelaw_core::column::ColumnSpec;
use framelaw_core::dataframe_checks::{
    ColumnComparison, CompareOp, DataFrameCheck, DataFrameCheckSpec, UniqueTogether,
};
use framelaw_core::inference::infer_schema as infer_core_schema;
use framelaw_core::schema::Schema;

#[derive(Clone)]
enum CheckType {
    NotNull,
    UniqueValues,
    GreaterThan(f64),
    LessThan(f64),
    GreaterThanOrEqual(f64),
    LessThanOrEqual(f64),
    BetweenF64(f64, f64),
    MatchesRegex(String),
    OneOfString(Vec<String>),
    ListLenBetween(usize, usize),
    ListNotEmpty,
    StructField(String, Vec<PyCheck>),
}

#[pyclass(module = "framelaw", name = "Check", from_py_object)]
#[derive(Clone)]
pub struct PyCheck {
    inner: CheckType,
}

impl PyCheck {
    fn from_spec(spec: CheckSpec) -> Self {
        let inner = match spec {
            CheckSpec::NotNull => CheckType::NotNull,
            CheckSpec::UniqueValues => CheckType::UniqueValues,
            CheckSpec::GreaterThan(value) => CheckType::GreaterThan(value),
            CheckSpec::LessThan(value) => CheckType::LessThan(value),
            CheckSpec::GreaterThanOrEqual(value) => CheckType::GreaterThanOrEqual(value),
            CheckSpec::LessThanOrEqual(value) => CheckType::LessThanOrEqual(value),
            CheckSpec::Between(min, max) => CheckType::BetweenF64(min, max),
            CheckSpec::Matches(pattern) => CheckType::MatchesRegex(pattern),
            CheckSpec::OneOf(values) => CheckType::OneOfString(values),
            CheckSpec::ListLenBetween(lo, hi) => CheckType::ListLenBetween(lo, hi),
            CheckSpec::ListNotEmpty => CheckType::ListNotEmpty,
            CheckSpec::StructField(field, inner) => {
                CheckType::StructField(field, inner.into_iter().map(PyCheck::from_spec).collect())
            }
        };
        Self { inner }
    }

    fn into_check(self) -> PyResult<Box<dyn Check>> {
        Ok(match self.inner {
            CheckType::NotNull => Box::new(NotNull),
            CheckType::UniqueValues => Box::new(UniqueValues),
            CheckType::GreaterThan(v) => Box::new(GreaterThanF64 { min_value: v }),
            CheckType::LessThan(v) => Box::new(LessThanF64 { max_value: v }),
            CheckType::GreaterThanOrEqual(v) => Box::new(GreaterThanOrEqualF64 { min_value: v }),
            CheckType::LessThanOrEqual(v) => Box::new(LessThanOrEqualF64 { max_value: v }),
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
            CheckType::ListLenBetween(lo, hi) => Box::new(ListLenBetween {
                min_len: lo,
                max_len: hi,
            }),
            CheckType::ListNotEmpty => Box::new(ListNotEmpty),
            CheckType::StructField(field, inner) => {
                let inner_checks = inner
                    .into_iter()
                    .map(PyCheck::into_check)
                    .collect::<PyResult<Vec<_>>>()?;
                Box::new(StructFieldCheck {
                    field,
                    inner_checks,
                })
            }
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
pub fn gte(value: f64) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::GreaterThanOrEqual(value),
    })
}

#[pyfunction]
pub fn lte(value: f64) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::LessThanOrEqual(value),
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

#[pyfunction]
fn list_len_between(min_len: usize, max_len: usize) -> PyResult<PyCheck> {
    if min_len > max_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_len must be <= max_len",
        ));
    }
    Ok(PyCheck {
        inner: CheckType::ListLenBetween(min_len, max_len),
    })
}

#[pyfunction]
fn list_not_empty() -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::ListNotEmpty,
    })
}

#[pyfunction]
fn struct_field(field: String, checks: Vec<PyCheck>) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::StructField(field, checks),
    })
}

#[derive(Clone)]
enum TableCheckType {
    Comparison {
        left: String,
        op: CompareOp,
        right: String,
    },
    UniqueTogether {
        columns: Vec<String>,
    },
}

#[pyclass(module = "framelaw", name = "DataFrameCheck", from_py_object)]
#[derive(Clone)]
pub struct PyDataFrameCheck {
    inner: TableCheckType,
}

impl PyDataFrameCheck {
    fn from_spec(spec: DataFrameCheckSpec) -> Self {
        let inner = match spec {
            DataFrameCheckSpec::ColumnComparison { left, op, right } => {
                TableCheckType::Comparison { left, op, right }
            }
            DataFrameCheckSpec::UniqueTogether { columns } => {
                TableCheckType::UniqueTogether { columns }
            }
        };
        Self { inner }
    }

    fn into_check(self) -> Box<dyn DataFrameCheck> {
        match self.inner {
            TableCheckType::Comparison { left, op, right } => {
                Box::new(ColumnComparison { left, op, right })
            }
            TableCheckType::UniqueTogether { columns } => Box::new(UniqueTogether { columns }),
        }
    }
}

fn comparison(left: String, op: CompareOp, right: String) -> PyResult<PyDataFrameCheck> {
    if left == right {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cross-column checks need two different columns",
        ));
    }
    Ok(PyDataFrameCheck {
        inner: TableCheckType::Comparison { left, op, right },
    })
}

#[pyfunction]
fn gt_column(left: String, right: String) -> PyResult<PyDataFrameCheck> {
    comparison(left, CompareOp::Gt, right)
}

#[pyfunction]
fn lt_column(left: String, right: String) -> PyResult<PyDataFrameCheck> {
    comparison(left, CompareOp::Lt, right)
}

#[pyfunction]
fn gte_column(left: String, right: String) -> PyResult<PyDataFrameCheck> {
    comparison(left, CompareOp::Ge, right)
}

#[pyfunction]
fn lte_column(left: String, right: String) -> PyResult<PyDataFrameCheck> {
    comparison(left, CompareOp::Le, right)
}

#[pyfunction]
fn eq_columns(left: String, right: String) -> PyResult<PyDataFrameCheck> {
    comparison(left, CompareOp::Eq, right)
}

#[pyfunction]
fn ne_columns(left: String, right: String) -> PyResult<PyDataFrameCheck> {
    comparison(left, CompareOp::Ne, right)
}

#[pyfunction]
fn unique_together(columns: Vec<String>) -> PyResult<PyDataFrameCheck> {
    if columns.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "unique_together needs at least two columns",
        ));
    }
    Ok(PyDataFrameCheck {
        inner: TableCheckType::UniqueTogether { columns },
    })
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
    table_checks: Vec<PyDataFrameCheck>,
    lazy: bool,
}

const JSON_SCHEMA_VERSION: u32 = 2;
const JSON_SCHEMA_MIN_VERSION: u32 = 1;

#[derive(Deserialize, Serialize)]
struct JsonSchema {
    version: u32,
    columns: Vec<JsonColumn>,
    #[serde(default)]
    table_checks: Vec<JsonTableCheck>,
    #[serde(default)]
    lazy: bool,
}

#[derive(Deserialize, Serialize)]
struct JsonColumn {
    name: String,
    dtype: String,
    nullable: bool,
    checks: Vec<JsonCheck>,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
enum JsonCheck {
    NotNull,
    Unique,
    GreaterThan {
        value: f64,
    },
    LessThan {
        value: f64,
    },
    GreaterThanOrEqual {
        value: f64,
    },
    LessThanOrEqual {
        value: f64,
    },
    Between {
        min: f64,
        max: f64,
    },
    Matches {
        pattern: String,
    },
    OneOf {
        values: Vec<String>,
    },
    ListLenBetween {
        min_len: usize,
        max_len: usize,
    },
    ListNotEmpty,
    StructField {
        field: String,
        checks: Vec<JsonCheck>,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
enum JsonTableCheck {
    ColumnComparison {
        left: String,
        op: String,
        right: String,
    },
    UniqueTogether {
        columns: Vec<String>,
    },
}

#[pyclass(module = "framelaw", name = "ColumnRef", skip_from_py_object)]
pub struct PyColumnRef {
    schema: Py<PySchema>,
    name: String,
}

#[pyclass(module = "framelaw", name = "ValidationError", skip_from_py_object)]
#[derive(Clone)]
pub struct PyValidationError {
    #[pyo3(get)]
    column: String,
    #[pyo3(get)]
    columns: Vec<String>,
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

    fn raise_if_invalid(&self) -> PyResult<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Validation failed:\n{}",
                self.summary()
            )))
        }
    }

    fn summary(&self) -> String {
        self.errors
            .iter()
            .map(|error| {
                let location = if error.columns.len() > 1 {
                    format!("Columns '{}'", error.columns.join("', '"))
                } else {
                    format!("Column '{}'", error.column)
                };
                format!(
                    "{}: rows {:?}, check '{}': {}",
                    location, error.rows, error.check, error.message
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn to_dataframe(&self) -> PyResult<PyDataFrame> {
        let mut columns = Vec::new();
        let mut all_columns = Vec::new();
        let mut checks = Vec::new();
        let mut rows = Vec::new();
        let mut messages = Vec::new();

        for error in &self.errors {
            let joined = error.columns.join("+");
            if error.rows.is_empty() {
                columns.push(error.column.clone());
                all_columns.push(joined);
                checks.push(error.check.clone());
                rows.push(None::<u64>);
                messages.push(error.message.clone());
                continue;
            }

            for row in &error.rows {
                columns.push(error.column.clone());
                all_columns.push(joined.clone());
                checks.push(error.check.clone());
                rows.push(Some(*row as u64));
                messages.push(error.message.clone());
            }
        }

        let height = columns.len();
        let dataframe = DataFrame::new(
            height,
            vec![
                Series::new("column".into(), columns).into(),
                Series::new("columns".into(), all_columns).into(),
                Series::new("check".into(), checks).into(),
                Series::new("row".into(), rows).into(),
                Series::new("message".into(), messages).into(),
            ],
        )
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;

        Ok(PyDataFrame(dataframe))
    }
}

impl From<framelaw_core::errors::ValidationReport> for PyValidationReport {
    fn from(report: framelaw_core::errors::ValidationReport) -> Self {
        Self {
            errors: report
                .errors
                .into_iter()
                .map(|error| {
                    let column = error.columns.first().cloned().unwrap_or_default();
                    PyValidationError {
                        column,
                        columns: error.columns,
                        check: error.check,
                        rows: error.rows,
                        message: error.message,
                    }
                })
                .collect(),
        }
    }
}

impl PySchema {
    fn core_schema_value<'py>(
        schema: &'py Bound<'py, PyDict>,
        key: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        schema.get_item(key)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Pydantic core schema is missing '{key}'"
            ))
        })
    }

    fn check_from_core_schema(
        schema: &Bound<'_, PyDict>,
        checks: &mut Vec<PyCheck>,
        is_list: bool,
    ) -> PyResult<()> {
        if let Some(value) = schema.get_item("gt")? {
            checks.push(gt(value.extract()?)?);
        }
        if let Some(value) = schema.get_item("lt")? {
            checks.push(lt(value.extract()?)?);
        }
        if let Some(value) = schema.get_item("ge")? {
            checks.push(gte(value.extract()?)?);
        }
        if let Some(value) = schema.get_item("le")? {
            checks.push(lte(value.extract()?)?);
        }
        if let Some(value) = schema.get_item("pattern")? {
            checks.push(matches_regex(value.extract()?)?);
        }
        // Length constraints only exist on list schemas; string fields also
        // carry min/max_length, which has no list-length meaning.
        if !is_list {
            return Ok(());
        }
        let min_len: Option<usize> = schema
            .get_item("min_length")?
            .map(|v| v.extract())
            .transpose()?;
        let max_len: Option<usize> = schema
            .get_item("max_length")?
            .map(|v| v.extract())
            .transpose()?;
        if min_len.is_some() || max_len.is_some() {
            checks.push(list_len_between(
                min_len.unwrap_or(0),
                max_len.unwrap_or(usize::MAX),
            )?);
        }
        Ok(())
    }

    fn pydantic_list_dtype(
        py: Python<'_>,
        field_schema: &Bound<'_, PyDict>,
    ) -> PyResult<PyDataType> {
        let _ = py;
        let items = Self::core_schema_value(field_schema, "items_schema")?
            .cast::<PyDict>()?
            .clone();
        let inner_type = Self::core_schema_value(&items, "type")?.extract::<String>()?;
        let inner = match inner_type.as_str() {
            "int" => polars::prelude::DataType::Int64,
            "float" => polars::prelude::DataType::Float64,
            "str" => polars::prelude::DataType::String,
            "bool" => polars::prelude::DataType::Boolean,
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported Pydantic list item type in core schema: '{other}'"
                )))
            }
        };
        Ok(PyDataType(polars::prelude::DataType::List(Box::new(inner))))
    }

    fn from_pydantic_core_schema(py: Python<'_>, model: &Bound<'_, PyAny>) -> PyResult<Self> {
        let root = model
            .getattr("__pydantic_core_schema__")?
            .cast::<PyDict>()?
            .clone();
        let model_fields = Self::core_schema_value(&root, "schema")?
            .cast::<PyDict>()?
            .clone();
        let fields = Self::core_schema_value(&model_fields, "fields")?
            .cast::<PyDict>()?
            .clone();
        let mut columns = Vec::with_capacity(fields.len());

        for (name, field) in fields.iter() {
            let name = name.extract::<String>()?;
            let field = field.cast::<PyDict>()?;
            let initial_schema = Self::core_schema_value(field, "schema")?
                .cast::<PyDict>()?
                .clone();
            let mut field_schema_owned = initial_schema.unbind();
            let mut nullable = false;
            loop {
                let field_schema = field_schema_owned.bind(py);
                let field_type =
                    Self::core_schema_value(field_schema, "type")?.extract::<String>()?;
                if field_type == "nullable" {
                    nullable = true;
                } else if field_type != "default" {
                    break;
                }
                let inner_schema = Self::core_schema_value(field_schema, "schema")?
                    .cast::<PyDict>()?
                    .clone();
                field_schema_owned = inner_schema.unbind();
            }
            let field_schema = field_schema_owned.bind(py);

            let field_type = Self::core_schema_value(field_schema, "type")?.extract::<String>()?;
            let (dtype, mut checks) = match field_type.as_str() {
                "int" => (PyDataType(polars::prelude::DataType::Int64), Vec::new()),
                "float" => (PyDataType(polars::prelude::DataType::Float64), Vec::new()),
                "str" => (PyDataType(polars::prelude::DataType::String), Vec::new()),
                "bool" => (PyDataType(polars::prelude::DataType::Boolean), Vec::new()),
                "list" => (Self::pydantic_list_dtype(py, field_schema)?, Vec::new()),
                "literal" => {
                    let expected = Self::core_schema_value(field_schema, "expected")?
                        .extract::<Vec<String>>()?;
                    (
                        PyDataType(polars::prelude::DataType::String),
                        vec![is_in(expected)?],
                    )
                }
                other => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unsupported Pydantic field type in core schema: '{other}'"
                    )))
                }
            };
            Self::check_from_core_schema(field_schema, &mut checks, field_type.as_str() == "list")?;
            if !nullable && !matches!(field_type.as_str(), "nullable") {
                checks.push(not_null()?);
            }

            columns.push((
                name,
                PyColumn {
                    dtype: dtype.into_pyobject(py)?.into_any().unbind(),
                    nullable,
                    checks,
                },
            ));
        }

        Ok(Self {
            columns,
            table_checks: Vec::new(),
            lazy: false,
        })
    }

    fn dtype_name(dtype: &polars::prelude::DataType) -> PyResult<String> {
        let name = match dtype {
            polars::prelude::DataType::Boolean => "Boolean".to_string(),
            polars::prelude::DataType::Int8 => "Int8".to_string(),
            polars::prelude::DataType::Int16 => "Int16".to_string(),
            polars::prelude::DataType::Int32 => "Int32".to_string(),
            polars::prelude::DataType::Int64 => "Int64".to_string(),
            polars::prelude::DataType::Int128 => "Int128".to_string(),
            polars::prelude::DataType::UInt8 => "UInt8".to_string(),
            polars::prelude::DataType::UInt16 => "UInt16".to_string(),
            polars::prelude::DataType::UInt32 => "UInt32".to_string(),
            polars::prelude::DataType::UInt64 => "UInt64".to_string(),
            polars::prelude::DataType::UInt128 => "UInt128".to_string(),
            polars::prelude::DataType::Float16 => "Float16".to_string(),
            polars::prelude::DataType::Float32 => "Float32".to_string(),
            polars::prelude::DataType::Float64 => "Float64".to_string(),
            polars::prelude::DataType::String => "String".to_string(),
            polars::prelude::DataType::Binary => "Binary".to_string(),
            polars::prelude::DataType::Null => "Null".to_string(),
            polars::prelude::DataType::Date => "Date".to_string(),
            polars::prelude::DataType::Time => "Time".to_string(),
            polars::prelude::DataType::Duration(unit) => format!("Duration:{unit:?}"),
            polars::prelude::DataType::Datetime(unit, None) => format!("Datetime:{unit:?}"),
            polars::prelude::DataType::Datetime(_, Some(timezone)) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "JSON serialization does not support timezone-aware dtype {timezone}"
                )))
            }
            polars::prelude::DataType::List(inner) => {
                format!("List:{}", Self::dtype_name(inner)?)
            }
            polars::prelude::DataType::Array(inner, width) => {
                format!("Array:{}:{}", Self::dtype_name(inner)?, width)
            }
            polars::prelude::DataType::Struct(fields) => {
                let parts = fields
                    .iter()
                    .map(|field| {
                        Ok(format!(
                            "{}={}",
                            field.name,
                            Self::dtype_name(&field.dtype)?
                        ))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                format!("Struct:{}", parts.join(","))
            }
            other => {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "JSON serialization does not support Polars dtype {other}"
                )))
            }
        };
        Ok(name)
    }

    fn scalar_dtype_from_name(name: &str) -> Option<polars::prelude::DataType> {
        Some(match name {
            "Boolean" => polars::prelude::DataType::Boolean,
            "Int8" => polars::prelude::DataType::Int8,
            "Int16" => polars::prelude::DataType::Int16,
            "Int32" => polars::prelude::DataType::Int32,
            "Int64" => polars::prelude::DataType::Int64,
            "Int128" => polars::prelude::DataType::Int128,
            "UInt8" => polars::prelude::DataType::UInt8,
            "UInt16" => polars::prelude::DataType::UInt16,
            "UInt32" => polars::prelude::DataType::UInt32,
            "UInt64" => polars::prelude::DataType::UInt64,
            "UInt128" => polars::prelude::DataType::UInt128,
            "Float16" => polars::prelude::DataType::Float16,
            "Float32" => polars::prelude::DataType::Float32,
            "Float64" => polars::prelude::DataType::Float64,
            "String" => polars::prelude::DataType::String,
            "Binary" => polars::prelude::DataType::Binary,
            "Null" => polars::prelude::DataType::Null,
            "Date" => polars::prelude::DataType::Date,
            "Time" => polars::prelude::DataType::Time,
            _ => return None,
        })
    }

    fn dtype_from_name(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let dtype = Self::parse_dtype_name(name)?;
        Ok(PyDataType(dtype).into_pyobject(py)?.into_any().unbind())
    }

    /// Parses a dtype tag produced by `dtype_name`, recursing through nested
    /// `List`/`Array`/`Struct` tags to arbitrary depth.
    fn parse_dtype_name(name: &str) -> PyResult<polars::prelude::DataType> {
        if let Some(dtype) = Self::scalar_dtype_from_name(name) {
            return Ok(dtype);
        }
        if let Some(inner) = name.strip_prefix("List:") {
            return Ok(polars::prelude::DataType::List(Box::new(
                Self::parse_dtype_name(inner)?,
            )));
        }
        if let Some(rest) = name.strip_prefix("Array:") {
            let (inner_name, width) = rest.rsplit_once(':').ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported JSON Polars dtype '{name}'"
                ))
            })?;
            let width: usize = width.parse().map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported JSON Polars dtype '{name}'"
                ))
            })?;
            return Ok(polars::prelude::DataType::Array(
                Box::new(Self::parse_dtype_name(inner_name)?),
                width,
            ));
        }
        if let Some(rest) = name.strip_prefix("Struct:") {
            let mut fields = Vec::new();
            if !rest.is_empty() {
                for part in rest.split(',') {
                    let (field_name, field_dtype) = part.split_once('=').ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported JSON Polars dtype '{name}'"
                        ))
                    })?;
                    fields.push(polars::prelude::Field::new(
                        field_name.into(),
                        Self::parse_dtype_name(field_dtype)?,
                    ));
                }
            }
            return Ok(polars::prelude::DataType::Struct(fields));
        }
        if name.starts_with("Duration:") {
            return Ok(polars::prelude::DataType::Duration(Self::parse_time_unit(
                name,
            )?));
        }
        if name.starts_with("Datetime:") {
            return Ok(polars::prelude::DataType::Datetime(
                Self::parse_time_unit(name)?,
                None,
            ));
        }
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unsupported JSON Polars dtype '{name}'"
        )))
    }

    fn parse_time_unit(value: &str) -> PyResult<polars::prelude::TimeUnit> {
        match value
            .strip_prefix("Duration:")
            .or_else(|| value.strip_prefix("Datetime:"))
        {
            Some("Nanoseconds") => Ok(polars::prelude::TimeUnit::Nanoseconds),
            Some("Microseconds") => Ok(polars::prelude::TimeUnit::Microseconds),
            Some("Milliseconds") => Ok(polars::prelude::TimeUnit::Milliseconds),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported temporal JSON dtype '{value}'"
            ))),
        }
    }

    fn json_check(check: &PyCheck) -> JsonCheck {
        match &check.inner {
            CheckType::NotNull => JsonCheck::NotNull,
            CheckType::UniqueValues => JsonCheck::Unique,
            CheckType::GreaterThan(value) => JsonCheck::GreaterThan { value: *value },
            CheckType::LessThan(value) => JsonCheck::LessThan { value: *value },
            CheckType::GreaterThanOrEqual(value) => JsonCheck::GreaterThanOrEqual { value: *value },
            CheckType::LessThanOrEqual(value) => JsonCheck::LessThanOrEqual { value: *value },
            CheckType::BetweenF64(min, max) => JsonCheck::Between {
                min: *min,
                max: *max,
            },
            CheckType::MatchesRegex(pattern) => JsonCheck::Matches {
                pattern: pattern.clone(),
            },
            CheckType::OneOfString(values) => {
                let mut values = values.clone();
                values.sort_unstable();
                JsonCheck::OneOf { values }
            }
            CheckType::ListLenBetween(min_len, max_len) => JsonCheck::ListLenBetween {
                min_len: *min_len,
                max_len: *max_len,
            },
            CheckType::ListNotEmpty => JsonCheck::ListNotEmpty,
            CheckType::StructField(field, inner) => JsonCheck::StructField {
                field: field.clone(),
                checks: inner.iter().map(Self::json_check).collect(),
            },
        }
    }

    fn check_from_json(check: JsonCheck) -> PyResult<PyCheck> {
        match check {
            JsonCheck::NotNull => not_null(),
            JsonCheck::Unique => unique_values(),
            JsonCheck::GreaterThan { value } => gt(value),
            JsonCheck::LessThan { value } => lt(value),
            JsonCheck::GreaterThanOrEqual { value } => gte(value),
            JsonCheck::LessThanOrEqual { value } => lte(value),
            JsonCheck::Between { min, max } => between(min, max),
            JsonCheck::Matches { pattern } => matches_regex(pattern),
            JsonCheck::OneOf { values } => is_in(values),
            JsonCheck::ListLenBetween { min_len, max_len } => list_len_between(min_len, max_len),
            JsonCheck::ListNotEmpty => list_not_empty(),
            JsonCheck::StructField { field, checks } => {
                let inner = checks
                    .into_iter()
                    .map(Self::check_from_json)
                    .collect::<PyResult<Vec<_>>>()?;
                struct_field(field, inner)
            }
        }
    }

    fn json_table_check(check: &PyDataFrameCheck) -> JsonTableCheck {
        match &check.inner {
            TableCheckType::Comparison { left, op, right } => {
                let op_name = match op {
                    CompareOp::Lt => "lt",
                    CompareOp::Le => "le",
                    CompareOp::Gt => "gt",
                    CompareOp::Ge => "ge",
                    CompareOp::Eq => "eq",
                    CompareOp::Ne => "ne",
                };
                JsonTableCheck::ColumnComparison {
                    left: left.clone(),
                    op: op_name.to_string(),
                    right: right.clone(),
                }
            }
            TableCheckType::UniqueTogether { columns } => JsonTableCheck::UniqueTogether {
                columns: columns.clone(),
            },
        }
    }

    fn table_check_from_json(check: JsonTableCheck) -> PyResult<PyDataFrameCheck> {
        match check {
            JsonTableCheck::ColumnComparison { left, op, right } => match op.as_str() {
                "lt" => lt_column(left, right),
                "le" => lte_column(left, right),
                "gt" => gt_column(left, right),
                "ge" => gte_column(left, right),
                "eq" => eq_columns(left, right),
                "ne" => ne_columns(left, right),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unsupported table check op '{other}'"
                ))),
            },
            JsonTableCheck::UniqueTogether { columns } => unique_together(columns),
        }
    }

    fn from_dtype_dict(schema: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut columns = Vec::with_capacity(schema.len());
        for (name, dtype) in schema.iter() {
            let name = name.extract::<String>()?;
            let dtype = dtype.cast::<PyAny>()?;
            dtype.extract::<PyDataType>()?;
            columns.push((
                name,
                PyColumn {
                    dtype: dtype.clone().unbind(),
                    nullable: true,
                    checks: Vec::new(),
                },
            ));
        }
        Ok(Self {
            columns,
            table_checks: Vec::new(),
            lazy: false,
        })
    }

    fn from_core(py: Python<'_>, schema: Schema) -> PyResult<Self> {
        let columns = schema
            .columns
            .into_iter()
            .map(|column| {
                let ColumnSpec {
                    name,
                    dtype,
                    nullable,
                    checks,
                } = column;
                let checks = checks
                    .into_iter()
                    .map(|check| PyCheck::from_spec(check.spec()))
                    .collect();
                let dtype = PyDataType(dtype).into_pyobject(py)?.into_any().unbind();

                Ok((
                    name,
                    PyColumn {
                        dtype,
                        nullable,
                        checks,
                    },
                ))
            })
            .collect::<PyResult<_>>()?;
        let table_checks = schema
            .table_checks
            .into_iter()
            .map(|check| PyDataFrameCheck::from_spec(check.spec()))
            .collect();

        Ok(Self {
            columns,
            table_checks,
            lazy: false,
        })
    }

    fn from_json_value(py: Python<'_>, document: JsonSchema) -> PyResult<Self> {
        if document.version < JSON_SCHEMA_MIN_VERSION || document.version > JSON_SCHEMA_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported schema JSON version {}; expected {}",
                document.version, JSON_SCHEMA_VERSION
            )));
        }

        let columns = document
            .columns
            .into_iter()
            .map(|column| {
                let checks = column
                    .checks
                    .into_iter()
                    .map(Self::check_from_json)
                    .collect::<PyResult<_>>()?;
                Ok((
                    column.name,
                    PyColumn {
                        dtype: Self::dtype_from_name(py, &column.dtype)?,
                        nullable: column.nullable,
                        checks,
                    },
                ))
            })
            .collect::<PyResult<_>>()?;

        let table_checks = document
            .table_checks
            .into_iter()
            .map(Self::table_check_from_json)
            .collect::<PyResult<_>>()?;

        Ok(Self {
            columns,
            table_checks,
            lazy: document.lazy,
        })
    }

    fn core_schema(&self, py: Python<'_>) -> PyResult<Schema> {
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

        let table_checks = self
            .table_checks
            .iter()
            .cloned()
            .map(|check| check.into_check())
            .collect::<Vec<Box<dyn DataFrameCheck>>>();

        Ok(Schema::with_table_checks(specs, table_checks))
    }
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (columns, dataframe_checks=None, lazy=false))]
    fn new(
        columns: Bound<'_, PyDict>,
        dataframe_checks: Option<Vec<PyDataFrameCheck>>,
        lazy: bool,
    ) -> PyResult<Self> {
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
        Ok(PySchema {
            columns: cols,
            table_checks: dataframe_checks.unwrap_or_default(),
            lazy,
        })
    }

    #[staticmethod]
    fn from_polars_schema(schema: Bound<'_, PyDict>) -> PyResult<Self> {
        Self::from_dtype_dict(&schema)
    }

    fn to_json(&self, py: Python<'_>) -> PyResult<String> {
        let columns = self
            .columns
            .iter()
            .map(|(name, column)| {
                let dtype = column.dtype.bind(py).extract::<PyDataType>()?.0;
                Ok(JsonColumn {
                    name: name.clone(),
                    dtype: Self::dtype_name(&dtype)?,
                    nullable: column.nullable,
                    checks: column.checks.iter().map(Self::json_check).collect(),
                })
            })
            .collect::<PyResult<_>>()?;
        let table_checks = self
            .table_checks
            .iter()
            .map(Self::json_table_check)
            .collect();
        serde_json::to_string(&JsonSchema {
            version: JSON_SCHEMA_VERSION,
            columns,
            table_checks,
            lazy: self.lazy,
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, json: &str) -> PyResult<Self> {
        jiter::JsonValue::parse(json.as_bytes(), false).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid schema JSON: {error}"))
        })?;
        let document: JsonSchema = serde_json::from_str(json).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid schema JSON: {error}"))
        })?;
        Self::from_json_value(py, document)
    }

    #[classmethod]
    fn from_pydantic(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        model: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Self::from_pydantic_core_schema(py, &model)
    }

    fn __getitem__(
        slf: &Bound<'_, Self>,
        name: String,
        py: Python<'_>,
    ) -> PyResult<Py<PyColumnRef>> {
        if !slf
            .borrow()
            .columns
            .iter()
            .any(|(column_name, _)| column_name == &name)
        {
            return Err(pyo3::exceptions::PyKeyError::new_err(name));
        }

        Py::new(
            py,
            PyColumnRef {
                schema: slf.clone().unbind(),
                name,
            },
        )
    }

    #[pyo3(signature = (df, raise_on_error=None))]
    fn validate(
        &self,
        py: Python<'_>,
        df: &Bound<'_, PyAny>,
        raise_on_error: Option<bool>,
    ) -> PyResult<Option<PyValidationReport>> {
        let raise = raise_on_error.unwrap_or(!self.lazy);
        let rust_schema = self.core_schema(py)?;
        let polars_df = materialize_frame(df)?;
        let report = rust_schema.validate(&polars_df);

        if report.is_valid() {
            Ok(None)
        } else if raise {
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

    fn validate_lazy(
        &self,
        py: Python<'_>,
        df: &Bound<'_, PyAny>,
    ) -> PyResult<Option<PyValidationReport>> {
        let rust_schema = self.core_schema(py)?;
        let polars_df = materialize_frame(df)?;
        let report = rust_schema.validate(&polars_df);
        if report.is_valid() {
            Ok(None)
        } else {
            Ok(Some(PyValidationReport::from(report)))
        }
    }

    #[pyo3(signature = (n=10, seed=None))]
    fn sample(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<PyDataFrame> {
        let rust_schema = self.core_schema(py)?;
        let seed = seed.unwrap_or_else(rand::random::<u64>);
        let frame = framelaw_core::generate::sample(&rust_schema, n, seed)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(PyDataFrame(frame))
    }

    #[getter]
    fn lazy(&self) -> bool {
        self.lazy
    }
}

#[pymethods]
impl PyColumnRef {
    fn add_check(&self, py: Python<'_>, check: PyCheck) -> PyResult<()> {
        let mut schema = self.schema.bind(py).borrow_mut();
        let (_, column) = schema
            .columns
            .iter_mut()
            .find(|(name, _)| name == &self.name)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(self.name.clone()))?;
        column.checks.push(check);
        Ok(())
    }
}

#[pyfunction]
fn infer_schema(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<PySchema> {
    let dataframe = materialize_frame(df)?;
    PySchema::from_core(py, infer_core_schema(&dataframe))
}

#[pyfunction(name = "sample")]
#[pyo3(signature = (schema, n=10, seed=None))]
fn sample_function(
    schema: &Bound<'_, PySchema>,
    n: usize,
    seed: Option<u64>,
) -> PyResult<PyDataFrame> {
    let borrowed = schema.borrow();
    borrowed.sample(schema.py(), n, seed)
}

fn materialize_frame(value: &Bound<'_, PyAny>) -> PyResult<DataFrame> {
    if let Ok(dataframe) = value.extract::<PyDataFrame>() {
        return Ok(dataframe.into());
    }

    if let Ok(collected) = value.call_method0("collect") {
        let dataframe = collected.extract::<PyDataFrame>()?;
        return Ok(dataframe.into());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a Polars DataFrame or LazyFrame",
    ))
}

#[pymodule]
fn _framelaw(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCheck>()?;
    m.add_class::<PyDataFrameCheck>()?;
    m.add_class::<PyColumn>()?;
    m.add_class::<PyColumnRef>()?;
    m.add_class::<PySchema>()?;
    m.add_class::<PyValidationError>()?;
    m.add_class::<PyValidationReport>()?;

    m.add_function(wrap_pyfunction!(not_null, m)?)?;
    m.add_function(wrap_pyfunction!(unique_values, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(gte, m)?)?;
    m.add_function(wrap_pyfunction!(lte, m)?)?;
    m.add_function(wrap_pyfunction!(between, m)?)?;
    m.add_function(wrap_pyfunction!(matches_regex, m)?)?;
    m.add_function(wrap_pyfunction!(is_in, m)?)?;
    m.add_function(wrap_pyfunction!(matches_check, m)?)?;
    m.add_function(wrap_pyfunction!(one_of, m)?)?;
    m.add_function(wrap_pyfunction!(unique, m)?)?;
    m.add_function(wrap_pyfunction!(list_len_between, m)?)?;
    m.add_function(wrap_pyfunction!(list_not_empty, m)?)?;
    m.add_function(wrap_pyfunction!(struct_field, m)?)?;
    m.add_function(wrap_pyfunction!(gt_column, m)?)?;
    m.add_function(wrap_pyfunction!(lt_column, m)?)?;
    m.add_function(wrap_pyfunction!(gte_column, m)?)?;
    m.add_function(wrap_pyfunction!(lte_column, m)?)?;
    m.add_function(wrap_pyfunction!(eq_columns, m)?)?;
    m.add_function(wrap_pyfunction!(ne_columns, m)?)?;
    m.add_function(wrap_pyfunction!(unique_together, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema, m)?)?;
    m.add_function(wrap_pyfunction!(sample_function, m)?)?;

    Ok(())
}
