use std::sync::Arc;

use polars::prelude::{BooleanChunked, DataFrame, NamedFrom, Series};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDelta, PyDict, PyIterator, PyList, PyTuple, PyType};
use pyo3_polars::{PyDataFrame, PyDataType, PySeries};
use regex::Regex;
use serde::{Deserialize, Serialize};

use framelaw_core::checks::{
    BetweenF64, Check, CheckSpec, EndsWith, GreaterThanF64, GreaterThanOrEqualF64, LessThanF64,
    LessThanOrEqualF64, ListLenBetween, ListNotEmpty, MatchesRegex, NotBlank, NotNull,
    OneOfNumericValues, OneOfString, StartsWith, StrLengthBetween, StructFieldCheck, UniqueValues,
};
use framelaw_core::column::ColumnSpec;
use framelaw_core::dataframe_checks::{
    ColumnComparison, CompareOp, DataFrameCheck, DataFrameCheckSpec, Determines, Implies, Ordered,
    SumColumns, UniqueTogether, Within,
};
use framelaw_core::errors::ValidationError;
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
    OneOfNumeric(Vec<f64>),
    StrLengthBetween(usize, usize),
    NotBlank,
    StartsWith(String),
    EndsWith(String),
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
            CheckSpec::OneOfNumeric(values) => CheckType::OneOfNumeric(values),
            CheckSpec::StrLengthBetween(min, max) => CheckType::StrLengthBetween(min, max),
            CheckSpec::NotBlank => CheckType::NotBlank,
            CheckSpec::StartsWith(prefix) => CheckType::StartsWith(prefix),
            CheckSpec::EndsWith(suffix) => CheckType::EndsWith(suffix),
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
            CheckType::OneOfNumeric(v) => Box::new(OneOfNumericValues { allowed_values: v }),
            CheckType::StrLengthBetween(min, max) => Box::new(StrLengthBetween {
                min_len: min,
                max_len: max,
            }),
            CheckType::NotBlank => Box::new(NotBlank),
            CheckType::StartsWith(prefix) => Box::new(StartsWith { prefix }),
            CheckType::EndsWith(suffix) => Box::new(EndsWith { suffix }),
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
pub fn is_in_strings(allowed: Vec<String>) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::OneOfString(allowed),
    })
}

/// Classifies an allowed-value list into a string or a numeric membership
/// check. All elements must share one class: strings, or numbers/booleans.
/// Numeric membership compares at `f64` granularity against `f64` rows and
/// exactly against integer rows through the widened committed sets.
#[pyfunction]
pub fn is_in(allowed: &Bound<'_, PyList>) -> PyResult<PyCheck> {
    if allowed.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "allowed list must not be empty",
        ));
    }
    let mut strings: Vec<String> = Vec::with_capacity(allowed.len());
    let mut numeric: Vec<f64> = Vec::with_capacity(allowed.len());
    let mut saw_string = false;
    let mut saw_number = false;
    for item in allowed.iter() {
        if item.is_instance_of::<pyo3::types::PyBool>() {
            saw_number = true;
            let truthy: bool = item.extract()?;
            numeric.push(if truthy { 1.0 } else { 0.0 });
        } else if item.is_instance_of::<pyo3::types::PyString>() {
            saw_string = true;
            strings.push(item.extract()?);
        } else if item.is_instance_of::<pyo3::types::PyInt>() {
            saw_number = true;
            // Any i128 converts to a finite f64 (i128 tops out around
            // 1.7e38, far below f64's range); values beyond the 2^53
            // mantissa round, which the exact integer comparison in the
            // core will honor per-row.
            let value: i128 = item.extract()?;
            numeric.push(value as f64);
        } else if item.is_instance_of::<pyo3::types::PyFloat>() {
            saw_number = true;
            let value: f64 = item.extract()?;
            if !value.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "numeric allowed values must be finite",
                ));
            }
            numeric.push(value);
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "is_in supports lists of strings, numbers, or booleans",
            ));
        }
        if saw_string && saw_number {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "allowed list must contain only strings or only numbers/booleans",
            ));
        }
    }
    Ok(PyCheck {
        inner: if saw_string {
            CheckType::OneOfString(strings)
        } else {
            CheckType::OneOfNumeric(numeric)
        },
    })
}

#[pyfunction(name = "matches")]
fn matches_check(pattern: String) -> PyResult<PyCheck> {
    matches_regex(pattern)
}

#[pyfunction(name = "one_of")]
fn one_of(allowed: &Bound<'_, PyList>) -> PyResult<PyCheck> {
    is_in(allowed)
}

#[pyfunction(name = "unique")]
fn unique() -> PyResult<PyCheck> {
    unique_values()
}

#[pyfunction]
fn str_length_between(min_len: usize, max_len: usize) -> PyResult<PyCheck> {
    if min_len > max_len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "min_len must be <= max_len",
        ));
    }
    Ok(PyCheck {
        inner: CheckType::StrLengthBetween(min_len, max_len),
    })
}

#[pyfunction]
fn not_blank() -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::NotBlank,
    })
}

#[pyfunction]
fn starts_with(prefix: String) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::StartsWith(prefix),
    })
}

#[pyfunction]
fn ends_with(suffix: String) -> PyResult<PyCheck> {
    Ok(PyCheck {
        inner: CheckType::EndsWith(suffix),
    })
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
    Ordered {
        columns: Vec<String>,
        strict: bool,
    },
    SumColumns {
        addends: Vec<String>,
        target: String,
        tolerance: Option<f64>,
    },
    Implies {
        left: String,
        by: String,
    },
    Determines {
        a: String,
        b: String,
    },
    Within {
        left: String,
        right: String,
        max_gap_ns: i64,
    },
    /// Vectorized custom predicate: a `pl.Expr` (or a zero-argument callable
    /// returning one) that must produce a single Boolean column.
    CustomExpr {
        name: String,
        expr: Arc<Py<PyAny>>,
        max_reported_rows: usize,
    },
    /// Batch custom predicate: a callable receiving one `polars.DataFrame`
    /// per validation chunk and returning a Boolean mask.
    CustomMask {
        name: String,
        func: Arc<Py<PyAny>>,
        columns: Vec<String>,
        max_reported_rows: usize,
    },
    /// Row custom predicate: a callable receiving one `dict` per row.
    CustomRows {
        name: String,
        func: Arc<Py<PyAny>>,
        columns: Option<Vec<String>>,
        max_reported_rows: usize,
    },
}

/// A custom check resolved out of the Python-facing enum, ready to run.
#[derive(Clone)]
struct CustomCheck {
    name: String,
    max_reported_rows: usize,
    kind: CustomCheckKind,
}

#[derive(Clone)]
enum CustomCheckKind {
    Expr(Arc<Py<PyAny>>),
    Mask {
        func: Arc<Py<PyAny>>,
        columns: Vec<String>,
    },
    Rows {
        func: Arc<Py<PyAny>>,
        columns: Option<Vec<String>>,
    },
}

/// Result of splitting a `PyDataFrameCheck` into what the Rust core can run
/// and what only the Python binding can evaluate.
enum CoreTableCheck {
    Native(Box<dyn DataFrameCheck>),
    Custom(CustomCheck),
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
            DataFrameCheckSpec::Ordered { columns, strict } => {
                TableCheckType::Ordered { columns, strict }
            }
            DataFrameCheckSpec::SumColumns {
                addends,
                target,
                tolerance,
            } => TableCheckType::SumColumns {
                addends,
                target,
                tolerance,
            },
            DataFrameCheckSpec::Implies { left, by } => TableCheckType::Implies { left, by },
            DataFrameCheckSpec::Determines { a, b } => TableCheckType::Determines { a, b },
            DataFrameCheckSpec::Within {
                left,
                right,
                max_gap_ns,
            } => TableCheckType::Within {
                left,
                right,
                max_gap_ns,
            },
        };
        Self { inner }
    }

    fn is_custom(&self) -> bool {
        matches!(
            self.inner,
            TableCheckType::CustomExpr { .. }
                | TableCheckType::CustomMask { .. }
                | TableCheckType::CustomRows { .. }
        )
    }

    fn resolve(self) -> CoreTableCheck {
        match self.inner {
            TableCheckType::Comparison { left, op, right } => {
                CoreTableCheck::Native(Box::new(ColumnComparison { left, op, right }))
            }
            TableCheckType::UniqueTogether { columns } => {
                CoreTableCheck::Native(Box::new(UniqueTogether { columns }))
            }
            TableCheckType::Ordered { columns, strict } => {
                CoreTableCheck::Native(Box::new(Ordered { columns, strict }))
            }
            TableCheckType::SumColumns {
                addends,
                target,
                tolerance,
            } => CoreTableCheck::Native(Box::new(SumColumns {
                addends,
                target,
                tolerance,
            })),
            TableCheckType::Implies { left, by } => {
                CoreTableCheck::Native(Box::new(Implies { left, by }))
            }
            TableCheckType::Determines { a, b } => {
                CoreTableCheck::Native(Box::new(Determines { a, b }))
            }
            TableCheckType::Within {
                left,
                right,
                max_gap_ns,
            } => CoreTableCheck::Native(Box::new(Within {
                left,
                right,
                max_gap_ns,
            })),
            TableCheckType::CustomExpr {
                name,
                expr,
                max_reported_rows,
            } => CoreTableCheck::Custom(CustomCheck {
                name,
                max_reported_rows,
                kind: CustomCheckKind::Expr(expr),
            }),
            TableCheckType::CustomMask {
                name,
                func,
                columns,
                max_reported_rows,
            } => CoreTableCheck::Custom(CustomCheck {
                name,
                max_reported_rows,
                kind: CustomCheckKind::Mask { func, columns },
            }),
            TableCheckType::CustomRows {
                name,
                func,
                columns,
                max_reported_rows,
            } => CoreTableCheck::Custom(CustomCheck {
                name,
                max_reported_rows,
                kind: CustomCheckKind::Rows { func, columns },
            }),
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

#[pyfunction]
fn ordered(columns: Vec<String>, strict: bool) -> PyResult<PyDataFrameCheck> {
    if columns.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ordered needs at least two columns",
        ));
    }
    if columns.iter().any(|name| name.is_empty()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ordered column names must not be empty",
        ));
    }
    let mut deduped = columns.clone();
    deduped.sort_unstable();
    deduped.dedup();
    if deduped.len() != columns.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ordered column names must be distinct",
        ));
    }
    Ok(PyDataFrameCheck {
        inner: TableCheckType::Ordered { columns, strict },
    })
}

#[pyfunction]
#[pyo3(signature = (addends, target, tolerance = None))]
fn sum_columns(
    addends: Vec<String>,
    target: String,
    tolerance: Option<f64>,
) -> PyResult<PyDataFrameCheck> {
    if addends.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "sum_columns needs at least one addend",
        ));
    }
    if let Some(tolerance) = tolerance {
        if !(0.0..=f64::INFINITY).contains(&tolerance) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tolerance must be a non-negative finite number",
            ));
        }
    }
    Ok(PyDataFrameCheck {
        inner: TableCheckType::SumColumns {
            addends,
            target,
            tolerance,
        },
    })
}

#[pyfunction]
fn implies(left: String, by: String) -> PyResult<PyDataFrameCheck> {
    if left == by {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "implies needs two different columns",
        ));
    }
    Ok(PyDataFrameCheck {
        inner: TableCheckType::Implies { left, by },
    })
}

#[pyfunction]
fn determines(a: String, b: String) -> PyResult<PyDataFrameCheck> {
    if a == b {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "determines needs two different columns",
        ));
    }
    Ok(PyDataFrameCheck {
        inner: TableCheckType::Determines { a, b },
    })
}

/// Builds the per-row temporal gap bound from a Python `timedelta`.
/// Resolution is microseconds, scaled up to nanoseconds; the bound is
/// interpreted against the shared dtype's time unit by the core.
fn timedelta_gap_ns(max_gap: &Bound<'_, PyDelta>) -> PyResult<i64> {
    let days = max_gap.getattr("days")?.extract::<i64>()?;
    let seconds = max_gap.getattr("seconds")?.extract::<i64>()?;
    let microseconds = max_gap.getattr("microseconds")?.extract::<i64>()?;
    let nanoseconds = days
        .checked_mul(86_400_000_000_000)
        .and_then(|n| n.checked_add(seconds * 1_000_000_000))
        .and_then(|n| n.checked_add(microseconds * 1_000));
    match nanoseconds {
        Some(ns) => Ok(ns),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "max_gap exceeds the storable nanosecond range",
        )),
    }
}

#[pyfunction]
fn within(left: String, right: String, max_gap: &Bound<'_, PyDelta>) -> PyResult<PyDataFrameCheck> {
    if left == right {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "within needs two different columns",
        ));
    }
    let max_gap_ns = timedelta_gap_ns(max_gap)?;
    Ok(PyDataFrameCheck {
        inner: TableCheckType::Within {
            left,
            right,
            max_gap_ns,
        },
    })
}

fn custom_check_name(name: Option<String>, default: &str) -> PyResult<String> {
    let name = name.unwrap_or_else(|| default.to_string());
    if name.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "custom check name must not be empty",
        ));
    }
    Ok(name)
}

fn validate_max_reported_rows(max_reported_rows: usize) -> PyResult<()> {
    if max_reported_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_reported_rows must be at least 1",
        ));
    }
    Ok(())
}

/// Builds a vectorized custom check from a Polars expression.
///
/// The expression (or zero-argument callable returning one) is evaluated with
/// `DataFrame.select` and must yield exactly one Boolean column of the same
/// height as the validated frame. Rows where the predicate is `False` fail;
/// rows where it is null are skipped.
#[pyfunction]
#[pyo3(signature = (expr, name = None, max_reported_rows = 1000))]
fn check_expr(
    py: Python<'_>,
    expr: Py<PyAny>,
    name: Option<String>,
    max_reported_rows: usize,
) -> PyResult<PyDataFrameCheck> {
    let bound = expr.bind(py);
    if !bound.is_callable() && !bound.hasattr("meta")? {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "check_expr expects a polars.Expr or a zero-argument callable returning one",
        ));
    }
    validate_max_reported_rows(max_reported_rows)?;
    Ok(PyDataFrameCheck {
        inner: TableCheckType::CustomExpr {
            name: custom_check_name(name, "CustomExpr")?,
            expr: Arc::new(expr),
            max_reported_rows,
        },
    })
}

/// Builds a batch custom check from a Python callable.
///
/// The callable receives one `polars.DataFrame` per validation chunk and must
/// return a Boolean `polars.Series` (or a sequence of booleans) of the same
/// height. Rows where the mask is `False` fail; nulls are skipped.
#[pyfunction]
#[pyo3(signature = (func, columns = None, name = None, max_reported_rows = 1000))]
fn check_mask(
    py: Python<'_>,
    func: Py<PyAny>,
    columns: Option<Vec<String>>,
    name: Option<String>,
    max_reported_rows: usize,
) -> PyResult<PyDataFrameCheck> {
    if !func.bind(py).is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "check_mask expects a callable taking one polars.DataFrame",
        ));
    }
    validate_max_reported_rows(max_reported_rows)?;
    Ok(PyDataFrameCheck {
        inner: TableCheckType::CustomMask {
            name: custom_check_name(name, "CustomMask")?,
            func: Arc::new(func),
            columns: columns.unwrap_or_default(),
            max_reported_rows,
        },
    })
}

/// Builds a row custom check from a Python callable.
///
/// The callable receives one `dict` per row (all columns by default, or the
/// selected `columns`) and must return a boolean, or `None` to skip the row.
/// It is compiled to `pl.struct(...).map_elements(...)`, so Polars batches
/// rows internally while the predicate stays a plain Python function.
#[pyfunction]
#[pyo3(signature = (func, columns = None, name = None, max_reported_rows = 1000))]
fn check_rows(
    py: Python<'_>,
    func: Py<PyAny>,
    columns: Option<Vec<String>>,
    name: Option<String>,
    max_reported_rows: usize,
) -> PyResult<PyDataFrameCheck> {
    if !func.bind(py).is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "check_rows expects a callable taking one dict per row",
        ));
    }
    if columns.as_ref().is_some_and(Vec::is_empty) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "check_rows columns must not be empty; omit columns to pass the whole row",
        ));
    }
    validate_max_reported_rows(max_reported_rows)?;
    Ok(PyDataFrameCheck {
        inner: TableCheckType::CustomRows {
            name: custom_check_name(name, "CustomRows")?,
            func: Arc::new(func),
            columns,
            max_reported_rows,
        },
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
    OneOfNumeric {
        values: Vec<f64>,
    },
    StrLengthBetween {
        min: usize,
        max: usize,
    },
    NotBlank,
    StartsWith {
        prefix: String,
    },
    EndsWith {
        suffix: String,
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

#[derive(Clone, Deserialize, Serialize)]
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
    Ordered {
        columns: Vec<String>,
        strict: bool,
    },
    SumColumns {
        addends: Vec<String>,
        target: String,
        #[serde(default)]
        tolerance: Option<f64>,
    },
    Implies {
        left: String,
        by: String,
    },
    Determines {
        a: String,
        b: String,
    },
    Within {
        left: String,
        right: String,
        max_gap_ns: i64,
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
                        vec![is_in_strings(expected)?],
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
                format!("List:{}", Self::braced_nested(&Self::dtype_name(inner)?))
            }
            polars::prelude::DataType::Array(inner, width) => {
                format!(
                    "Array:{}:{}",
                    Self::braced_nested(&Self::dtype_name(inner)?),
                    width
                )
            }
            polars::prelude::DataType::Struct(fields) => {
                let parts = fields
                    .iter()
                    .map(|field| {
                        Ok(format!(
                            "{}={}",
                            Self::escape_name(field.name.as_str()),
                            Self::braced_nested(&Self::dtype_name(&field.dtype)?)
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
            // Braced payloads come from the current writer; bare payloads
            // are historical tags, which take the whole remainder.
            let inner = Self::strip_braces(inner).unwrap_or(inner);
            return Ok(polars::prelude::DataType::List(Box::new(
                Self::parse_dtype_name(inner)?,
            )));
        }
        if let Some(rest) = name.strip_prefix("Array:") {
            if let Some((inner, suffix)) = Self::split_braced_head(rest) {
                let width: usize = suffix
                    .strip_prefix(':')
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported JSON Polars dtype '{name}'"
                        ))
                    })?
                    .parse()
                    .map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported JSON Polars dtype '{name}'"
                        ))
                    })?;
                return Ok(polars::prelude::DataType::Array(
                    Box::new(Self::parse_dtype_name(inner)?),
                    width,
                ));
            }
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
                for part in Self::split_top_level(rest) {
                    let (field_name, field_dtype) = Self::split_field(&part).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Unsupported JSON Polars dtype '{name}'"
                        ))
                    })?;
                    // Braced payloads come from the current writer; bare
                    // payloads are historical tags, parsed as a whole.
                    let field_dtype = Self::strip_braces(&field_dtype).unwrap_or(&field_dtype);
                    fields.push(polars::prelude::Field::new(
                        Self::unescape(&field_name).into(),
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

    /// Escapes a struct field name for dtype tags. Separators (`,`, `=`),
    /// braces, and backslashes would otherwise parse as structure;
    /// backslashes are escaped first so unescaping is unambiguous.
    fn escape_name(name: &str) -> String {
        name.replace('\\', "\\\\")
            .replace(',', "\\,")
            .replace('=', "\\=")
            .replace('{', "\\{")
            .replace('}', "\\}")
    }

    /// Wraps a composite child dtype tag in braces when it contains
    /// structural characters, so enclosing comma-separated lists stay
    /// unambiguous at any nesting depth. Scalar tags pass through, keeping
    /// historical payloads byte-identical.
    fn braced_nested(tag: &str) -> String {
        if tag.contains([',', '{', '}']) {
            format!("{{{tag}}}")
        } else {
            tag.to_string()
        }
    }

    /// Reverses `escape_name`: `\x` becomes `x` for any `x`.
    fn unescape(value: &str) -> String {
        let mut out = String::with_capacity(value.len());
        let mut chars = value.chars();
        while let Some(char) = chars.next() {
            if char == '\\' {
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            } else {
                out.push(char);
            }
        }
        out
    }

    /// Splits a struct payload on top-level commas, skipping braced regions
    /// and backslash escape pairs.
    fn split_top_level(value: &str) -> Vec<String> {
        let mut parts = Vec::new();
        let mut current = String::new();
        let mut chars = value.chars().peekable();
        let mut depth = 0usize;
        while let Some(char) = chars.next() {
            if char == '\\' {
                current.push(char);
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            } else if char == '{' {
                depth += 1;
                current.push(char);
            } else if char == '}' {
                depth = depth.saturating_sub(1);
                current.push(char);
            } else if char == ',' && depth == 0 {
                parts.push(std::mem::take(&mut current));
            } else {
                current.push(char);
            }
        }
        parts.push(current);
        parts
    }

    /// Splits one `name=dtype` field on the first top-level `=`.
    fn split_field(value: &str) -> Option<(String, String)> {
        let mut name = String::new();
        let mut chars = value.chars().peekable();
        let mut depth = 0usize;
        while let Some(char) = chars.next() {
            if char == '\\' {
                name.push(char);
                if let Some(next) = chars.next() {
                    name.push(next);
                }
            } else if char == '{' {
                depth += 1;
                name.push(char);
            } else if char == '}' {
                depth = depth.saturating_sub(1);
                name.push(char);
            } else if char == '=' && depth == 0 {
                let rest: String = chars.collect();
                return Some((name, rest));
            } else {
                name.push(char);
            }
        }
        None
    }

    /// Strips one layer of surrounding braces, requiring them to balance
    /// over the whole string (escapes respected).
    fn strip_braces(value: &str) -> Option<&str> {
        let inner = value.strip_prefix('{')?.strip_suffix('}')?;
        let mut depth = 0usize;
        let mut chars = inner.chars().peekable();
        while let Some(char) = chars.next() {
            if char == '\\' {
                chars.next();
            } else if char == '{' {
                depth += 1;
            } else if char == '}' {
                if depth == 0 {
                    return None;
                }
                depth -= 1;
            }
        }
        (depth == 0).then_some(inner)
    }

    /// Splits a leading `{...}` balanced group into its inner payload and
    /// the remainder, respecting backslash escapes.
    fn split_braced_head(value: &str) -> Option<(&str, &str)> {
        let rest = value.strip_prefix('{')?;
        let mut depth = 0usize;
        let mut end = None;
        let mut chars = rest.char_indices().peekable();
        while let Some((index, char)) = chars.next() {
            if char == '\\' {
                chars.next();
            } else if char == '{' {
                depth += 1;
            } else if char == '}' {
                if depth == 0 {
                    end = Some(index);
                    break;
                }
                depth -= 1;
            }
        }
        let end = end?;
        // Byte indices align: braces and backslashes are single-byte, and
        // only ASCII is special to this scanner.
        if !rest.is_char_boundary(end) {
            return None;
        }
        Some((&rest[..end], &rest[end + 1..]))
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
            CheckType::OneOfNumeric(values) => {
                let mut values = values.clone();
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                JsonCheck::OneOfNumeric { values }
            }
            CheckType::StrLengthBetween(min, max) => JsonCheck::StrLengthBetween {
                min: *min,
                max: *max,
            },
            CheckType::NotBlank => JsonCheck::NotBlank,
            CheckType::StartsWith(prefix) => JsonCheck::StartsWith {
                prefix: prefix.clone(),
            },
            CheckType::EndsWith(suffix) => JsonCheck::EndsWith {
                suffix: suffix.clone(),
            },
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
            JsonCheck::OneOf { values } => is_in_strings(values),
            JsonCheck::OneOfNumeric { values } => Ok(PyCheck {
                inner: CheckType::OneOfNumeric(values),
            }),
            JsonCheck::StrLengthBetween { min, max } => str_length_between(min, max),
            JsonCheck::NotBlank => not_blank(),
            JsonCheck::StartsWith { prefix } => starts_with(prefix),
            JsonCheck::EndsWith { suffix } => ends_with(suffix),
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
            TableCheckType::Ordered { columns, strict } => JsonTableCheck::Ordered {
                columns: columns.clone(),
                strict: *strict,
            },
            TableCheckType::SumColumns {
                addends,
                target,
                tolerance,
            } => JsonTableCheck::SumColumns {
                addends: addends.clone(),
                target: target.clone(),
                tolerance: *tolerance,
            },
            TableCheckType::Implies { left, by } => JsonTableCheck::Implies {
                left: left.clone(),
                by: by.clone(),
            },
            TableCheckType::Determines { a, b } => JsonTableCheck::Determines {
                a: a.clone(),
                b: b.clone(),
            },
            TableCheckType::Within {
                left,
                right,
                max_gap_ns,
            } => JsonTableCheck::Within {
                left: left.clone(),
                right: right.clone(),
                max_gap_ns: *max_gap_ns,
            },
            // `to_json` rejects schemas containing custom checks before this
            // point; custom callables are inherently not serializable.
            TableCheckType::CustomExpr { .. }
            | TableCheckType::CustomMask { .. }
            | TableCheckType::CustomRows { .. } => {
                unreachable!("custom checks are rejected before serialization")
            }
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
            JsonTableCheck::Ordered { columns, strict } => ordered(columns, strict),
            JsonTableCheck::SumColumns {
                addends,
                target,
                tolerance,
            } => sum_columns(addends, target, tolerance),
            JsonTableCheck::Implies { left, by } => implies(left, by),
            JsonTableCheck::Determines { a, b } => determines(a, b),
            JsonTableCheck::Within {
                left,
                right,
                max_gap_ns,
            } => Ok(PyDataFrameCheck {
                inner: TableCheckType::Within {
                    left,
                    right,
                    max_gap_ns,
                },
            }),
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

    fn core_schema(&self, py: Python<'_>) -> PyResult<(Schema, Vec<CustomCheck>)> {
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

        let mut table_checks: Vec<Box<dyn DataFrameCheck>> = Vec::new();
        let mut custom_checks: Vec<CustomCheck> = Vec::new();
        for check in &self.table_checks {
            match check.clone().resolve() {
                CoreTableCheck::Native(check) => table_checks.push(check),
                CoreTableCheck::Custom(check) => custom_checks.push(check),
            }
        }

        Ok((
            Schema::with_table_checks(specs, table_checks),
            custom_checks,
        ))
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
        if self.table_checks.iter().any(PyDataFrameCheck::is_custom) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot serialize a schema that contains custom checks; remove them or rebuild \
                 the schema without check_expr/check_mask/check_rows before calling to_json",
            ));
        }
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

    #[pyo3(signature = (df, raise_on_error=None, chunk_rows=None))]
    fn validate(
        &self,
        py: Python<'_>,
        df: &Bound<'_, PyAny>,
        raise_on_error: Option<bool>,
        chunk_rows: Option<u64>,
    ) -> PyResult<Option<PyValidationReport>> {
        let (rust_schema, custom_checks) = self.core_schema(py)?;

        if let Some(chunk_rows) = chunk_rows {
            return validate_chunked(py, df, &rust_schema, &custom_checks, chunk_rows);
        }

        let raise = raise_on_error.unwrap_or(!self.lazy);
        let polars_df = materialize_frame(df)?;
        let mut report = py.detach(|| rust_schema.validate(&polars_df));
        report
            .errors
            .extend(run_custom_checks(py, &polars_df, &custom_checks)?);

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

    #[pyo3(signature = (df, chunk_rows=None))]
    fn validate_lazy(
        &self,
        py: Python<'_>,
        df: &Bound<'_, PyAny>,
        chunk_rows: Option<u64>,
    ) -> PyResult<Option<PyValidationReport>> {
        let (rust_schema, custom_checks) = self.core_schema(py)?;
        if let Some(chunk_rows) = chunk_rows {
            return validate_chunked(py, df, &rust_schema, &custom_checks, chunk_rows);
        }
        let polars_df = materialize_frame(df)?;
        let mut report = py.detach(|| rust_schema.validate(&polars_df));
        report
            .errors
            .extend(run_custom_checks(py, &polars_df, &custom_checks)?);
        if report.is_valid() {
            Ok(None)
        } else {
            Ok(Some(PyValidationReport::from(report)))
        }
    }

    #[pyo3(signature = (n=10, seed=None))]
    fn sample(&self, py: Python<'_>, n: usize, seed: Option<u64>) -> PyResult<PyDataFrame> {
        let (rust_schema, _) = self.core_schema(py)?;
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

/// Extracts failing row indices from a Boolean mask. `True` passes, `False`
/// fails, and nulls are skipped. Only the first `max_rows` indices are kept,
/// alongside the total failure count, so a fully broken chunk cannot balloon
/// the report.
fn failing_rows(mask: &BooleanChunked, max_rows: usize) -> (Vec<usize>, usize) {
    let mut rows = Vec::new();
    let mut total = 0usize;
    for (index, value) in mask.iter().enumerate() {
        if value == Some(false) {
            total += 1;
            if rows.len() < max_rows {
                rows.push(index);
            }
        }
    }
    (rows, total)
}

/// Evaluates a resolved `pl.Expr` against one frame and returns its Boolean
/// mask. The expression must produce exactly one Boolean column and preserve
/// the frame height.
fn evaluate_expr_mask(
    df: &Bound<'_, PyAny>,
    expr: &Bound<'_, PyAny>,
    height: usize,
) -> PyResult<BooleanChunked> {
    let selected = df.call_method1("select", (expr,))?;
    let frame = selected.extract::<PyDataFrame>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "custom expression check must be a Polars expression producing one Boolean column",
        )
    })?;
    let frame = frame.0;
    if frame.width() != 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "custom expression check must select exactly one column, got {}",
            frame.width()
        )));
    }
    if frame.height() != height {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "custom expression check must preserve the row count ({}), got {}",
            height,
            frame.height()
        )));
    }
    let series = frame.columns()[0].as_materialized_series();
    series.bool().cloned().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "custom expression check must return Boolean, got {}",
            series.dtype()
        ))
    })
}

/// Calls one batch custom check and coerces its result to a Boolean mask.
/// A `polars.Series` is extracted directly; other sequences are wrapped with
/// `polars.Series(...)`.
fn evaluate_mask(
    df: &Bound<'_, PyAny>,
    func: &Bound<'_, PyAny>,
    height: usize,
) -> PyResult<BooleanChunked> {
    let result = func.call1((df,))?;
    if result.is_none() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "custom mask check must return a Boolean sequence, not None",
        ));
    }
    let series: Series = if let Ok(series) = result.extract::<PySeries>() {
        series.0
    } else {
        let polars = result.py().import("polars")?;
        let series_class = polars.getattr("Series")?;
        let wrapped = series_class.call1((&result,)).map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "custom mask check must return a polars.Series of booleans or a sequence of booleans",
            )
        })?;
        wrapped.extract::<PySeries>()?.0
    };
    if series.len() != height {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "custom mask check must return {} value(s) for the current chunk, got {}",
            height,
            series.len()
        )));
    }
    series.bool().cloned().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "custom mask check must return Boolean values, got {}",
            series.dtype()
        ))
    })
}

/// Best-effort column attribution for custom expression checks.
fn expr_root_names(expr: &Bound<'_, PyAny>) -> Vec<String> {
    let Ok(meta) = expr.getattr("meta") else {
        return Vec::new();
    };
    let Ok(names) = meta.call_method0("root_names") else {
        return Vec::new();
    };
    names.extract::<Vec<String>>().unwrap_or_default()
}

/// Builds the `pl.struct(...).map_elements(...)` expression for a row check.
///
/// Wrapping the selected columns in a struct keeps the callback input a plain
/// `dict`, which is the documented `fn(row: dict) -> bool` contract.
fn rows_mask_expr<'py>(
    py: Python<'py>,
    func: &Bound<'py, PyAny>,
    columns: Option<&[String]>,
) -> PyResult<Bound<'py, PyAny>> {
    let polars = py.import("polars")?;
    let struct_expr = match columns {
        Some(columns) if !columns.is_empty() => {
            let mut args = Vec::with_capacity(columns.len());
            for column in columns {
                args.push(polars.getattr("col")?.call1((column,))?);
            }
            polars.getattr("struct")?.call1(PyTuple::new(py, args)?)?
        }
        _ => {
            let all = polars.getattr("all")?.call0()?;
            polars.getattr("struct")?.call1((all,))?
        }
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("return_dtype", polars.getattr("Boolean")?)?;
    struct_expr.call_method("map_elements", (func,), Some(&kwargs))
}

/// Runs every custom check sequentially against the materialized frame.
///
/// Deliberately not parallel: custom checks re-enter Python, and running them
/// concurrently with the Rayon-powered native pass risks GIL convoys. Each
/// Polars expression already uses the global thread pool internally.
fn run_custom_checks(
    py: Python<'_>,
    df: &DataFrame,
    checks: &[CustomCheck],
) -> PyResult<Vec<ValidationError>> {
    if checks.is_empty() || df.height() == 0 {
        return Ok(Vec::new());
    }

    let height = df.height();
    let py_df = PyDataFrame(df.clone()).into_pyobject(py)?;
    let mut errors = Vec::with_capacity(checks.len());

    for check in checks {
        let (mask, columns) = match &check.kind {
            CustomCheckKind::Expr(expr) => {
                let bound = expr.bind(py);
                let resolved = if bound.is_callable() {
                    bound.call0()?
                } else {
                    bound.clone()
                };
                let columns = expr_root_names(&resolved);
                (evaluate_expr_mask(&py_df, &resolved, height)?, columns)
            }
            CustomCheckKind::Mask { func, columns } => (
                evaluate_mask(&py_df, func.bind(py), height)?,
                columns.clone(),
            ),
            CustomCheckKind::Rows { func, columns } => {
                let expr = rows_mask_expr(py, func.bind(py), columns.as_deref())?;
                let attributed = match columns {
                    Some(columns) => columns.clone(),
                    None => df
                        .get_column_names()
                        .iter()
                        .map(|name| name.to_string())
                        .collect(),
                };
                (evaluate_expr_mask(&py_df, &expr, height)?, attributed)
            }
        };

        let (rows, total) = failing_rows(&mask, check.max_reported_rows);
        if total == 0 {
            continue;
        }

        let columns = if columns.is_empty() {
            vec![check.name.clone()]
        } else {
            columns
        };
        let message = if total > rows.len() {
            format!(
                "Custom check '{}' failed for {} row(s); first {} at {:?}",
                check.name,
                total,
                rows.len(),
                rows
            )
        } else {
            format!(
                "Custom check '{}' failed for {} row(s) at {:?}",
                check.name, total, rows
            )
        };
        errors.push(ValidationError::multi(
            columns,
            check.name.clone(),
            rows,
            message,
        ));
    }

    Ok(errors)
}

/// Validates one batch, shifting its error row indices by `offset`.
fn validate_chunk_into(
    py: Python<'_>,
    chunk: &DataFrame,
    rust_schema: &Schema,
    custom_checks: &[CustomCheck],
    offset: u64,
    errors: &mut Vec<framelaw_core::errors::ValidationError>,
) -> PyResult<()> {
    let chunk_report = py.detach(|| rust_schema.validate(chunk));
    let mut chunk_errors = chunk_report.errors;
    chunk_errors.extend(run_custom_checks(py, chunk, custom_checks)?);
    for mut error in chunk_errors {
        for row in &mut error.rows {
            *row += offset as usize;
        }
        errors.push(error);
    }
    Ok(())
}

/// Returns a streaming batch iterator for a `LazyFrame`, or `None` when the
/// input is eager (or the installed Polars does not support batched collects).
///
/// `collect_batches` executes the plan once and yields chunks, so chunked
/// validation no longer re-runs the whole lazy plan for every slice.
fn streaming_batches<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    chunk_rows: u64,
) -> PyResult<Option<Bound<'py, PyIterator>>> {
    if !df.hasattr("collect_batches")? {
        return Ok(None);
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("chunk_size", chunk_rows)?;
    kwargs.set_item("engine", "streaming")?;
    match df.call_method("collect_batches", (), Some(&kwargs)) {
        Ok(batches) => Ok(Some(batches.try_iter()?)),
        // Older Polars exposes `collect_batches` without the `engine` keyword;
        // fall back to the slice-and-collect loop in that case.
        Err(error) if error.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => Ok(None),
        Err(error) => Err(error),
    }
}

/// Validates the frame in batches of `chunk_rows` rows. Each batch is
/// re-validated with the GIL released, and error row indices are shifted
/// back to absolute frame positions. Chunked validation always reports
/// instead of raising; note that dataset-wide checks (unique, joint table
/// checks) only see one chunk at a time.
///
/// Lazy frames are streamed through `LazyFrame.collect_batches`, so the plan
/// runs once and peak memory stays proportional to `chunk_rows` rather than
/// the full dataset.
fn validate_chunked<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    rust_schema: &Schema,
    custom_checks: &[CustomCheck],
    chunk_rows: u64,
) -> PyResult<Option<PyValidationReport>> {
    if chunk_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "chunk_rows must be at least 1",
        ));
    }
    let mut errors: Vec<framelaw_core::errors::ValidationError> = Vec::new();
    let mut offset: u64 = 0;

    if let Some(batches) = streaming_batches(py, df, chunk_rows)? {
        for batch in batches {
            let batch = batch?;
            let chunk = batch.extract::<PyDataFrame>()?;
            let height = chunk.0.height() as u64;
            if height == 0 {
                continue;
            }
            validate_chunk_into(
                py,
                &chunk.0,
                rust_schema,
                custom_checks,
                offset,
                &mut errors,
            )?;
            offset += height;
        }
    } else {
        loop {
            let sliced = df.call_method1("slice", (offset as isize, chunk_rows as isize))?;
            let batch = if sliced.hasattr("collect")? {
                sliced.call_method0("collect")?
            } else {
                sliced
            };
            let chunk = batch.extract::<PyDataFrame>()?;
            let height = chunk.0.height() as u64;
            if height == 0 {
                break;
            }
            validate_chunk_into(
                py,
                &chunk.0,
                rust_schema,
                custom_checks,
                offset,
                &mut errors,
            )?;
            offset += height;
            if height < chunk_rows {
                break;
            }
        }
    }

    let report = framelaw_core::errors::ValidationReport { errors };
    if report.is_valid() {
        Ok(None)
    } else {
        Ok(Some(PyValidationReport::from(report)))
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
    m.add_function(wrap_pyfunction!(str_length_between, m)?)?;
    m.add_function(wrap_pyfunction!(not_blank, m)?)?;
    m.add_function(wrap_pyfunction!(starts_with, m)?)?;
    m.add_function(wrap_pyfunction!(ends_with, m)?)?;
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
    m.add_function(wrap_pyfunction!(ordered, m)?)?;
    m.add_function(wrap_pyfunction!(sum_columns, m)?)?;
    m.add_function(wrap_pyfunction!(implies, m)?)?;
    m.add_function(wrap_pyfunction!(determines, m)?)?;
    m.add_function(wrap_pyfunction!(within, m)?)?;
    m.add_function(wrap_pyfunction!(check_expr, m)?)?;
    m.add_function(wrap_pyfunction!(check_mask, m)?)?;
    m.add_function(wrap_pyfunction!(check_rows, m)?)?;
    m.add_function(wrap_pyfunction!(infer_schema, m)?)?;
    m.add_function(wrap_pyfunction!(sample_function, m)?)?;

    Ok(())
}
