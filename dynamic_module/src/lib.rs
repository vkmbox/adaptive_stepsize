pub mod stepsize;

use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};
use numpy::{PyReadonlyArrayDyn, PyArrayMethods};
use stepsize::eta_analytic_n2;

#[pymodule]
fn dynamic_module<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "eta_analytic_n2")]
    pub fn add_layer_data_py<'py>(pp: PyReadonlyArrayDyn<'py, i64>
            , qq0: PyReadonlyArrayDyn<'py, f32>, qq_test: PyReadonlyArrayDyn<'py, f32>, eta_test: f32, alpha: f32, beta: f32)
                -> PyResult<(f32, f32, f32)> {
        let result = 
            eta_analytic_n2(pp.to_owned_array(), qq0.to_owned_array(), qq_test.to_owned_array(), eta_test, alpha, beta)
            .unwrap();
        Ok(result)
    }

    Ok(())
}
