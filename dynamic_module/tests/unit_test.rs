use dynamic_module::stepsize::eta_analytic_n2;
use numpy::ndarray::prelude::*;

#[test]
fn it_eta_analytic_n2() {
    //Python::with_gil(|py| {
    let qq0 = ArrayD::<f32>::zeros(IxDyn(&[26, 64]));
        //arr2(&[[0.11,0.12,0.13,0.14], [0.21,0.22,0.23,0.24], [0.211,0.221,0.231,0.241]]); //.to_owned();
    let qq_test = ArrayD::<f32>::zeros(IxDyn(&[26, 64]));
        //array![[0.12,0.13,0.14,0.15], [0.31,0.33,0.32,0.35], [0.315,0.321,0.365,0.387]]; //.to_owned();
    let pp = ArrayD::<i64>::zeros(IxDyn(&[64]));
    //from_shape_vec
    //pp[0] = 1;
        //pyarray![py,[1, 2, 0, 2]].to_owned_array();
    let eta_test: f32 = 0.00001;
    let eta_expected = eta_analytic_n2(pp, qq0, qq_test, eta_test, 0.001, 0.00001);
    assert!(eta_expected.is_ok());
    assert_eq!(eta_expected.unwrap(), (0.0, 8.0, 0.0));
    //})
    //ArrayD::<f64>::ones(IxDyn(&[26, 64, 250, 28*28]));
}