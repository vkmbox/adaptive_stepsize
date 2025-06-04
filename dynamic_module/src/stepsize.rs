use numpy::ndarray::ArrayD;

static EPSILON: f32 = 1e-9;

pub fn eta_analytic_n2(pp: ArrayD<i64>, qq0: ArrayD<f32>, qq_test: ArrayD<f32>, eta_test: f32, alpha: f32, beta: f32)
        -> Result<(f32, f32, f32), &'static str> {
    //delta_pq, delta_qq = pp-qq0, qq_test-qq0
    let mut delta_pq = -&qq0;
    for ii in 0..pp.shape()[0] {
        delta_pq[[pp[ii] as usize, ii]] += 1.0;
    }
    let delta_qq = &qq_test-&qq0;
    let norm_pq = matrix_norm_fro(&delta_pq);
    let norm_qq = matrix_norm_fro(&delta_qq);
    let cos_phi1 = matrix_dot(&delta_pq, &delta_qq)/(norm_pq*norm_qq + EPSILON);
    let eta_next = sign(cos_phi1)*((norm_pq*cos_phi1*eta_test*alpha)/(norm_qq + beta)).abs();
    Ok((eta_next, norm_pq, norm_qq))
}

fn matrix_norm_fro(matrix: &ArrayD<f32>) -> f32 {
    let dims = matrix.shape();
    let mut sum : f32 = 0.0;
    for row in 0..dims[0] {
        for col in 0..dims[1] {
            let val: f32 = matrix[[row, col]];
            sum += val.powf(2.0);
        }
    }
    return sum.sqrt();
}

fn matrix_dot(matrix1: &ArrayD<f32>, matrix2: &ArrayD<f32>) -> f32 {
    let dims = matrix1.shape();
    let mut sum : f32 = 0.0;
    for row in 0..dims[0] {
        for col in 0..dims[1] {
            let val1: f32 = matrix1[[row, col]];
            let val2: f32 = matrix2[[row, col]];
            sum += val1 * val2;
        }
    }
    return sum;
}

#[allow(unused_parens)]
fn sign(value: f32) -> f32 {
    if (value >= 0.0) {
        1.0
    } else {
        -1.0
    }
}