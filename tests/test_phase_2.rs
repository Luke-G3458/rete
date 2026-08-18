use rete_nn::phase_2::*;

#[test]
pub fn create() {
    let test_1_a = Tensor::new([1, 2, 3]);
    let test_1_b = Tensor::new([1, 2, 3]);
    let test_2 = Tensor::new([[1, 2], [3, 4]]);
    let test_3 = Tensor::new([[1, 2], [3, 4], [5, 6]]);
}

#[test]
pub fn dot_multiply() {
    let test_1_a = Tensor::new([1, 2, 3]);
    let test_1_b = Tensor::new([1, 2, 3]);
    let result = test_1_a * test_1_b;
    assert_eq!(result, 14)
}
