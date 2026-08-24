use rete_nn::phase_2::*;

#[test]
pub fn create() {
    let _test_1_a = Tensor::new([1, 2, 3]);
    let _test_1_b = Tensor::new([1, 2, 3]);
    let _test_2 = Tensor::new([[1, 2], [3, 4]]);
    let _test_3 = Tensor::new([[1, 2], [3, 4], [5, 6]]);
}

#[test]
pub fn dot_multiply() {
    let test_1_a = Tensor::new([1, 2, 3]);
    let test_1_b = Tensor::new([1, 2, 3]);
    let result = test_1_a * test_1_b;
    assert_eq!(result, 14)
}

/* This failes to compile since arrays are empty

#[test]
pub fn dot_multiply_empty() {
    let a = Tensor::new([]);
    let b = Tensor::new([]);
    let result: i32 = a * b;
    println!("{:?}", result);
}

*/

#[test]
pub fn one_in_two_out_no_hidden_layers_nn() {
    let network = NetworkNoHidden::<f64, 1, 2>::new();
    assert_eq!(
        network,
        NetworkNoHidden {
            input: Tensor::new([0.0; 1]),
            layer: Layer {
                weights: [Tensor::new([0.0; 1]); 2],
                biases: [0.0; 2],
            },
        }
    )
}
