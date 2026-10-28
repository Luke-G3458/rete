use rete_nn::phase_2::*;

#[test]
pub fn create() {
    let _test_1_a = Tensor::new([1.0_f32, 2.0_f32, 3.0_f32]);
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

#[test]
pub fn network_creation_size_comparisons() {
    use std::mem;

    // 1 -> 2 -> 2 -> 1

    // size = 52 bytes
    let foo = (
        Layer {
            weights: [Tensor::new([0.0_f32; 1]); 2],
            biases: [0.0_f32; 2],
        },
        (
            Layer {
                weights: [Tensor::new([0.0_f32; 2]); 2],
                biases: [0.0_f32; 2],
            },
            (
                Layer {
                    weights: [Tensor::new([0.0_f32; 2]); 1],
                    biases: [0.0_f32; 1],
                },
                (),
            ),
        ),
    );

    // size = 52 bytes
    let bar = Recursive::Tuple {
        layer: Layer {
            weights: [Tensor::new([0.0_f32; 1]); 2],
            biases: [0.0_f32; 2],
        },
        other: Recursive::Tuple {
            layer: Layer {
                weights: [Tensor::new([0.0_f32; 2]); 2],
                biases: [0.0_f32; 2],
            },
            other: Recursive::Tuple {
                layer: Layer {
                    weights: [Tensor::new([0.0_f32; 2]); 1],
                    biases: [0.0_f32; 1],
                },
                other: (),
            },
        },
    };

    let network = rete_nn::network![f32; 1, 2, 2, 1];

    assert_eq!(mem::size_of_val(&foo), mem::size_of_val(&bar));
    assert_eq!(mem::size_of_val(&foo), mem::size_of_val(&network));
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
