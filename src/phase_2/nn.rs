use super::tensor::Tensor;

#[derive(Debug, PartialEq)]
pub struct Layer<T, const I: usize, const N: usize>
where
    T: Copy,
{
    pub weights: [Tensor<[T; I]>; N],
    pub biases: [T; N],
}

#[derive(Debug, PartialEq)]
pub struct NetworkNoHidden<T, const I: usize, const N: usize>
where
    T: From<f64> + Copy,
{
    pub input: Tensor<[T; I]>,
    pub layer: Layer<T, I, N>,
}

impl<T, const I: usize, const N: usize> NetworkNoHidden<T, I, N>
where
    T: From<f64> + Copy,
{
    pub fn new() -> Self {
        Self {
            input: Tensor::new([T::from(0.0); I]),
            layer: Layer {
                weights: [Tensor::new([T::from(0.0); I]); N],
                biases: [T::from(0.0); N],
            },
        }
    }
}
