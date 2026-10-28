use super::tensor::Tensor;

#[derive(Debug, PartialEq)]
pub struct Layer<T, const I: usize, const N: usize>
where
    T: Copy,
{
    pub weights: [Tensor<[T; I]>; N],
    pub biases: [T; N],
}

pub enum Recursive<T, const I: usize, const N: usize, R>
where
    T: Copy,
{
    Tuple { layer: Layer<T, I, N>, other: R },
}

pub struct Network<L> {
    pub layers: L,
}

#[macro_export]
macro_rules! network {
    // End of the recursive chain.
    (@layers $ty:ty; $last:expr) => {
        ()
    };

    // Build one layer and the remaining chain.
    (@layers $ty:ty; $input:expr, $output:expr $(, $rest:expr)*) => {
        Recursive::Tuple {
            layer: Layer {
                weights: [
                    Tensor::new([<$ty>::default(); $input]);
                    $output
                ],
                biases: [<$ty>::default(); $output],
            },
            other: $crate::network!(@layers $ty; $output $(, $rest)*),
        }
    };

    // Public entry point.
    ($ty:ty; $input:expr, $output:expr $(, $rest:expr)* $(,)?) => {
        Network {
            layers: $crate::network!(@layers $ty; $input, $output $(, $rest)*),
        }
    };
}
