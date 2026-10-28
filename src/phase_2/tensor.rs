use std::ops::{Add, Mul};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Tensor<A>
where
    A: Copy,
{
    value: A,
}

impl<A> Tensor<A>
where
    A: Copy,
{
    pub fn new(data: A) -> Self {
        Self { value: data }
    }
}

impl<T, const S: usize> Mul for Tensor<[T; S]>
where
    T: Mul<Output = T> + Add<Output = T> + Copy,
{
    type Output = T;
    fn mul(self, other: Self) -> T {
        const {
            assert!(S > 0, "array must be non-empty");
        }
        let Some(result) = self
            .value
            .into_iter()
            .zip(other.value.into_iter())
            .map(|(i, j)| i * j)
            .reduce(|accumulator, next| accumulator + next)
        else {
            panic!("unreachable")
        };
        result
    }
}
