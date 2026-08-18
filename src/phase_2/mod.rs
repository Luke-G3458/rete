use std::ops::{AddAssign, Mul};

#[derive(Debug)]
pub struct Tensor<A> {
    value: A,
}

impl<A> Tensor<A> {
    pub fn new(data: A) -> Self {
        Self { value: data }
    }
}

impl<T, const S: usize> Mul for Tensor<[T; S]>
where
    T: Mul<Output = T> + AddAssign,
{
    type Output = T;
    fn mul(self, other: Self) -> T {
        const {
            assert!(S > 0);
        }
        let mut result: T;
        let mut products = self.value.into_iter().zip(other.value.into_iter());
        let Some((i, j)) = products.next() else {
            panic!()
        };
        result = i * j;
        for (i, j) in products {
            result += i * j
        }
        result
    }
}
