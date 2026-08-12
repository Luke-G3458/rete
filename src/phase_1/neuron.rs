//! neuron, one node in a neural network containing a set of weights and a bias

use super::value::Value;
use rand::{Rng, RngExt};

/// Neuron struct, containing parameters for a single neuron
#[derive(Debug)]
pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
}

impl Neuron {
    /// generate neuron with specified weights and bias
    pub fn new(weights: Vec<Value>, bias: Value) -> Self {
        Self { weights, bias }
    }

    /// generate neuron with random weights and bias
    pub fn new_random(input_count: usize) -> Self {
        let mut rng = rand::rng();
        Self::new_random_with_rng(input_count, &mut rng)
    }

    pub fn new_random_with_rng<R>(input_count: usize, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let weights = (0..input_count)
            .map(|_| Value::new(rng.random_range(-1.0..1.0), None))
            .collect();
        let bias = Value::new(rng.random_range(-1.0..1.0), None);

        Self { weights, bias }
    }

    /// run forward pass
    pub fn forward(&self, inputs: &[Value]) -> Value {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "neuron expected {} inputs, but received {}",
            self.weights.len(),
            inputs.len()
        );

        let weighted_sum = inputs
            .iter()
            .zip(&self.weights)
            .fold(Value::new(0.0, None), |output, (input, weight)| {
                output + input.clone() * weight.clone()
            });

        (weighted_sum + self.bias.clone()).tanh()
    }

    /// zero all gradients of parameters
    pub fn zero_gradients(&self) {
        for weight in &self.weights {
            weight.reset_gradient();
        }
        self.bias.reset_gradient();
    }

    /// bump all parameters by given rate
    pub fn bump(&mut self, rate: f64) {
        for weight in &mut self.weights {
            weight.bump(rate);
        }

        self.bias.bump(rate);
    }
}
