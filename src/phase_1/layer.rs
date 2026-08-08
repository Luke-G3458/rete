//! layer, a collection of neurons in a neural network

use super::{neuron::Neuron, value::Value};
use rand::Rng;

/// Layer enum, representing a layer in a neural network, either an input layer or a dense layer (hidden or output).
#[derive(Debug)]
pub enum Layer {
    Input(Vec<Value>),
    Dense(Vec<Neuron>),
}

impl Layer {
    /// generate an input layer
    pub fn input(values: Vec<Value>) -> Self {
        Self::Input(values)
    }

    /// generate a dense layer
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self::Dense(neurons)
    }

    /// generate a dense layer with random weights
    pub fn new_random(input_count: usize, output_count: usize) -> Self {
        let mut rng = rand::rng();
        Self::new_random_with_rng(input_count, output_count, &mut rng)
    }

    pub fn new_random_with_rng<R>(input_count: usize, output_count: usize, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let neurons = (0..output_count)
            .map(|_| Neuron::new_random_with_rng(input_count, rng))
            .collect();

        Self::Dense(neurons)
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Input(values) => values.len(),
            Self::Dense(neurons) => neurons.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn inputs(&self) -> Option<&[Value]> {
        match self {
            Self::Input(values) => Some(values),
            Self::Dense(_) => None,
        }
    }

    pub fn neurons(&self) -> Option<&[Neuron]> {
        match self {
            Self::Input(_) => None,
            Self::Dense(neurons) => Some(neurons),
        }
    }

    /// run forward pass
    pub fn forward(&self, inputs: &[Value]) -> Vec<Value> {
        match self {
            Self::Input(values) => values.clone(),
            Self::Dense(neurons) => neurons
                .iter()
                .map(|neuron| neuron.forward(inputs))
                .collect(),
        }
    }

    /// zero all gradients of parameters
    pub fn zero_gradients(&self) {
        match self {
            Self::Input(values) => {
                for value in values {
                    value.reset_gradient();
                }
            }
            Self::Dense(neurons) => {
                for neuron in neurons {
                    neuron.zero_gradients();
                }
            }
        }
    }

    /// bump all parameters by a given rate
    pub fn bump(&mut self, rate: f64) {
        match self {
            Self::Dense(neurons) => {
                for neuron in neurons {
                    neuron.bump(rate);
                }
            }
            Self::Input(_) => {}
        }
    }
}
