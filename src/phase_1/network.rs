//! neural network

use super::{layer::Layer, value::Value};
use rand::Rng;

/// Neural network struct containing associated layers
#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    /// generate a network from fully pre-defined parameters
    pub fn new(layers: Vec<Layer>) -> Self {
        assert!(
            matches!(layers.first(), Some(Layer::Input(_))),
            "the first network layer must be an input layer"
        );
        assert!(
            layers
                .iter()
                .skip(1)
                .all(|layer| matches!(layer, Layer::Dense(_))),
            "only the first network layer can be an input layer"
        );

        Self { layers }
    }

    /// generate a neural network with a specified shape and randomly generated parameters
    pub fn new_random(structure: Vec<usize>) -> Self {
        let mut rng = rand::rng();
        Self::new_random_with_rng(structure, &mut rng)
    }

    pub fn new_random_with_rng<R>(structure: Vec<usize>, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        assert!(
            structure.len() >= 2,
            "network structure must include input and output widths"
        );

        let inputs = (0..structure[0]).map(|_| Value::new(0.0, None)).collect();
        let mut layers = vec![Layer::input(inputs)];
        layers.extend(
            structure
                .windows(2)
                .map(|widths| Layer::new_random_with_rng(widths[0], widths[1], rng)),
        );

        Self::new(layers)
    }

    pub fn inputs(&self) -> &[Value] {
        self.layers[0]
            .inputs()
            .expect("a network always starts with an input layer")
    }

    /// set the inputs of the network
    pub fn set_inputs(&mut self, inputs: Vec<Value>) {
        let expected = self.inputs().len();
        assert_eq!(
            inputs.len(),
            expected,
            "network expected {expected} inputs, but received {}",
            inputs.len()
        );

        self.layers[0] = Layer::input(inputs);
    }

    /// run forward pass
    pub fn forward(&self) -> Vec<Value> {
        self.layers
            .iter()
            .fold(Vec::new(), |values, layer| layer.forward(&values))
    }

    /// zero the gradients of all parameters
    pub fn zero_gradients(&self) {
        for layer in &self.layers {
            layer.zero_gradients();
        }
    }

    /// bump all parameters by a given rate
    pub fn bump(&mut self, rate: f64) {
        for layer in &mut self.layers {
            layer.bump(rate);
        }
    }
}
