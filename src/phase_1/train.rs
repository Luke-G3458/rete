//! training loop for a neural network, utilizing backpropagation to update weights

use crate::phase_1::network::Network;
use crate::phase_1::value::Value;

/// Training loop for a neural network using backpropagation to update parameters. Requires
/// a mutable network, a dataset, a learning rate, and the number of epochs.
pub fn train(
    network: &mut Network,
    data: Vec<(Vec<f64>, Vec<f64>)>,
    rate: f64,
    epochs: usize,
) -> f64 {
    let mut loss = Value::new(0.0, None);
    for epoch in 0..epochs {
        loss = Value::new(0.0, None);
        let mut predictions = Vec::new();
        for (inputs, _) in &data {
            network.set_inputs(inputs.iter().map(|v| Value::new(*v, None)).collect());
            network.forward();
            predictions.push(network.forward());
        }
        for (prediction, (_, target)) in predictions.iter().zip(&data) {
            for (p, t) in prediction.iter().zip(target) {
                loss = loss + (p.clone() - Value::new(*t, None)).pow(2.0);
            }
        }
        network.zero_gradients();
        loss.set_gradient(1.0);
        loss.backward_recursive();
        println!("{}: {:?}", epoch, loss.data());
        network.bump(rate);
    }
    loss.data()
}
