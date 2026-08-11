use rand::{SeedableRng, rngs::StdRng};
use rete_nn::phase_1::{Layer, Network, Neuron, Value};
use std::{cell::Cell, rc::Rc};

fn assert_approximately_equal(actual: Value, expected: f64) {
    let tolerance = f64::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual.data() - expected).abs() <= tolerance,
        "expected {expected}, got {}",
        actual.data()
    );
}

fn assert_number_approximately_equal(actual: f64, expected: f64) {
    let tolerance = f64::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

fn assert_number_close(actual: f64, expected: f64) {
    let tolerance = 1e-12 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected}, got {actual}"
    );
}

// Tests for the Value struct

#[test]
fn value_adds() {
    assert_eq!(
        Value::new(1, None) + Value::new(2, None),
        Value::new(3, None)
    );
    assert_eq!(
        Value::new(-1.5, None) + Value::new(0.5, None),
        Value::new(-1.0, None)
    );
}

#[test]
fn addition_backward_accumulates_gradient_into_both_inputs() {
    let left = Value::new(2, None);
    let right = Value::new(3, None);
    let sum = left.clone() + right.clone();

    sum.set_gradient(2.5);
    sum.backward();

    assert_eq!(left.gradient(), 2.5);
    assert_eq!(right.gradient(), 2.5);

    sum.backward();

    assert_eq!(left.gradient(), 5.0);
    assert_eq!(right.gradient(), 5.0);
}

#[test]
fn backward_recursive_runs_shared_nodes_once_in_reverse_topological_order() {
    let input = Value::new(2, None);
    let square = input.clone() * input.clone();
    let output = square.clone() + square.clone();

    output.backward_recursive();

    assert_eq!(output.gradient(), 1.0);
    assert_eq!(square.gradient(), 2.0);
    assert_eq!(input.gradient(), 8.0);
}

#[test]
fn backward_recursive_treats_equal_values_as_distinct_nodes() {
    let left = Value::new(2, None);
    let right = Value::new(2, None);
    let product = left.clone() * right.clone();

    product.backward_recursive();

    assert_eq!(left.gradient(), 2.0);
    assert_eq!(right.gradient(), 2.0);
}

#[test]
fn value_multiplies() {
    assert_eq!(
        Value::new(3, None) * Value::new(2, None),
        Value::new(6, None)
    );
    assert_eq!(
        Value::new(-1.5, None) * Value::new(2, None),
        Value::new(-3.0, None)
    );
}

#[test]
fn multiplication_backward_accumulates_gradients_into_both_inputs() {
    let left = Value::new(2, None);
    let right = Value::new(3, None);
    let product = left.clone() * right.clone();

    left.set_gradient(1.0);
    right.set_gradient(-1.0);
    product.set_gradient(4.0);
    product.backward();

    assert_eq!(left.gradient(), 13.0);
    assert_eq!(right.gradient(), 7.0);
}

#[test]
fn value_raises_to_a_power() {
    assert_eq!(Value::new(2, None).pow(3.0), Value::new(8, None));
    assert_eq!(Value::new(4, None).pow(-1.0), Value::new(0.25, None));
    assert_eq!(Value::new(9, None).pow(0.5), Value::new(3, None));
}

#[test]
fn power_backward_accumulates_gradient_into_the_base() {
    let base = Value::new(2, None);
    let exponent = 3.0;
    let power = base.clone().pow(exponent);

    base.set_gradient(1.0);
    power.set_gradient(0.5);
    power.backward();

    assert_eq!(base.gradient(), 7.0);
}

#[test]
fn value_divides_by_multiplying_by_the_reciprocal() {
    let dividend = Value::new(7, None);
    let divisor = Value::new(2, None);

    assert_eq!(dividend.clone() / divisor.clone(), Value::new(3.5, None));
    assert_eq!(
        dividend.clone() / divisor.clone(),
        dividend * divisor.pow(-1.0)
    );
}

#[test]
fn value_negates() {
    assert_eq!(-Value::new(6, None), Value::new(-6, None));
    assert_eq!(-Value::new(-2.5, None), Value::new(2.5, None));
}

#[test]
fn value_subtracts_by_adding_the_negative() {
    let left = Value::new(1, None);
    let right = Value::new(2, None);

    assert_eq!(left.clone() - right.clone(), Value::new(-1, None));
    assert_eq!(left.clone() - right.clone(), left + (-right));
}

#[test]
fn value_computes_e_to_its_power() {
    assert_eq!(Value::new(0, None).exp(), Value::new(1, None));
    assert_approximately_equal(Value::new(1, None).exp(), std::f64::consts::E);
    assert_approximately_equal(Value::new(-1, None).exp(), 1.0 / std::f64::consts::E);
}

#[test]
fn exponential_backward_accumulates_gradient_into_its_input() {
    let input = Value::new(1, None);
    let output = input.clone().exp();

    input.set_gradient(1.0);
    output.set_gradient(2.0);
    output.backward();

    assert_number_approximately_equal(input.gradient(), 1.0 + 2.0 * std::f64::consts::E);
}

#[test]
fn gradient_can_be_reset_to_zero() {
    let value = Value::new(1, None);
    let shared_value = value.clone();

    value.set_gradient(42.0);
    shared_value.reset_gradient();

    assert_eq!(value.gradient(), 0.0);
    assert_eq!(shared_value.gradient(), 0.0);
}

#[test]
fn backward_can_be_set_to_a_capturing_closure_and_run_later() {
    let was_called = Rc::new(Cell::new(false));
    let called_from_closure = Rc::clone(&was_called);
    let mut value = Value::new(1, None);

    value.backward();
    assert!(!was_called.get());

    value.set_backward(move || called_from_closure.set(true));
    value.backward();

    assert!(was_called.get());
}

#[test]
fn value_accepts_optional_parents() {
    let parent = Rc::new(Value::new(1, None));
    let value = Value::new(2, Some(vec![Rc::clone(&parent)]));
    let parents = value.parents().expect("value should have a parent");

    assert_eq!(parents.len(), 1);
    assert!(Rc::ptr_eq(&parents[0], &parent));
}

#[test]
fn random_neuron_has_one_weight_per_input_and_a_random_bias() {
    let mut rng = StdRng::seed_from_u64(7);
    let neuron = Neuron::new_random_with_rng(4, &mut rng);

    assert_eq!(neuron.weights.len(), 4);
    assert!(
        neuron
            .weights
            .iter()
            .all(|weight| (-1.0..1.0).contains(&weight.data()))
    );
    assert!((-1.0..1.0).contains(&neuron.bias.data()));
}

#[test]
fn seeded_random_neuron_initialization_is_reproducible() {
    let mut first_rng = StdRng::seed_from_u64(42);
    let mut second_rng = StdRng::seed_from_u64(42);
    let first = Neuron::new_random_with_rng(3, &mut first_rng);
    let second = Neuron::new_random_with_rng(3, &mut second_rng);

    assert_eq!(first.weights, second.weights);
    assert_eq!(first.bias, second.bias);
}

#[test]
fn random_network_uses_each_adjacent_pair_of_widths() {
    let mut rng = StdRng::seed_from_u64(11);
    let network = Network::new_random_with_rng(vec![2, 3, 1], &mut rng);

    assert_eq!(network.layers.len(), 3);
    assert_eq!(network.inputs().len(), 2);
    assert!(network.inputs().iter().all(|input| input.data() == 0.0));

    let hidden = network.layers[1]
        .neurons()
        .expect("second layer should be dense");
    let output = network.layers[2]
        .neurons()
        .expect("third layer should be dense");

    assert_eq!(hidden.len(), 3);
    assert_eq!(output.len(), 1);
    assert!(hidden.iter().all(|neuron| neuron.weights.len() == 2));
    assert_eq!(output[0].weights.len(), 3);

    for layer in network.layers.iter().skip(1) {
        for neuron in layer.neurons().expect("non-input layers should be dense") {
            assert!((-1.0..1.0).contains(&neuron.bias.data()));
            assert!(
                neuron
                    .weights
                    .iter()
                    .all(|weight| (-1.0..1.0).contains(&weight.data()))
            );
        }
    }
}

#[test]
fn neuron_forward_adds_its_bias_to_the_weighted_sum() {
    let neuron = Neuron::new(
        vec![Value::new(2.0, None), Value::new(-1.0, None)],
        Value::new(0.5, None),
    );
    let inputs = [Value::new(3.0, None), Value::new(4.0, None)];

    assert_approximately_equal(neuron.forward(&inputs), 2.5_f64.tanh());
}

#[test]
fn network_forward_passes_each_layers_outputs_to_the_next_layer() {
    let inputs = Layer::input(vec![Value::new(4.0, None)]);
    let first = Layer::new(vec![Neuron::new(
        vec![Value::new(2.0, None)],
        Value::new(1.0, None),
    )]);
    let second = Layer::new(vec![Neuron::new(
        vec![Value::new(3.0, None)],
        Value::new(-1.0, None),
    )]);
    let network = Network::new(vec![inputs, first, second]);

    let outputs = network.forward();
    let expected = (3.0 * 9.0_f64.tanh() - 1.0).tanh();

    assert_eq!(outputs.len(), 1);
    assert_approximately_equal(outputs[0].clone(), expected);
}

#[test]
fn network_forward_and_backward_pass_produce_expected_values() {
    let input_data: f64 = 0.5;
    let first_weight_data = 0.4;
    let first_bias_data = 0.5;
    let second_weight_data = 0.3;
    let second_bias_data = 0.5;

    let input = Value::new(input_data, None);
    let first_weight = Value::new(first_weight_data, None);
    let first_bias = Value::new(first_bias_data, None);
    let second_weight = Value::new(second_weight_data, None);
    let second_bias = Value::new(second_bias_data, None);

    let network = Network::new(vec![
        Layer::input(vec![input.clone()]),
        Layer::new(vec![Neuron::new(
            vec![first_weight.clone()],
            first_bias.clone(),
        )]),
        Layer::new(vec![Neuron::new(
            vec![second_weight.clone()],
            second_bias.clone(),
        )]),
    ]);

    let first_pre_activation = input_data * first_weight_data + first_bias_data;
    let first_output = first_pre_activation.tanh();
    let second_pre_activation = first_output * second_weight_data + second_bias_data;
    let expected_output = second_pre_activation.tanh();

    let outputs = network.forward();
    assert_eq!(outputs.len(), 1);
    assert_number_approximately_equal(outputs[0].data(), expected_output);

    outputs[0].backward_recursive();

    let second_activation_gradient = 1.0 - expected_output.powi(2);
    let first_activation_gradient = 1.0 - first_output.powi(2);
    let gradient_into_first_layer = second_activation_gradient * second_weight_data;

    assert_eq!(outputs[0].gradient(), 1.0);
    assert_number_approximately_equal(
        input.gradient(),
        gradient_into_first_layer * first_activation_gradient * first_weight_data,
    );
    assert_number_approximately_equal(
        first_weight.gradient(),
        gradient_into_first_layer * first_activation_gradient * input_data,
    );
    assert_number_approximately_equal(
        first_bias.gradient(),
        gradient_into_first_layer * first_activation_gradient,
    );
    assert_number_approximately_equal(
        second_weight.gradient(),
        second_activation_gradient * first_output,
    );
    assert_number_approximately_equal(second_bias.gradient(), second_activation_gradient);
}

#[test]
fn random_network_forward_and_backward_pass_match_its_generated_parameters() {
    let input_data = [0.25, -0.75];
    let mut rng = StdRng::seed_from_u64(37);
    let mut network = Network::new_random_with_rng(vec![2, 3, 1], &mut rng);
    network.set_inputs(
        input_data
            .iter()
            .map(|input| Value::new(*input, None))
            .collect(),
    );

    let hidden_neurons = network.layers[1]
        .neurons()
        .expect("second layer should be dense");
    let output_neurons = network.layers[2]
        .neurons()
        .expect("third layer should be dense");
    assert_eq!(hidden_neurons.len(), 3);
    assert_eq!(output_neurons.len(), 1);

    let hidden_weights: Vec<Vec<f64>> = hidden_neurons
        .iter()
        .map(|neuron| neuron.weights.iter().map(Value::data).collect())
        .collect();
    let hidden_biases: Vec<f64> = hidden_neurons
        .iter()
        .map(|neuron| neuron.bias.data())
        .collect();
    let output_neuron = &output_neurons[0];
    let output_weights: Vec<f64> = output_neuron.weights.iter().map(Value::data).collect();
    let output_bias = output_neuron.bias.data();

    assert!(
        hidden_weights
            .iter()
            .flatten()
            .chain(output_weights.iter())
            .chain(hidden_biases.iter())
            .chain(std::iter::once(&output_bias))
            .all(|parameter| (-1.0..1.0).contains(parameter))
    );

    let hidden_weighted_sums: Vec<f64> = hidden_weights
        .iter()
        .map(|weights| {
            input_data
                .iter()
                .zip(weights)
                .map(|(input, weight)| input * weight)
                .sum()
        })
        .collect();
    let hidden_outputs: Vec<f64> = hidden_weighted_sums
        .iter()
        .zip(&hidden_biases)
        .map(|(weighted_sum, bias)| (weighted_sum + bias).tanh())
        .collect();
    let output_weighted_sum: f64 = hidden_outputs
        .iter()
        .zip(&output_weights)
        .map(|(input, weight)| input * weight)
        .sum();
    let expected_output = (output_weighted_sum + output_bias).tanh();

    let outputs = network.forward();
    assert_eq!(outputs.len(), 1);
    assert_number_close(outputs[0].data(), expected_output);

    outputs[0].backward_recursive();

    let output_activation_gradient = 1.0 - expected_output.powi(2);
    let mut expected_input_gradients = [0.0; 2];

    for hidden_index in 0..3 {
        let hidden_activation_gradient = 1.0 - hidden_outputs[hidden_index].powi(2);
        let gradient_into_hidden =
            output_activation_gradient * output_weights[hidden_index] * hidden_activation_gradient;

        for input_index in 0..2 {
            expected_input_gradients[input_index] +=
                gradient_into_hidden * hidden_weights[hidden_index][input_index];
            assert_number_close(
                hidden_neurons[hidden_index].weights[input_index].gradient(),
                gradient_into_hidden * input_data[input_index],
            );
        }

        assert_number_close(
            hidden_neurons[hidden_index].bias.gradient(),
            gradient_into_hidden,
        );
        assert_number_close(
            output_neuron.weights[hidden_index].gradient(),
            output_activation_gradient * hidden_outputs[hidden_index],
        );
    }

    for (input, expected_gradient) in network.inputs().iter().zip(expected_input_gradients) {
        assert_number_close(input.gradient(), expected_gradient);
    }
    assert_number_close(output_neuron.bias.gradient(), output_activation_gradient);
}

#[test]
fn network_inputs_can_be_replaced() {
    let mut rng = StdRng::seed_from_u64(19);
    let mut network = Network::new_random_with_rng(vec![2, 1], &mut rng);

    network.set_inputs(vec![Value::new(3.0, None), Value::new(-2.0, None)]);

    assert_eq!(network.inputs()[0].data(), 3.0);
    assert_eq!(network.inputs()[1].data(), -2.0);
    assert_eq!(network.forward().len(), 1);
}

#[test]
fn network_can_zero_all_input_weight_and_bias_gradients() {
    let mut rng = StdRng::seed_from_u64(31);
    let network = Network::new_random_with_rng(vec![2, 3, 1], &mut rng);

    for input in network.inputs() {
        input.set_gradient(1.0);
    }
    for layer in network.layers.iter().skip(1) {
        for neuron in layer.neurons().expect("non-input layers should be dense") {
            for weight in &neuron.weights {
                weight.set_gradient(2.0);
            }
            neuron.bias.set_gradient(3.0);
        }
    }

    network.zero_gradients();

    assert!(network.inputs().iter().all(|input| input.gradient() == 0.0));
    for layer in network.layers.iter().skip(1) {
        for neuron in layer.neurons().expect("non-input layers should be dense") {
            assert!(neuron.weights.iter().all(|weight| weight.gradient() == 0.0));
            assert_eq!(neuron.bias.gradient(), 0.0);
        }
    }
}

#[test]
#[should_panic(expected = "network expected 2 inputs, but received 1")]
fn network_rejects_the_wrong_number_of_inputs() {
    let mut rng = StdRng::seed_from_u64(23);
    let mut network = Network::new_random_with_rng(vec![2, 1], &mut rng);

    network.set_inputs(vec![Value::new(3.0, None)]);
}

#[test]
fn network_debug_output_includes_inputs_layers_weights_and_biases() {
    let mut rng = StdRng::seed_from_u64(29);
    let network = Network::new_random_with_rng(vec![2, 1], &mut rng);
    let debug = format!("{network:#?}");

    assert!(debug.contains("Network"));
    assert!(debug.contains("Input"));
    assert!(debug.contains("Dense"));
    assert!(debug.contains("weights"));
    assert!(debug.contains("bias"));
}

#[test]
fn create_and_train_network_from_data() {
    let mut nn = Network::new_random(vec![3, 4, 1]);

    let inputs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let outputs = vec![vec![1.0], vec![-1.0], vec![-1.0], vec![1.0]];

    let loss = nn.train_batch(
        inputs
            .into_iter()
            .zip(outputs)
            .collect::<Vec<(Vec<f64>, Vec<f64>)>>(),
        0.05,
        1000,
    );

    assert!(loss < 1.0)
}
