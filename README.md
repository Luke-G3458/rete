# rete
> fast + small


**Goal**: Neural networks that train fast and run faster, written and optimized in rust

## phase 1
> Current


Basic forward and back prop functionality, allowing bad code and no optimization. Learn basics of neural networks


Usage:
```rust
use rete_nn::phase_1::Network;

fn main() {
    let mut nn = Network::new_random(vec![3, 4, 2]);

    let inputs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];
    let outputs = vec![
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
        vec![-1.0, 1.0],
        vec![1.0, -1.0],
    ];

    nn.train_batch(
        inputs
            .into_iter()
            .zip(outputs)
            .collect::<Vec<(Vec<f64>, Vec<f64>)>>(),
        0.05,
        10000,
    );

    let prediction = nn.predict(&vec![2.0, 3.0, -1.0]);

    println!(
        "results should be close to 1.0 and -1.0\n{} and {}",
        prediction[0].data(),
        prediction[1].data()
    );
}
```

Output:
```
results should be close to 1.0 and -1.0
0.9980129081740541 and -0.9977496787228837
```

## phase 2
> Up next


Overhaul design, focusing on and balancing two things:
- small memory footprint
- optimized computation speed for forward and backward propagation


Things to consider:
- This may require tensor types and operations
- Phase 1 has lots of memory overhead that may need to be addressed

## phase 3
multi-threaded, optimized for multiple cpu cores

## phase 4
gpu??

## Todo
- [ ] Create tests for benching different phases, evaluating both accuracy/correctness and performance
