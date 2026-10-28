# rete
> small + fast

![image](./rete.png)


**Goal**: Blazing fast, low-resource, small-scale neural networks in rust that run completely on the stack

## phase 1
> Legacy

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
> Current


Overhaul design, focusing on:
- moving to the stack
- making everything type-checked at compile time
- removing memory overhead of phase 1 implementation

Open items:
- Network initalization API (see [IDEAS.md](IDEAS.md)) and architecture (in progress...)

## phase 3
add an option for moving to the heap for larger models, include multi-threaded operations, optimize for multiple cpu cores
