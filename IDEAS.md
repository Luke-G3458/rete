# Ideas and Brainstorming
## Example network
| layer 0 (input): 3 | layer 1: 10 | layer 2: 15 | layer 3 (output): 2 |

## API Syntax
### 1. Pythonic
``` rust
Network::new(3, 10, 15, 2)
```
Ergonomic, but a nightmare to implement, probably can't even implement

### 2. Macro
```rust
network![3, 10, 15, 2]
```
Most ergonomic while still possible in rust. Makeing macros is annoying though.

### 3. Methods
```rust
Network<3>::new()
    .layer<10>()
    .layer<15>()
    .layer<3>()
```
Not as ergonomic, but possible to implement and make type safe at compile time.

## Architecture
### 1. Fully concrete
```rust
struct Network0<T, const I: usize, const O: usize>
where
    T: Copy,
{
    inputs: [T; I],
    output: Layer<T, I, O>,
}

struct Network1<T, const I: usize, const N: usize, const O: usize>
where
    T: Copy,
{
    inputs: [T; I],
    hidden: Layer<T, I, N>,
    output: Layer<T, N, O>,
}

struct Network2<T, const I: usize, const A: usize, const B: usize, const O: usize>
where
    T: Copy,
{
    inputs: [T; I],
    hidden_1: Layer<T, I, A>,
    hidden_2: Layer<T, A, B>,
    output: Layer<T, B, O>,
}

//...
```
#### Downsides
- We'd have to create a macro to generate this, otherwise the API is aweful
- We'd need to create like a trait or something for all of them to actually implement forward and back propigation.

### 2. Tuple thing
```rust
let foo = network![3, 4, 4, 2]
```

```rust
// foo
(
    Layer<3,4>,
    (
        Layer<4,4>,
        (
            Layer<4,2>,
            ()
        )
    )
)
```
