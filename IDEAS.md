# Ideas and Brainstorming
## API and resulting architectural decision
### Example network
| layer 0 (input): 3 | layer 1: 10 | layer 2: 15 | layer 3 (output): 2 |

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
