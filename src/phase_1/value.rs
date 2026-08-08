//! scalar value
use std::{
    cell::Cell,
    cmp::Ordering,
    collections::HashSet,
    fmt,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

/// value struct which stores the scalar value, a gradient, mathematical "parents", and the backward function which computes the gradient of the parents
#[derive(Clone)]
pub struct Value {
    data: f64,
    gradient: Rc<Cell<f64>>,
    parents: Option<Vec<Rc<Value>>>,
    backward: Option<Rc<dyn Fn()>>,
}

impl Value {
    pub fn new<T>(data: T, parents: Option<Vec<Rc<Value>>>) -> Self
    where
        T: Into<f64>,
    {
        Self {
            data: data.into(),
            gradient: Rc::new(Cell::new(0.0)),
            parents,
            backward: None,
        }
    }

    pub fn data(&self) -> f64 {
        self.data
    }

    pub fn gradient(&self) -> f64 {
        self.gradient.get()
    }

    pub fn set_gradient<T>(&self, gradient: T)
    where
        T: Into<f64>,
    {
        self.gradient.set(gradient.into());
    }

    pub fn reset_gradient(&self) {
        self.gradient.set(0.0);
    }

    fn add_gradient(&self, gradient: f64) {
        self.gradient.set(self.gradient.get() + gradient);
    }

    pub fn parents(&self) -> Option<&[Rc<Value>]> {
        self.parents.as_deref()
    }

    pub fn set_backward<F>(&mut self, backward: F)
    where
        F: Fn() + 'static,
    {
        self.backward = Some(Rc::new(backward));
    }

    /// run backward pass on this value
    pub fn backward(&self) {
        if let Some(backward) = &self.backward {
            backward();
        }
    }

    /// run backward pass on this value and all parent values in correct topological order
    pub fn backward_recursive(&self) {
        self.set_gradient(1.0);
        let mut topo: Vec<Rc<Value>> = Vec::new();
        let mut visited: HashSet<*const Cell<f64>> = HashSet::new();

        fn build_topo(
            value: Rc<Value>,
            topo: &mut Vec<Rc<Value>>,
            visited: &mut HashSet<*const Cell<f64>>,
        ) {
            let identity = Rc::as_ptr(&value.gradient);
            if !visited.insert(identity) {
                return;
            }

            if let Some(parents) = &value.parents {
                for parent in parents {
                    build_topo(Rc::clone(parent), topo, visited);
                }
            }

            topo.push(value);
        }

        build_topo(Rc::new(self.clone()), &mut topo, &mut visited);

        for node in topo.iter().rev() {
            node.backward();
        }
    }

    /// implementation of power
    pub fn pow(self, exponent: f64) -> Self {
        let data = self.data.powf(exponent);
        let base = Rc::new(self);
        let rc_exponent = Rc::new(Value::new(exponent, None));
        let mut output = Self::new(data, Some(vec![Rc::clone(&base), Rc::clone(&rc_exponent)]));
        let output_gradient = Rc::clone(&output.gradient);

        output.set_backward(move || {
            let local_gradient = exponent * base.data.powf(exponent - 1.0);
            base.add_gradient(local_gradient * output_gradient.get());
        });

        output
    }

    /// implementation of e^x
    pub fn exp(self) -> Self {
        let data = self.data.exp();
        let input = Rc::new(self);
        let mut output = Self::new(data, Some(vec![Rc::clone(&input)]));
        let output_gradient = Rc::clone(&output.gradient);

        output.set_backward(move || {
            input.add_gradient(data * output_gradient.get());
        });

        output
    }

    /// implementation of tanh activation function
    pub fn tanh(self) -> Self {
        let data = self.data.tanh();
        let input = Rc::new(self);
        let mut output = Self::new(data, Some(vec![Rc::clone(&input)]));
        let output_gradient = Rc::clone(&output.gradient);

        output.set_backward(move || {
            input.add_gradient(output_gradient.get() * (1.0 - data * data));
        });

        output
    }

    /// Bump all parameters by a given rate. Based on the gradient calculated by the backward pass.
    pub fn bump(&mut self, rate: f64) {
        self.data += -rate * self.gradient();
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let backward = self.backward.as_ref().map(|_| "<closure>");

        formatter
            .debug_struct("Value")
            .field("data", &self.data)
            .field("gradient", &self.gradient())
            .field("backward", &backward)
            .finish()
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.gradient() == other.gradient()
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.data.partial_cmp(&other.data) {
            Some(Ordering::Equal) => self.gradient().partial_cmp(&other.gradient()),
            ordering => ordering,
        }
    }
}

impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Self::new(data, None)
    }
}

/// implementation of addition
impl Add for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let data = self.data + other.data;
        let left = Rc::new(self);
        let right = Rc::new(other);
        let mut output = Self::new(data, Some(vec![Rc::clone(&left), Rc::clone(&right)]));
        let output_gradient = Rc::clone(&output.gradient);

        output.set_backward(move || {
            let gradient = output_gradient.get();
            left.add_gradient(gradient);
            right.add_gradient(gradient);
        });

        output
    }
}

/// implementation of multiplication
impl Mul for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        let data = self.data * other.data;
        let left = Rc::new(self);
        let right = Rc::new(other);
        let mut output = Self::new(data, Some(vec![Rc::clone(&left), Rc::clone(&right)]));
        let output_gradient = Rc::clone(&output.gradient);

        output.set_backward(move || {
            let gradient = output_gradient.get();
            left.add_gradient(right.data * gradient);
            right.add_gradient(left.data * gradient);
        });

        output
    }
}

/// implementation of division
impl Div for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        self * other.pow(-1.0)
    }
}

/// implementation of negation
impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * Self::new(-1.0, None)
    }
}

/// implementation of subtraction
impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}
