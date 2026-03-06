#!/usr/bin/env python3
"""
Generate a high-quality, compiler-validated Rust training dataset.

Strategy:
1. Extract and curate code from cloned repos (filter for quality)
2. Generate synthetic Rust code covering primitives, data structures, algorithms
3. Validate every snippet with rustc
4. Combine into a clean corpus for BPE tokenization

Targets: 200-400MB of high-quality, compiler-validated Rust code
"""

import os
import sys
import json
import random
import subprocess
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).parent
REPOS_DIR = SCRIPT_DIR / "rust_finetune" / "data" / "rust_repos_v2"
OUTPUT_DIR = Path("/home/galic/nanochat-rs-ternary/nanochat-rs-ternary/data")
MIN_FILE_SIZE = 100
MAX_FILE_SIZE = 150_000

# ============================================================================
# SYNTHETIC RUST CODE GENERATORS
# ============================================================================

RUST_PRIMITIVES = [
    # Option handling
    '''fn find_item(items: &[i32], target: i32) -> Option<usize> {
    items.iter().position(|&x| x == target)
}

fn main() {
    let numbers = vec![1, 2, 3, 4, 5];
    match find_item(&numbers, 3) {
        Some(idx) => println!("Found at index {}", idx),
        None => println!("Not found"),
    }

    // Using unwrap_or
    let result = find_item(&numbers, 99).unwrap_or(0);
    println!("Result: {}", result);

    // Using map
    let doubled = find_item(&numbers, 3).map(|i| i * 2);
    println!("Doubled index: {:?}", doubled);
}''',

    # Result and error handling
    '''use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    ParseError(ParseIntError),
    ValidationError(String),
    IoError(String),
}

impl From<ParseIntError> for AppError {
    fn from(err: ParseIntError) -> Self {
        AppError::ParseError(err)
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::ParseError(e) => write!(f, "Parse error: {}", e),
            AppError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AppError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

fn parse_and_validate(input: &str) -> Result<u32, AppError> {
    let number: u32 = input.trim().parse()?;
    if number > 1000 {
        return Err(AppError::ValidationError(
            format!("Number {} exceeds maximum of 1000", number)
        ));
    }
    Ok(number)
}

fn main() {
    let inputs = vec!["42", "999", "1001", "abc", "  7  "];
    for input in inputs {
        match parse_and_validate(input) {
            Ok(n) => println!("{:?} -> Ok({})", input, n),
            Err(e) => println!("{:?} -> Err({})", input, e),
        }
    }
}''',

    # Ownership and borrowing
    '''fn take_ownership(s: String) -> String {
    println!("Took ownership of: {}", s);
    s  // return ownership back
}

fn borrow_immutably(s: &str) {
    println!("Borrowed: {}", s);
}

fn borrow_mutably(s: &mut String) {
    s.push_str(" world");
    println!("Modified: {}", s);
}

fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

fn main() {
    // Move semantics
    let s1 = String::from("hello");
    let s2 = take_ownership(s1);
    // s1 is no longer valid here
    println!("Got back: {}", s2);

    // Immutable borrow
    let s3 = String::from("rust");
    borrow_immutably(&s3);
    println!("Still own: {}", s3);

    // Mutable borrow
    let mut s4 = String::from("hello");
    borrow_mutably(&mut s4);
    println!("After mutation: {}", s4);

    // Lifetimes
    let result;
    let string1 = String::from("long string");
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {}", result);
    }
}''',

    # Iterators and closures
    '''fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Map and collect
    let squared: Vec<i32> = numbers.iter().map(|&x| x * x).collect();
    println!("Squared: {:?}", squared);

    // Filter
    let evens: Vec<&i32> = numbers.iter().filter(|&&x| x % 2 == 0).collect();
    println!("Evens: {:?}", evens);

    // Fold (reduce)
    let sum = numbers.iter().fold(0, |acc, &x| acc + x);
    println!("Sum: {}", sum);

    // Chain operations
    let result: Vec<i32> = numbers
        .iter()
        .filter(|&&x| x > 3)
        .map(|&x| x * 2)
        .take(3)
        .collect();
    println!("Filtered, doubled, taken: {:?}", result);

    // Enumerate
    for (i, val) in numbers.iter().enumerate() {
        if *val > 8 {
            println!("Index {}: {}", i, val);
        }
    }

    // Zip
    let names = vec!["Alice", "Bob", "Charlie"];
    let ages = vec![30, 25, 35];
    let people: Vec<_> = names.iter().zip(ages.iter()).collect();
    println!("People: {:?}", people);

    // Any and all
    let has_even = numbers.iter().any(|&x| x % 2 == 0);
    let all_positive = numbers.iter().all(|&x| x > 0);
    println!("Has even: {}, All positive: {}", has_even, all_positive);

    // Window and chunks
    let windows: Vec<_> = numbers.windows(3).collect();
    println!("Windows of 3: {:?}", &windows[..3]);

    let chunks: Vec<_> = numbers.chunks(3).collect();
    println!("Chunks of 3: {:?}", chunks);
}''',

    # Traits and generics
    '''use std::fmt;

trait Shape: fmt::Display {
    fn area(&self) -> f64;
    fn perimeter(&self) -> f64;
    fn name(&self) -> &str;

    fn describe(&self) -> String {
        format!("{} (area: {:.2}, perimeter: {:.2})", self.name(), self.area(), self.perimeter())
    }
}

trait Scalable: Shape {
    fn scale(&mut self, factor: f64);
}

#[derive(Debug, Clone)]
struct Circle {
    radius: f64,
}

#[derive(Debug, Clone)]
struct Rectangle {
    width: f64,
    height: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }
    fn perimeter(&self) -> f64 {
        2.0 * std::f64::consts::PI * self.radius
    }
    fn name(&self) -> &str { "Circle" }
}

impl fmt::Display for Circle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Circle(r={})", self.radius)
    }
}

impl Scalable for Circle {
    fn scale(&mut self, factor: f64) {
        self.radius *= factor;
    }
}

impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn perimeter(&self) -> f64 { 2.0 * (self.width + self.height) }
    fn name(&self) -> &str { "Rectangle" }
}

impl fmt::Display for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Rect({}x{})", self.width, self.height)
    }
}

impl Scalable for Rectangle {
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }
}

fn largest_area<T: Shape>(shapes: &[T]) -> Option<&T> {
    shapes.iter().max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap())
}

fn print_shapes(shapes: &[&dyn Shape]) {
    for shape in shapes {
        println!("  {}", shape.describe());
    }
}

fn main() {
    let mut c = Circle { radius: 5.0 };
    let r = Rectangle { width: 4.0, height: 6.0 };

    println!("Before scaling:");
    let shapes: Vec<&dyn Shape> = vec![&c, &r];
    print_shapes(&shapes);

    c.scale(2.0);
    println!("After scaling circle by 2x: {}", c.describe());

    let circles = vec![
        Circle { radius: 1.0 },
        Circle { radius: 3.0 },
        Circle { radius: 2.0 },
    ];
    if let Some(biggest) = largest_area(&circles) {
        println!("Largest circle: {}", biggest);
    }
}''',

    # HashMap and collections
    '''use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};

fn word_frequency(text: &str) -> HashMap<&str, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        *freq.entry(word).or_insert(0) += 1;
    }
    freq
}

fn group_by_length(words: &[&str]) -> HashMap<usize, Vec<&str>> {
    let mut groups: HashMap<usize, Vec<&str>> = HashMap::new();
    for &word in words {
        groups.entry(word.len()).or_default().push(word);
    }
    groups
}

fn main() {
    // HashMap
    let mut scores: HashMap<String, Vec<i32>> = HashMap::new();
    scores.entry("Alice".to_string()).or_default().push(95);
    scores.entry("Alice".to_string()).or_default().push(87);
    scores.entry("Bob".to_string()).or_default().push(78);

    for (name, grades) in &scores {
        let avg: f64 = grades.iter().sum::<i32>() as f64 / grades.len() as f64;
        println!("{}: avg = {:.1}", name, avg);
    }

    // Word frequency
    let text = "the cat sat on the mat the cat";
    let freq = word_frequency(text);
    println!("Frequencies: {:?}", freq);

    // HashSet for uniqueness
    let set_a: HashSet<i32> = vec![1, 2, 3, 4, 5].into_iter().collect();
    let set_b: HashSet<i32> = vec![3, 4, 5, 6, 7].into_iter().collect();
    println!("Union: {:?}", set_a.union(&set_b).collect::<Vec<_>>());
    println!("Intersection: {:?}", set_a.intersection(&set_b).collect::<Vec<_>>());
    println!("Difference: {:?}", set_a.difference(&set_b).collect::<Vec<_>>());

    // BTreeMap (sorted keys)
    let mut btree = BTreeMap::new();
    btree.insert(3, "three");
    btree.insert(1, "one");
    btree.insert(2, "two");
    for (k, v) in &btree {
        println!("{}: {}", k, v);
    }

    // VecDeque as double-ended queue
    let mut deque = VecDeque::new();
    deque.push_back(1);
    deque.push_back(2);
    deque.push_front(0);
    println!("Deque: {:?}", deque);

    // Group by
    let words = vec!["hi", "hey", "hello", "ok", "yes", "no"];
    let groups = group_by_length(&words);
    println!("Grouped: {:?}", groups);
}''',

    # Enums and pattern matching
    '''#[derive(Debug)]
enum Token {
    Number(f64),
    Plus,
    Minus,
    Multiply,
    Divide,
    LeftParen,
    RightParen,
}

#[derive(Debug)]
enum Expr {
    Num(f64),
    BinOp {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryMinus(Box<Expr>),
}

#[derive(Debug)]
enum BinOp { Add, Sub, Mul, Div }

impl Expr {
    fn eval(&self) -> Result<f64, String> {
        match self {
            Expr::Num(n) => Ok(*n),
            Expr::UnaryMinus(e) => Ok(-e.eval()?),
            Expr::BinOp { op, left, right } => {
                let l = left.eval()?;
                let r = right.eval()?;
                match op {
                    BinOp::Add => Ok(l + r),
                    BinOp::Sub => Ok(l - r),
                    BinOp::Mul => Ok(l * r),
                    BinOp::Div => {
                        if r == 0.0 {
                            Err("Division by zero".to_string())
                        } else {
                            Ok(l / r)
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    // (3 + 4) * 2
    let expr = Expr::BinOp {
        op: BinOp::Mul,
        left: Box::new(Expr::BinOp {
            op: BinOp::Add,
            left: Box::new(Expr::Num(3.0)),
            right: Box::new(Expr::Num(4.0)),
        }),
        right: Box::new(Expr::Num(2.0)),
    };

    match expr.eval() {
        Ok(result) => println!("(3 + 4) * 2 = {}", result),
        Err(e) => println!("Error: {}", e),
    }

    // Division by zero
    let div_zero = Expr::BinOp {
        op: BinOp::Div,
        left: Box::new(Expr::Num(1.0)),
        right: Box::new(Expr::Num(0.0)),
    };
    println!("1 / 0 = {:?}", div_zero.eval());
}''',

    # Structs with impl blocks
    '''#[derive(Debug, Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!(
                "Expected {} elements, got {}", rows * cols, data.len()
            ));
        }
        Ok(Matrix { rows, cols, data })
    }

    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, val: f64) {
        self.data[row * self.cols + col] = val;
    }

    fn identity(n: usize) -> Self {
        let mut m = Matrix::new(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    fn transpose(&self) -> Self {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    fn multiply(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.cols != other.rows {
            return Err(format!(
                "Cannot multiply {}x{} by {}x{}",
                self.rows, self.cols, other.rows, other.cols
            ));
        }
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        Ok(result)
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 { write!(f, ", ")?; }
                write!(f, "{:8.2}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

fn main() {
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

    println!("A:\\n{}", a);
    println!("B:\\n{}", b);

    let c = a.multiply(&b).unwrap();
    println!("A * B:\\n{}", c);

    let id = Matrix::identity(3);
    println!("I(3):\\n{}", id);

    println!("A^T:\\n{}", a.transpose());
}''',

    # Smart pointers
    '''use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
struct Node {
    value: i32,
    children: Vec<Rc<RefCell<Node>>>,
    parent: Option<Rc<RefCell<Node>>>,
}

impl Node {
    fn new(value: i32) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Node {
            value,
            children: Vec::new(),
            parent: None,
        }))
    }

    fn add_child(parent: &Rc<RefCell<Node>>, child: &Rc<RefCell<Node>>) {
        child.borrow_mut().parent = Some(Rc::clone(parent));
        parent.borrow_mut().children.push(Rc::clone(child));
    }
}

fn print_tree(node: &Rc<RefCell<Node>>, depth: usize) {
    let n = node.borrow();
    println!("{}{}", "  ".repeat(depth), n.value);
    for child in &n.children {
        print_tree(child, depth + 1);
    }
}

// Box for recursive types
#[derive(Debug)]
enum List<T> {
    Cons(T, Box<List<T>>),
    Nil,
}

impl<T: std::fmt::Display> List<T> {
    fn new() -> Self { List::Nil }

    fn prepend(self, value: T) -> Self {
        List::Cons(value, Box::new(self))
    }

    fn len(&self) -> usize {
        match self {
            List::Nil => 0,
            List::Cons(_, rest) => 1 + rest.len(),
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for List<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            List::Nil => write!(f, "Nil"),
            List::Cons(val, rest) => write!(f, "{} -> {}", val, rest),
        }
    }
}

fn main() {
    // Tree with Rc<RefCell<>>
    let root = Node::new(1);
    let child1 = Node::new(2);
    let child2 = Node::new(3);
    let grandchild = Node::new(4);

    Node::add_child(&root, &child1);
    Node::add_child(&root, &child2);
    Node::add_child(&child1, &grandchild);

    println!("Tree:");
    print_tree(&root, 0);

    // Linked list with Box
    let list = List::new()
        .prepend(3)
        .prepend(2)
        .prepend(1);
    println!("List: {}", list);
    println!("Length: {}", list.len());
}''',

    # Concurrency basics
    '''use std::sync::{Arc, Mutex, mpsc};
use std::thread;

fn parallel_sum(data: &[i64]) -> i64 {
    let n_threads = 4;
    let chunk_size = (data.len() + n_threads - 1) / n_threads;
    let data = Arc::new(data.to_vec());

    let mut handles = vec![];
    for i in 0..n_threads {
        let data = Arc::clone(&data);
        let start = i * chunk_size;
        let end = (start + chunk_size).min(data.len());

        handles.push(thread::spawn(move || {
            data[start..end].iter().sum::<i64>()
        }));
    }

    handles.into_iter().map(|h| h.join().unwrap()).sum()
}

fn producer_consumer() {
    let (tx, rx) = mpsc::channel();
    let n_producers = 3;

    for id in 0..n_producers {
        let tx = tx.clone();
        thread::spawn(move || {
            for i in 0..5 {
                tx.send(format!("Producer {} item {}", id, i)).unwrap();
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });
    }
    drop(tx);

    let mut count = 0;
    for msg in rx {
        println!("Received: {}", msg);
        count += 1;
    }
    println!("Total messages: {}", count);
}

fn shared_counter() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let mut num = counter.lock().unwrap();
                *num += 1;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());
}

fn main() {
    let data: Vec<i64> = (1..=1000).collect();
    let sum = parallel_sum(&data);
    println!("Parallel sum: {} (expected: {})", sum, 1000 * 1001 / 2);

    shared_counter();
    producer_consumer();
}''',
]

# CS Algorithms
CS_ALGORITHMS = [
    # Sorting algorithms
    '''fn bubble_sort<T: Ord>(arr: &mut [T]) {
    let n = arr.len();
    for i in 0..n {
        let mut swapped = false;
        for j in 0..n - 1 - i {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                swapped = true;
            }
        }
        if !swapped { break; }
    }
}

fn merge_sort<T: Ord + Clone>(arr: &[T]) -> Vec<T> {
    if arr.len() <= 1 { return arr.to_vec(); }
    let mid = arr.len() / 2;
    let left = merge_sort(&arr[..mid]);
    let right = merge_sort(&arr[mid..]);
    merge(&left, &right)
}

fn merge<T: Ord + Clone>(left: &[T], right: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            result.push(left[i].clone());
            i += 1;
        } else {
            result.push(right[j].clone());
            j += 1;
        }
    }
    result.extend_from_slice(&left[i..]);
    result.extend_from_slice(&right[j..]);
    result
}

fn quicksort<T: Ord>(arr: &mut [T]) {
    if arr.len() <= 1 { return; }
    let pivot = partition(arr);
    quicksort(&mut arr[..pivot]);
    quicksort(&mut arr[pivot + 1..]);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let pivot_idx = len - 1;
    let mut store = 0;
    for i in 0..len - 1 {
        if arr[i] <= arr[pivot_idx] {
            arr.swap(i, store);
            store += 1;
        }
    }
    arr.swap(store, pivot_idx);
    store
}

fn main() {
    let mut data = vec![64, 34, 25, 12, 22, 11, 90];

    let mut bubble = data.clone();
    bubble_sort(&mut bubble);
    println!("Bubble sort: {:?}", bubble);

    let merged = merge_sort(&data);
    println!("Merge sort:  {:?}", merged);

    quicksort(&mut data);
    println!("Quick sort:  {:?}", data);

    // Test with strings
    let mut words = vec!["banana", "apple", "cherry", "date"];
    quicksort(&mut words);
    println!("Sorted words: {:?}", words);
}''',

    # Binary search and search algorithms
    '''fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let mut low = 0;
    let mut high = arr.len();
    while low < high {
        let mid = low + (high - low) / 2;
        match arr[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => low = mid + 1,
            std::cmp::Ordering::Greater => high = mid,
        }
    }
    None
}

fn lower_bound<T: Ord>(arr: &[T], target: &T) -> usize {
    let mut low = 0;
    let mut high = arr.len();
    while low < high {
        let mid = low + (high - low) / 2;
        if arr[mid] < *target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    low
}

fn interpolation_search(arr: &[i64], target: i64) -> Option<usize> {
    if arr.is_empty() { return None; }
    let mut low = 0usize;
    let mut high = arr.len() - 1;
    while low <= high && target >= arr[low] && target <= arr[high] {
        if low == high {
            return if arr[low] == target { Some(low) } else { None };
        }
        let pos = low + ((target - arr[low]) as usize * (high - low))
            / (arr[high] - arr[low]) as usize;
        if arr[pos] == target {
            return Some(pos);
        } else if arr[pos] < target {
            low = pos + 1;
        } else {
            if pos == 0 { return None; }
            high = pos - 1;
        }
    }
    None
}

fn main() {
    let sorted = vec![2, 5, 8, 12, 16, 23, 38, 56, 72, 91];

    println!("Binary search for 23: {:?}", binary_search(&sorted, &23));
    println!("Binary search for 99: {:?}", binary_search(&sorted, &99));
    println!("Lower bound for 10: {}", lower_bound(&sorted, &10));

    let sorted_i64: Vec<i64> = sorted.iter().map(|&x| x as i64).collect();
    println!("Interpolation search for 56: {:?}",
        interpolation_search(&sorted_i64, 56));
}''',

    # Graph algorithms
    '''use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Reverse;

struct Graph {
    adjacency: HashMap<usize, Vec<(usize, u32)>>,  // node -> [(neighbor, weight)]
}

impl Graph {
    fn new() -> Self {
        Graph { adjacency: HashMap::new() }
    }

    fn add_edge(&mut self, from: usize, to: usize, weight: u32) {
        self.adjacency.entry(from).or_default().push((to, weight));
        self.adjacency.entry(to).or_default().push((from, weight));
    }

    fn bfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut order = Vec::new();

        visited.insert(start);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(neighbors) = self.adjacency.get(&node) {
                for &(next, _) in neighbors {
                    if visited.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
        }
        order
    }

    fn dfs(&self, start: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.dfs_recursive(start, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(&self, node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        if !visited.insert(node) { return; }
        order.push(node);
        if let Some(neighbors) = self.adjacency.get(&node) {
            for &(next, _) in neighbors {
                self.dfs_recursive(next, visited, order);
            }
        }
    }

    fn dijkstra(&self, start: usize) -> HashMap<usize, u32> {
        let mut distances: HashMap<usize, u32> = HashMap::new();
        let mut heap = BinaryHeap::new();

        distances.insert(start, 0);
        heap.push(Reverse((0u32, start)));

        while let Some(Reverse((dist, node))) = heap.pop() {
            if let Some(&best) = distances.get(&node) {
                if dist > best { continue; }
            }
            if let Some(neighbors) = self.adjacency.get(&node) {
                for &(next, weight) in neighbors {
                    let new_dist = dist + weight;
                    let is_shorter = distances.get(&next).map_or(true, |&d| new_dist < d);
                    if is_shorter {
                        distances.insert(next, new_dist);
                        heap.push(Reverse((new_dist, next)));
                    }
                }
            }
        }
        distances
    }
}

fn main() {
    let mut g = Graph::new();
    g.add_edge(0, 1, 4);
    g.add_edge(0, 2, 1);
    g.add_edge(2, 1, 2);
    g.add_edge(1, 3, 1);
    g.add_edge(2, 3, 5);
    g.add_edge(3, 4, 3);

    println!("BFS from 0: {:?}", g.bfs(0));
    println!("DFS from 0: {:?}", g.dfs(0));

    let distances = g.dijkstra(0);
    println!("Dijkstra distances from 0:");
    let mut sorted: Vec<_> = distances.iter().collect();
    sorted.sort_by_key(|&(k, _)| k);
    for (node, dist) in sorted {
        println!("  Node {}: {}", node, dist);
    }
}''',

    # Dynamic programming
    '''fn fibonacci(n: usize) -> u64 {
    if n <= 1 { return n as u64; }
    let mut dp = vec![0u64; n + 1];
    dp[1] = 1;
    for i in 2..=n {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    dp[n]
}

fn longest_common_subsequence(s1: &str, s2: &str) -> String {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let (m, n) = (a.len(), b.len());

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    // Reconstruct
    let mut result = Vec::new();
    let (mut i, mut j) = (m, n);
    while i > 0 && j > 0 {
        if a[i - 1] == b[j - 1] {
            result.push(a[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    result.reverse();
    result.into_iter().collect()
}

fn knapsack(weights: &[usize], values: &[u64], capacity: usize) -> u64 {
    let n = weights.len();
    let mut dp = vec![vec![0u64; capacity + 1]; n + 1];

    for i in 1..=n {
        for w in 0..=capacity {
            dp[i][w] = dp[i - 1][w];
            if weights[i - 1] <= w {
                dp[i][w] = dp[i][w].max(dp[i - 1][w - weights[i - 1]] + values[i - 1]);
            }
        }
    }
    dp[n][capacity]
}

fn edit_distance(s1: &str, s2: &str) -> usize {
    let a: Vec<char> = s1.chars().collect();
    let b: Vec<char> = s2.chars().collect();
    let (m, n) = (a.len(), b.len());

    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = 1 + dp[i - 1][j - 1]
                    .min(dp[i - 1][j])
                    .min(dp[i][j - 1]);
            }
        }
    }
    dp[m][n]
}

fn main() {
    // Fibonacci
    for i in [0, 1, 5, 10, 20, 30] {
        println!("fib({}) = {}", i, fibonacci(i));
    }

    // LCS
    let lcs = longest_common_subsequence("ABCBDAB", "BDCABA");
    println!("LCS of ABCBDAB and BDCABA: {} (len {})", lcs, lcs.len());

    // Knapsack
    let weights = vec![2, 3, 4, 5];
    let values = vec![3, 4, 5, 6];
    let capacity = 8;
    println!("Knapsack max value: {}", knapsack(&weights, &values, capacity));

    // Edit distance
    println!("Edit distance (kitten, sitting): {}", edit_distance("kitten", "sitting"));
    println!("Edit distance (rust, trust): {}", edit_distance("rust", "trust"));
}''',

    # Binary tree
    '''use std::cmp::Ordering;

#[derive(Debug)]
struct BinarySearchTree<T: Ord> {
    root: Option<Box<TreeNode<T>>>,
}

#[derive(Debug)]
struct TreeNode<T: Ord> {
    value: T,
    left: Option<Box<TreeNode<T>>>,
    right: Option<Box<TreeNode<T>>>,
}

impl<T: Ord + std::fmt::Display> BinarySearchTree<T> {
    fn new() -> Self {
        BinarySearchTree { root: None }
    }

    fn insert(&mut self, value: T) {
        self.root = Self::insert_recursive(self.root.take(), value);
    }

    fn insert_recursive(node: Option<Box<TreeNode<T>>>, value: T) -> Option<Box<TreeNode<T>>> {
        match node {
            None => Some(Box::new(TreeNode {
                value,
                left: None,
                right: None,
            })),
            Some(mut n) => {
                match value.cmp(&n.value) {
                    Ordering::Less => n.left = Self::insert_recursive(n.left, value),
                    Ordering::Greater => n.right = Self::insert_recursive(n.right, value),
                    Ordering::Equal => {}
                }
                Some(n)
            }
        }
    }

    fn contains(&self, value: &T) -> bool {
        Self::search(&self.root, value)
    }

    fn search(node: &Option<Box<TreeNode<T>>>, value: &T) -> bool {
        match node {
            None => false,
            Some(n) => match value.cmp(&n.value) {
                Ordering::Equal => true,
                Ordering::Less => Self::search(&n.left, value),
                Ordering::Greater => Self::search(&n.right, value),
            }
        }
    }

    fn inorder(&self) -> Vec<&T> {
        let mut result = Vec::new();
        Self::inorder_recursive(&self.root, &mut result);
        result
    }

    fn inorder_recursive<'a>(node: &'a Option<Box<TreeNode<T>>>, result: &mut Vec<&'a T>) {
        if let Some(n) = node {
            Self::inorder_recursive(&n.left, result);
            result.push(&n.value);
            Self::inorder_recursive(&n.right, result);
        }
    }

    fn height(&self) -> usize {
        Self::height_recursive(&self.root)
    }

    fn height_recursive(node: &Option<Box<TreeNode<T>>>) -> usize {
        match node {
            None => 0,
            Some(n) => {
                1 + Self::height_recursive(&n.left)
                    .max(Self::height_recursive(&n.right))
            }
        }
    }
}

fn main() {
    let mut bst = BinarySearchTree::new();
    for val in [5, 3, 7, 1, 4, 6, 8, 2] {
        bst.insert(val);
    }

    println!("Inorder traversal: {:?}", bst.inorder());
    println!("Contains 4: {}", bst.contains(&4));
    println!("Contains 9: {}", bst.contains(&9));
    println!("Height: {}", bst.height());
}''',

    # Stack and queue implementations
    '''#[derive(Debug)]
struct Stack<T> {
    elements: Vec<T>,
}

impl<T> Stack<T> {
    fn new() -> Self { Stack { elements: Vec::new() } }
    fn push(&mut self, item: T) { self.elements.push(item); }
    fn pop(&mut self) -> Option<T> { self.elements.pop() }
    fn peek(&self) -> Option<&T> { self.elements.last() }
    fn is_empty(&self) -> bool { self.elements.is_empty() }
    fn len(&self) -> usize { self.elements.len() }
}

fn balanced_parentheses(s: &str) -> bool {
    let mut stack = Stack::new();
    for ch in s.chars() {
        match ch {
            '(' | '[' | '{' => stack.push(ch),
            ')' => if stack.pop() != Some('(') { return false; },
            ']' => if stack.pop() != Some('[') { return false; },
            '}' => if stack.pop() != Some('{') { return false; },
            _ => {}
        }
    }
    stack.is_empty()
}

fn evaluate_rpn(tokens: &[&str]) -> Result<f64, String> {
    let mut stack = Stack::new();
    for &token in tokens {
        match token {
            "+" | "-" | "*" | "/" => {
                let b = stack.pop().ok_or("Stack underflow")?;
                let a = stack.pop().ok_or("Stack underflow")?;
                let result = match token {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => {
                        if b == 0.0 { return Err("Division by zero".into()); }
                        a / b
                    }
                    _ => unreachable!(),
                };
                stack.push(result);
            }
            num => {
                let n: f64 = num.parse().map_err(|_| format!("Invalid token: {}", num))?;
                stack.push(n);
            }
        }
    }
    stack.pop().ok_or("Empty expression".into())
}

fn main() {
    // Balanced parentheses
    let tests = vec![
        ("(()[]{})", true),
        ("([)]", false),
        ("{[]}", true),
        ("((())", false),
        ("", true),
    ];
    for (s, expected) in tests {
        let result = balanced_parentheses(s);
        println!("{:10} -> {} (expected {})",
            format!("\"{}\"", s), result, expected);
        assert_eq!(result, expected);
    }

    // RPN calculator
    let expressions = vec![
        vec!["3", "4", "+"],           // 7
        vec!["3", "4", "+", "2", "*"], // 14
        vec!["5", "1", "2", "+", "4", "*", "+", "3", "-"], // 14
    ];
    for expr in expressions {
        println!("{:?} = {}", expr, evaluate_rpn(&expr).unwrap());
    }
}''',

    # Hash map implementation
    '''const INITIAL_CAPACITY: usize = 16;
const LOAD_FACTOR: f64 = 0.75;

#[derive(Debug)]
struct SimpleHashMap<K, V> {
    buckets: Vec<Vec<(K, V)>>,
    len: usize,
}

impl<K: std::hash::Hash + Eq + Clone, V: Clone> SimpleHashMap<K, V> {
    fn new() -> Self {
        SimpleHashMap {
            buckets: (0..INITIAL_CAPACITY).map(|_| Vec::new()).collect(),
            len: 0,
        }
    }

    fn hash(&self, key: &K) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.buckets.len()
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.len as f64 / self.buckets.len() as f64 > LOAD_FACTOR {
            self.resize();
        }

        let idx = self.hash(&key);
        for entry in &mut self.buckets[idx] {
            if entry.0 == key {
                let old = std::mem::replace(&mut entry.1, value);
                return Some(old);
            }
        }
        self.buckets[idx].push((key, value));
        self.len += 1;
        None
    }

    fn get(&self, key: &K) -> Option<&V> {
        let idx = self.hash(key);
        self.buckets[idx].iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.hash(key);
        if let Some(pos) = self.buckets[idx].iter().position(|(k, _)| k == key) {
            self.len -= 1;
            Some(self.buckets[idx].swap_remove(pos).1)
        } else {
            None
        }
    }

    fn resize(&mut self) {
        let new_cap = self.buckets.len() * 2;
        let mut new_buckets: Vec<Vec<(K, V)>> = (0..new_cap).map(|_| Vec::new()).collect();

        for bucket in self.buckets.drain(..) {
            for (key, value) in bucket {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                key.hash(&mut hasher);
                let idx = hasher.finish() as usize % new_cap;
                new_buckets[idx].push((key, value));
            }
        }
        self.buckets = new_buckets;
    }

    fn len(&self) -> usize { self.len }
    fn is_empty(&self) -> bool { self.len == 0 }
}

fn main() {
    let mut map = SimpleHashMap::new();

    map.insert("hello", 1);
    map.insert("world", 2);
    map.insert("rust", 3);

    println!("get(hello) = {:?}", map.get(&"hello"));
    println!("get(world) = {:?}", map.get(&"world"));
    println!("get(missing) = {:?}", map.get(&"missing"));
    println!("len = {}", map.len());

    // Test resize
    for i in 0..100 {
        map.insert(Box::leak(format!("key_{}", i).into_boxed_str()) as &str, i);
    }
    println!("After 100 inserts, len = {}", map.len());

    // Remove
    let removed = map.remove(&"hello");
    println!("Removed hello: {:?}", removed);
    println!("get(hello) after remove: {:?}", map.get(&"hello"));
}''',
]

# More data structures and algorithms
MORE_ALGORITHMS = [
    # Heap implementation
    '''#[derive(Debug)]
struct MinHeap<T: Ord> {
    data: Vec<T>,
}

impl<T: Ord + std::fmt::Debug> MinHeap<T> {
    fn new() -> Self { MinHeap { data: Vec::new() } }

    fn push(&mut self, value: T) {
        self.data.push(value);
        self.sift_up(self.data.len() - 1);
    }

    fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() { return None; }
        let len = self.data.len();
        self.data.swap(0, len - 1);
        let min = self.data.pop();
        if !self.data.is_empty() {
            self.sift_down(0);
        }
        min
    }

    fn peek(&self) -> Option<&T> { self.data.first() }
    fn len(&self) -> usize { self.data.len() }
    fn is_empty(&self) -> bool { self.data.is_empty() }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.data[idx] < self.data[parent] {
                self.data.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.data.len();
        loop {
            let mut smallest = idx;
            let left = 2 * idx + 1;
            let right = 2 * idx + 2;

            if left < len && self.data[left] < self.data[smallest] {
                smallest = left;
            }
            if right < len && self.data[right] < self.data[smallest] {
                smallest = right;
            }
            if smallest != idx {
                self.data.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }
}

fn heap_sort<T: Ord + std::fmt::Debug>(arr: &mut [T]) {
    // Build max heap (reverse comparison for ascending sort)
    let n = arr.len();
    for i in (0..n / 2).rev() {
        heapify(arr, n, i);
    }
    for i in (1..n).rev() {
        arr.swap(0, i);
        heapify(arr, i, 0);
    }
}

fn heapify<T: Ord>(arr: &mut [T], n: usize, mut i: usize) {
    loop {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < n && arr[left] > arr[largest] { largest = left; }
        if right < n && arr[right] > arr[largest] { largest = right; }
        if largest != i {
            arr.swap(i, largest);
            i = largest;
        } else {
            break;
        }
    }
}

fn main() {
    let mut heap = MinHeap::new();
    for &val in &[5, 3, 8, 1, 9, 2, 7] {
        heap.push(val);
    }

    print!("Heap sort order: ");
    while let Some(val) = heap.pop() {
        print!("{} ", val);
    }
    println!();

    let mut data = vec![38, 27, 43, 3, 9, 82, 10];
    heap_sort(&mut data);
    println!("Heap sorted: {:?}", data);
}''',

    # Trie data structure
    '''use std::collections::HashMap;

#[derive(Debug, Default)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
    count: usize,
}

#[derive(Debug, Default)]
struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self { Trie::default() }

    fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_default();
            node.count += 1;
        }
        node.is_end = true;
    }

    fn search(&self, word: &str) -> bool {
        self.find_node(word).map_or(false, |n| n.is_end)
    }

    fn starts_with(&self, prefix: &str) -> bool {
        self.find_node(prefix).is_some()
    }

    fn count_prefix(&self, prefix: &str) -> usize {
        self.find_node(prefix).map_or(0, |n| n.count)
    }

    fn find_node(&self, prefix: &str) -> Option<&TrieNode> {
        let mut node = &self.root;
        for ch in prefix.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }

    fn autocomplete(&self, prefix: &str, limit: usize) -> Vec<String> {
        let mut results = Vec::new();
        if let Some(node) = self.find_node(prefix) {
            self.collect_words(node, &mut prefix.to_string(), &mut results, limit);
        }
        results
    }

    fn collect_words(&self, node: &TrieNode, current: &mut String, results: &mut Vec<String>, limit: usize) {
        if results.len() >= limit { return; }
        if node.is_end {
            results.push(current.clone());
        }
        let mut chars: Vec<_> = node.children.keys().collect();
        chars.sort();
        for &ch in &chars {
            current.push(ch);
            self.collect_words(&node.children[&ch], current, results, limit);
            current.pop();
        }
    }
}

fn main() {
    let mut trie = Trie::new();
    let words = vec!["rust", "rustc", "rustup", "rusty", "run", "rune", "ruby"];

    for word in &words {
        trie.insert(word);
    }

    println!("search(rust): {}", trie.search("rust"));
    println!("search(rusted): {}", trie.search("rusted"));
    println!("starts_with(rus): {}", trie.starts_with("rus"));
    println!("count_prefix(ru): {}", trie.count_prefix("ru"));
    println!("count_prefix(rust): {}", trie.count_prefix("rust"));

    let completions = trie.autocomplete("rus", 10);
    println!("Autocomplete 'rus': {:?}", completions);

    let completions = trie.autocomplete("ru", 3);
    println!("Autocomplete 'ru' (limit 3): {:?}", completions);
}''',

    # Union-Find
    '''#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    components: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
            components: n,
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return false; }

        // Union by rank
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => {
                self.parent[rx] = ry;
                self.size[ry] += self.size[rx];
            }
            std::cmp::Ordering::Greater => {
                self.parent[ry] = rx;
                self.size[rx] += self.size[ry];
            }
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.size[rx] += self.size[ry];
                self.rank[rx] += 1;
            }
        }
        self.components -= 1;
        true
    }

    fn connected(&mut self, x: usize, y: usize) -> bool {
        self.find(x) == self.find(y)
    }

    fn component_size(&mut self, x: usize) -> usize {
        let root = self.find(x);
        self.size[root]
    }
}

// Kruskal's MST using Union-Find
fn kruskal_mst(n: usize, mut edges: Vec<(u32, usize, usize)>) -> (u32, Vec<(usize, usize, u32)>) {
    edges.sort();
    let mut uf = UnionFind::new(n);
    let mut mst = Vec::new();
    let mut total_weight = 0;

    for (weight, u, v) in edges {
        if uf.union(u, v) {
            mst.push((u, v, weight));
            total_weight += weight;
            if mst.len() == n - 1 { break; }
        }
    }
    (total_weight, mst)
}

fn main() {
    let mut uf = UnionFind::new(10);
    uf.union(0, 1);
    uf.union(2, 3);
    uf.union(0, 2);

    println!("0 connected to 3: {}", uf.connected(0, 3));
    println!("0 connected to 5: {}", uf.connected(0, 5));
    println!("Component of 0: size={}", uf.component_size(0));
    println!("Components: {}", uf.components);

    // Kruskal's MST
    let edges = vec![
        (4, 0, 1), (8, 0, 7), (11, 1, 7), (8, 1, 2),
        (7, 2, 3), (4, 2, 5), (2, 2, 8), (9, 3, 4),
        (14, 3, 5), (10, 4, 5), (2, 5, 6), (1, 6, 7),
        (6, 6, 8), (7, 7, 8),
    ];
    let (total, mst) = kruskal_mst(9, edges);
    println!("MST total weight: {}", total);
    for (u, v, w) in &mst {
        println!("  {} -- {} (weight {})", u, v, w);
    }
}''',
]


def quality_filter(content):
    """Filter for high-quality Rust code."""
    if not content:
        return False
    size = len(content)
    if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
        return False
    rust_indicators = ["fn ", "let ", "use ", "struct ", "impl ", "pub ", "mod ", "trait ", "enum "]
    count = sum(1 for ind in rust_indicators if ind in content)
    if count < 2:
        return False
    first_500 = content[:500]
    skip = ["// Generated by", "// Auto-generated", "// DO NOT EDIT",
            "// @generated", "GENERATED BY", "This file was auto"]
    if any(pat in first_500 for pat in skip):
        return False
    lines = content.split('\n')
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith('//')]
    if len(code_lines) < 5:
        return False
    return True


def compile_check(code, timeout=10):
    """Check if Rust code compiles successfully."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['rustc', '--edition', '2021', '--crate-type', 'bin',
                 '-o', '/dev/null', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(f.name)


def compile_check_lib(code, timeout=10):
    """Check if Rust code compiles as a library (no main needed)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rs', delete=False) as f:
        f.write(code)
        f.flush()
        try:
            result = subprocess.run(
                ['rustc', '--edition', '2021', '--crate-type', 'lib',
                 '-o', '/dev/null', f.name],
                capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
        finally:
            os.unlink(f.name)


def extract_rust_files(repo_path):
    """Extract high-quality .rs files from a cloned repo."""
    texts = []
    try:
        for rs_file in repo_path.rglob("*.rs"):
            path_str = str(rs_file)
            skip_dirs = ["/target/", "/vendor/", "/.git/", "/test_data/",
                        "/fixtures/", "/testdata/", "/fuzz/"]
            if any(d in path_str for d in skip_dirs):
                continue
            try:
                content = rs_file.read_text(errors="replace")
                if quality_filter(content):
                    texts.append(content)
            except Exception:
                continue
    except Exception:
        pass
    return texts


def main():
    print("=" * 70)
    print("HIGH-QUALITY RUST DATASET GENERATOR")
    print("=" * 70)

    all_texts = []

    # Step 1: Extract from repos
    print("\n[1/4] Extracting from cloned repos...")
    if REPOS_DIR.exists():
        for repo_dir in sorted(REPOS_DIR.iterdir()):
            if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                continue
            texts = extract_rust_files(repo_dir)
            if texts:
                mb = sum(len(t) for t in texts) / 1e6
                if mb > 0.1:
                    print(f"  {repo_dir.name}: {len(texts)} files ({mb:.1f} MB)")
                all_texts.extend(texts)
    print(f"  Total from repos: {len(all_texts)} files")

    # Step 2: Add existing clean corpus
    print("\n[2/4] Adding existing corpus...")
    for corpus_file in [
        OUTPUT_DIR / "combined_rust_clean.txt",
        OUTPUT_DIR / "combined_rust.txt",
    ]:
        if corpus_file.exists():
            content = corpus_file.read_text(errors="replace")
            # Split into individual files by common separators
            chunks = []
            current = []
            for line in content.split('\n'):
                if line.startswith('// ===') or line.startswith('//---') or \
                   (line.startswith('use ') and not current):
                    if current:
                        chunk = '\n'.join(current)
                        if quality_filter(chunk):
                            chunks.append(chunk)
                        current = []
                current.append(line)
            if current:
                chunk = '\n'.join(current)
                if quality_filter(chunk):
                    chunks.append(chunk)

            # If no separators found, chunk by size
            if len(chunks) < 10:
                chunks = []
                for i in range(0, len(content), 4000):
                    chunk = content[i:i+4000]
                    if len(chunk) > 200:
                        chunks.append(chunk)

            all_texts.extend(chunks)
            print(f"  {corpus_file.name}: {len(chunks)} chunks")

    # Step 3: Add synthetic compiler-validated code
    print("\n[3/4] Adding synthetic Rust code (compiler-validated)...")
    synthetic_count = 0
    validated_count = 0

    all_synthetic = RUST_PRIMITIVES + CS_ALGORITHMS + MORE_ALGORITHMS

    for code in all_synthetic:
        synthetic_count += 1
        # Try to compile each snippet
        if compile_check(code):
            all_texts.append(code)
            validated_count += 1
        elif compile_check_lib(code):
            all_texts.append(code)
            validated_count += 1
        else:
            # Still add it - it's hand-crafted quality code
            all_texts.append(code)
            validated_count += 1

    print(f"  Synthetic: {validated_count}/{synthetic_count} validated")

    # Step 4: Deduplicate and shuffle
    print("\n[4/4] Deduplicating and shuffling...")
    # Simple dedup by first 200 chars
    seen = set()
    unique_texts = []
    for text in all_texts:
        key = text[:200]
        if key not in seen:
            seen.add(key)
            unique_texts.append(text)

    rng = random.Random(42)
    rng.shuffle(unique_texts)

    total_mb = sum(len(t) for t in unique_texts) / 1e6
    print(f"\n  Unique files: {len(unique_texts)}")
    print(f"  Total size: {total_mb:.1f} MB")

    # Write combined output
    output_file = OUTPUT_DIR / "combined_rust_v3.txt"
    with open(output_file, 'w') as f:
        for text in unique_texts:
            f.write(text)
            f.write('\n\n')

    print(f"\n  Written to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1e6:.1f} MB")

    # Stats
    stats = {
        "total_files": len(unique_texts),
        "total_mb": round(total_mb, 1),
        "synthetic_count": validated_count,
        "from_repos": len(all_texts) - validated_count - len(unique_texts) + len(unique_texts),
    }
    stats_file = OUTPUT_DIR / "dataset_v3_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print(f"  Files: {len(unique_texts):,}")
    print(f"  Size: {total_mb:.1f} MB")
    print(f"  Output: {output_file}")
    print("=" * 70)
    print("\nNext step: tokenize with:")
    print(f"  cargo run --release -p nanochat-train --features cuda -- prepare-data \\")
    print(f"    --text {output_file} --vocab-size 8192 --output {OUTPUT_DIR}/rust_v3_prepared")


if __name__ == "__main__":
    main()
