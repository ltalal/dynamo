// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Example demonstrating the use of ComputePool for CPU-intensive operations
//!
//! This example shows various patterns for using Rayon with Tokio:
//! - Fork-join with scope
//! - Parallel batch processing
//! - Dynamic task spawning
//! - Integration with async services

use anyhow::Result;
use dynamo_runtime::{Worker, Runtime, compute::{ComputePool, ComputePoolExt}};
use std::time::Instant;

/// Simulate expensive CPU-bound computation
fn expensive_computation(input: u64) -> u64 {
    // Simulate work with a simple prime check
    let mut sum = 0u64;
    for i in 2..input {
        if is_prime(i) {
            sum += i;
        }
    }
    sum
}

fn is_prime(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    for i in 2..((n as f64).sqrt() as u64 + 1) {
        if n % i == 0 {
            return false;
        }
    }
    true
}

/// Example 1: Simple fork-join pattern
async fn example_fork_join(pool: &ComputePool) -> Result<()> {
    println!("\n=== Example 1: Fork-Join Pattern ===");

    let start = Instant::now();

    // Run two expensive computations in parallel
    let (result1, result2) = pool.join(
        || expensive_computation(10000),
        || expensive_computation(20000),
    ).await?;

    println!("Fork-join results: {} and {}", result1, result2);
    println!("Time: {:?}", start.elapsed());

    Ok(())
}

/// Example 2: Scope-based parallel execution
async fn example_scope(pool: &ComputePool) -> Result<()> {
    println!("\n=== Example 2: Scope-based Execution ===");

    let data = vec![1000, 2000, 3000, 4000, 5000];
    let start = Instant::now();

    let results = pool.execute_scoped(|scope| {
        let mut results = vec![0u64; data.len()];
        let results_ptr = results.as_mut_ptr();

        for (i, &value) in data.iter().enumerate() {
            scope.spawn(move |_| {
                let result = expensive_computation(value);
                unsafe {
                    results_ptr.add(i).write(result);
                }
            });
        }

        results
    }).await?;

    println!("Scope results: {:?}", results);
    println!("Time: {:?}", start.elapsed());

    Ok(())
}

/// Example 3: Parallel map using extension trait
async fn example_parallel_map(pool: &ComputePool) -> Result<()> {
    println!("\n=== Example 3: Parallel Map ===");

    let items: Vec<u64> = (1..=10).map(|i| i * 1000).collect();
    let start = Instant::now();

    let results = pool.parallel_map(items.clone(), expensive_computation).await?;

    println!("Parallel map processed {} items", results.len());
    println!("Time: {:?}", start.elapsed());

    // Compare with sequential processing
    let start_seq = Instant::now();
    let _sequential: Vec<_> = items.iter().map(|&i| expensive_computation(i)).collect();
    println!("Sequential time: {:?}", start_seq.elapsed());

    Ok(())
}

/// Example 4: Simulating tokenization workload
async fn example_tokenization(pool: &ComputePool) -> Result<()> {
    println!("\n=== Example 4: Batch Tokenization Simulation ===");

    // Simulate batch of texts to tokenize
    let texts: Vec<String> = (0..100)
        .map(|i| format!("This is sample text number {} that needs to be tokenized", i))
        .collect();

    let start = Instant::now();

    // Process in parallel using scope
    let token_counts = pool.execute_scoped(|scope| {
        let mut counts = vec![0usize; texts.len()];
        let counts_ptr = counts.as_mut_ptr();

        for (i, text) in texts.iter().enumerate() {
            let text = text.clone();
            scope.spawn(move |_| {
                // Simulate tokenization by counting words
                let count = text.split_whitespace().count();
                // Simulate more work
                std::thread::sleep(std::time::Duration::from_micros(100));
                unsafe {
                    counts_ptr.add(i).write(count);
                }
            });
        }

        counts
    }).await?;

    let total_tokens: usize = token_counts.iter().sum();
    println!("Tokenized {} texts, total tokens: {}", texts.len(), total_tokens);
    println!("Time: {:?}", start.elapsed());

    Ok(())
}

/// Example 5: Hierarchical computation
async fn example_hierarchical(pool: &ComputePool) -> Result<()> {
    println!("\n=== Example 5: Hierarchical Computation ===");

    let start = Instant::now();

    let result = pool.execute_scoped(|scope| {
        let mut phase1_results = vec![0u64; 4];
        let phase1_ptr = phase1_results.as_mut_ptr();

        // First level: compute initial values
        for i in 0..4 {
            scope.spawn(move |s2| {
                let intermediate = expensive_computation((i + 1) as u64 * 1000);

                // Second level: further process each result
                let mut phase2_results = vec![0u64; 2];
                let phase2_ptr = phase2_results.as_mut_ptr();

                for j in 0..2 {
                    let value = intermediate + (j as u64 * 100);
                    s2.spawn(move |_| {
                        let result = expensive_computation(value);
                        unsafe {
                            phase2_ptr.add(j).write(result);
                        }
                    });
                }

                let sum: u64 = phase2_results.iter().sum();
                unsafe {
                    phase1_ptr.add(i).write(sum);
                }
            });
        }

        phase1_results.iter().sum::<u64>()
    }).await?;

    println!("Hierarchical computation result: {}", result);
    println!("Time: {:?}", start.elapsed());

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // Create worker and runtime
    let worker = Worker::from_settings()?;
    let runtime = worker.runtime().clone();

    // Get compute pool
    let pool = runtime.compute_pool()
        .ok_or_else(|| anyhow::anyhow!("Compute pool not initialized"))?
        .clone();

    println!("Compute pool initialized with {} threads", pool.num_threads());

    // Run examples
    example_fork_join(&pool).await?;
    example_scope(&pool).await?;
    example_parallel_map(&pool).await?;
    example_tokenization(&pool).await?;
    example_hierarchical(&pool).await?;

    // Print metrics
    let metrics = pool.metrics();
    println!("\n=== Compute Pool Metrics ===");
    println!("{}", metrics);

    Ok(())
}