use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dynamo_runtime::compute::ComputePool;
use std::sync::Arc;
use tokio::runtime::Runtime as TokioRuntime;

/// Compute-intensive function: sum of all primes up to n
fn compute_primes_sum(n: u64) -> u64 {
    let mut sum = 0u64;
    for candidate in 2..=n {
        if is_prime(candidate) {
            sum += candidate;
        }
    }
    sum
}

fn is_prime(n: u64) -> bool {
    if n <= 1 {
        return false;
    }
    if n <= 3 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }

    let sqrt_n = (n as f64).sqrt() as u64;
    for i in (5..=sqrt_n).step_by(6) {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
    }
    true
}

fn bench_compute_overhead(c: &mut Criterion) {
    // Test various sizes to show overhead vs compute tradeoff
    let test_sizes = [10, 100, 1_000, 10_000, 100_000, 1_000_000];

    let mut group = c.benchmark_group("compute_overhead");
    group.sample_size(20); // Reduce sample size for longer benchmarks

    // Setup runtimes
    let tokio_4thread = TokioRuntime::new().unwrap();
    let tokio_1thread = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Setup compute pool
    let compute_config = dynamo_runtime::compute::ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "bench".to_string(),
        pin_threads: false,
    };
    let compute_pool = Arc::new(ComputePool::new(compute_config).unwrap());

    for n in test_sizes {
        // Benchmark 1: Direct execution on Tokio (4 threads)
        group.bench_with_input(BenchmarkId::new("tokio_direct", n), &n, |b, &n| {
            b.to_async(&tokio_4thread).iter(|| async move {
                black_box(compute_primes_sum(black_box(n)))
            });
        });

        // Benchmark 2: Rayon offload (1 Tokio thread + 4 Rayon threads)
        let pool = compute_pool.clone();
        group.bench_with_input(BenchmarkId::new("rayon_offload", n), &n, |b, &n| {
            b.to_async(&tokio_1thread).iter(|| {
                let pool = pool.clone();
                async move {
                    pool.execute(move || black_box(compute_primes_sum(black_box(n))))
                        .await
                        .unwrap()
                }
            });
        });

        // Benchmark 3: spawn_blocking (4 Tokio threads)
        group.bench_with_input(BenchmarkId::new("spawn_blocking", n), &n, |b, &n| {
            b.to_async(&tokio_4thread).iter(|| async move {
                tokio::task::spawn_blocking(move || black_box(compute_primes_sum(black_box(n))))
                    .await
                    .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_parallel_tasks(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_tasks");
    group.sample_size(10); // Even smaller sample for parallel benchmarks

    let tokio_runtime = TokioRuntime::new().unwrap();
    let compute_config = dynamo_runtime::compute::ComputeConfig {
        num_threads: Some(4),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "bench".to_string(),
        pin_threads: false,
    };
    let compute_pool = Arc::new(ComputePool::new(compute_config).unwrap());

    // Test with different batch sizes
    for batch_size in [10, 100, 1000] {
        let n = 10_000; // Fixed compute size

        // Parallel execution with Rayon
        let pool = compute_pool.clone();
        group.bench_with_input(
            BenchmarkId::new("rayon_parallel", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&tokio_runtime).iter(|| {
                    let pool = pool.clone();
                    async move {
                        let tasks = (0..batch_size)
                            .map(|_| {
                                let pool = pool.clone();
                                tokio::spawn(async move {
                                    pool.execute(move || compute_primes_sum(n))
                                        .await
                                        .unwrap()
                                })
                            })
                            .collect::<Vec<_>>();

                        for task in tasks {
                            black_box(task.await.unwrap());
                        }
                    }
                });
            },
        );

        // Parallel execution with spawn_blocking
        group.bench_with_input(
            BenchmarkId::new("spawn_blocking_parallel", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&tokio_runtime).iter(|| async move {
                    let tasks = (0..batch_size)
                        .map(|_| {
                            tokio::spawn(async move {
                                tokio::task::spawn_blocking(move || compute_primes_sum(n))
                                    .await
                                    .unwrap()
                            })
                        })
                        .collect::<Vec<_>>();

                    for task in tasks {
                        black_box(task.await.unwrap());
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compute_overhead, bench_parallel_tasks);
criterion_main!(benches);