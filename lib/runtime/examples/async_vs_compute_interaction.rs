//! Benchmark demonstrating async I/O vs compute workload interaction
//!
//! This example measures how different types of compute workloads interfere with
//! async I/O latency by comparing actual elapsed time vs expected sleep time.
//!
//! Key measurements:
//! - Baseline async overhead with no compute load
//! - Interference from small (<100μs), medium (~500μs), and large (~2-5ms) compute tasks
//! - Comparison between all-async (4 Tokio threads) vs hybrid (2 Tokio + 2 Rayon)
//! - Impact of offloading compute work to dedicated Rayon threads
//!
//! The benchmark spawns many lightweight async tasks doing timed sleeps, then runs
//! a fixed compute workload while measuring how much the compute work delays the
//! async tasks from being revisited after their sleeps complete.
//!
//! Two configurations are tested with EXACTLY 4 total threads:
//! 1. All-Async: 4 Tokio threads (compute runs inline, blocking async work)
//! 2. Hybrid: 2 Tokio threads + 2 Rayon threads (compute offloaded, async stays responsive)

use anyhow::Result;
use dynamo_runtime::{
    Runtime,
    compute::{ComputeConfig, ComputePool},
    compute_small,
};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;

/// Sleep latency measurement
#[derive(Debug, Clone)]
struct SleepMeasurement {
    _expected_ms: f64,
    _actual_ms: f64,
    overhead_ms: f64, // actual - expected
}

/// Statistics for latency measurements
#[derive(Debug, Clone)]
struct LatencyStats {
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
    count: usize,
}

/// Test results for a single configuration
#[derive(Debug)]
struct TestResults {
    baseline_overhead: LatencyStats,
    compute_overhead: Option<LatencyStats>,
    compute_duration: Option<Duration>,
    _total_sleep_measurements: usize,
}

/// Type of workload to run
#[derive(Debug, Clone, Copy, PartialEq)]
enum WorkloadType {
    None,   // No compute (baseline)
    Small,  // 100% small tasks
    Medium, // 100% medium tasks
    Large,  // 100% large tasks
    Mixed,  // 33/33/33 mix
}

/// Individual task type
#[derive(Debug, Clone, Copy)]
enum TaskType {
    Small,
    Medium,
    Large,
}

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

/// Small compute task (~10μs)
fn small_compute() -> u64 {
    compute_primes_sum(10)
}

/// Medium compute task (~500μs)
fn medium_compute() -> u64 {
    compute_primes_sum(1_000)
}

/// Large compute task (~2-5ms)
fn large_compute() -> u64 {
    compute_primes_sum(100_000)
}

/// Worker that repeatedly sleeps and measures latency
async fn sleep_worker(
    sleep_duration: Duration,
    results: Arc<Mutex<Vec<SleepMeasurement>>>,
    cancel: CancellationToken,
) {
    while !cancel.is_cancelled() {
        let start = Instant::now();
        tokio::select! {
            _ = sleep(sleep_duration) => {
                let elapsed = start.elapsed();
                let measurement = SleepMeasurement {
                    _expected_ms: sleep_duration.as_secs_f64() * 1000.0,
                    _actual_ms: elapsed.as_secs_f64() * 1000.0,
                    overhead_ms: (elapsed.as_secs_f64() - sleep_duration.as_secs_f64()) * 1000.0,
                };
                results.lock().unwrap().push(measurement);
            }
            _ = cancel.cancelled() => break,
        }
    }
}

/// Execute a single compute task based on type
async fn execute_compute_task(task_type: TaskType, pool: Option<Arc<ComputePool>>) -> Result<u64> {
    match task_type {
        TaskType::Small => {
            // Small tasks always run inline
            Ok(compute_small!(small_compute()))
        }
        TaskType::Medium => {
            // Medium tasks: offload if pool available, else run inline (blocking)
            if let Some(pool) = pool.clone() {
                pool.execute(medium_compute).await
            } else {
                // No pool - run inline on Tokio thread (will block!)
                Ok(medium_compute())
            }
        }
        TaskType::Large => {
            // Large tasks: offload if pool available, else run inline (severely blocking)
            if let Some(pool) = pool {
                pool.execute(large_compute).await
            } else {
                // No pool - run inline on Tokio thread (will severely block!)
                Ok(large_compute())
            }
        }
    }
}

/// Execute a batch of compute tasks with concurrency limiting
async fn execute_compute_batch(
    workload_type: WorkloadType,
    num_tasks: usize,
    concurrency_limit: Arc<Semaphore>,
    pool: Option<Arc<ComputePool>>,
) -> Duration {
    if workload_type == WorkloadType::None {
        return Duration::from_secs(0);
    }

    let start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let permit = concurrency_limit.clone().acquire_owned().await.unwrap();
        let pool = pool.clone();

        let task_type = match workload_type {
            WorkloadType::Small => TaskType::Small,
            WorkloadType::Medium => TaskType::Medium,
            WorkloadType::Large => TaskType::Large,
            WorkloadType::Mixed => {
                // Round-robin: 33% small, 33% medium, 33% large
                match i % 3 {
                    0 => TaskType::Small,
                    1 => TaskType::Medium,
                    _ => TaskType::Large,
                }
            }
            WorkloadType::None => unreachable!(),
        };

        let handle = tokio::spawn(async move {
            let _permit = permit; // Hold permit until task completes
            execute_compute_task(task_type, pool).await
        });
        handles.push(handle);
    }

    // Wait for all compute tasks
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    start.elapsed()
}

/// Calculate statistics from measurements
fn calculate_stats(measurements: &[SleepMeasurement]) -> LatencyStats {
    if measurements.is_empty() {
        return LatencyStats {
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            max: 0.0,
            mean: 0.0,
            count: 0,
        };
    }

    let mut overheads: Vec<f64> = measurements.iter().map(|m| m.overhead_ms).collect();
    overheads.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = overheads.len();
    LatencyStats {
        p50: overheads[len / 2],
        p95: overheads[len * 95 / 100],
        p99: overheads[len * 99 / 100],
        max: *overheads.last().unwrap(),
        mean: overheads.iter().sum::<f64>() / len as f64,
        count: len,
    }
}

/// Run a single interference test
async fn run_interference_test(
    _runtime: Arc<Runtime>,
    workload_type: WorkloadType,
    pool: Option<Arc<ComputePool>>,
) -> TestResults {
    // Configuration
    const NUM_SLEEP_TASKS: usize = 100;
    const SLEEP_DURATION_MS: u64 = 1;
    const NUM_COMPUTE_TASKS: usize = 1000;
    const CONCURRENCY_LIMIT: usize = 4; // Match total thread count
    const BASELINE_DURATION_SECS: u64 = 1;

    // 1. Start async load (100 tasks doing 1ms sleeps)
    let results = Arc::new(Mutex::new(Vec::new()));
    let cancel = CancellationToken::new();
    let mut handles = Vec::new();

    for _ in 0..NUM_SLEEP_TASKS {
        let r = results.clone();
        let c = cancel.clone();
        handles.push(tokio::spawn(sleep_worker(
            Duration::from_millis(SLEEP_DURATION_MS),
            r,
            c,
        )));
    }

    // 2. Collect baseline measurements
    sleep(Duration::from_secs(BASELINE_DURATION_SECS)).await;
    let baseline_count = results.lock().unwrap().len();

    // 3. Run compute workload (if not baseline)
    let compute_duration = if workload_type != WorkloadType::None {
        let semaphore = Arc::new(Semaphore::new(CONCURRENCY_LIMIT));
        Some(execute_compute_batch(workload_type, NUM_COMPUTE_TASKS, semaphore, pool).await)
    } else {
        // For baseline, just wait another second
        sleep(Duration::from_secs(1)).await;
        None
    };

    // 4. Stop async load
    cancel.cancel();
    for handle in handles {
        handle.await.unwrap();
    }

    // 5. Analyze results
    let all_measurements = results.lock().unwrap().clone();
    let baseline = &all_measurements[..baseline_count.min(all_measurements.len())];
    let during_compute = if baseline_count < all_measurements.len() {
        Some(&all_measurements[baseline_count..])
    } else {
        None
    };

    TestResults {
        baseline_overhead: calculate_stats(baseline),
        compute_overhead: during_compute.map(calculate_stats),
        compute_duration,
        _total_sleep_measurements: all_measurements.len(),
    }
}

/// Run all test workloads for a given configuration
async fn run_all_tests(pool: Option<Arc<ComputePool>>) -> Result<()> {
    let workload_types = vec![
        ("Baseline (no compute)", WorkloadType::None),
        ("100% Small (~10μs each)", WorkloadType::Small),
        ("100% Medium (~500μs each)", WorkloadType::Medium),
        ("100% Large (~2-5ms each)", WorkloadType::Large),
        ("Mixed 33/33/33", WorkloadType::Mixed),
    ];

    // Create dummy runtime for the test functions
    let runtime = Arc::new(Runtime::from_current()?);

    for (name, workload) in &workload_types {
        println!("\n📊 Workload: {}", name);
        println!("{:-<50}", "");

        let results = run_interference_test(runtime.clone(), *workload, pool.clone()).await;

        // Always show baseline overhead
        println!("  Baseline async overhead (first {}s):", 1);
        println!(
            "    Mean: {:.3}ms, P50: {:.3}ms, P95: {:.3}ms, P99: {:.3}ms",
            results.baseline_overhead.mean,
            results.baseline_overhead.p50,
            results.baseline_overhead.p95,
            results.baseline_overhead.p99
        );
        println!("    Measurements: {}", results.baseline_overhead.count);

        // Show compute interference if applicable
        if let Some(compute_overhead) = results.compute_overhead {
            println!("\n  During compute workload:");
            println!(
                "    Mean: {:.3}ms, P50: {:.3}ms, P95: {:.3}ms, P99: {:.3}ms",
                compute_overhead.mean,
                compute_overhead.p50,
                compute_overhead.p95,
                compute_overhead.p99
            );
            println!(
                "    Max: {:.3}ms, Measurements: {}",
                compute_overhead.max, compute_overhead.count
            );

            // Calculate interference factor
            if results.baseline_overhead.mean > 0.0 {
                let interference_factor = compute_overhead.mean / results.baseline_overhead.mean;
                println!(
                    "\n  ⚠️  Interference factor: {:.1}x slower",
                    interference_factor
                );

                // Provide interpretation
                let impact = if interference_factor < 2.0 {
                    "Minimal - async remains responsive"
                } else if interference_factor < 10.0 {
                    "Moderate - noticeable async delays"
                } else {
                    "SEVERE - async tasks are heavily blocked!"
                };
                println!("     Impact: {}", impact);
            }
        }

        // Show compute duration
        if let Some(duration) = results.compute_duration {
            println!(
                "\n  Compute workload completed in: {:.2}s",
                duration.as_secs_f64()
            );
            println!(
                "  Throughput: {:.0} tasks/sec",
                1000.0 / duration.as_secs_f64()
            );
        }
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("🔬 Async vs Compute Interaction Benchmark");
    println!("==========================================");
    println!();
    println!("This benchmark measures how compute workloads interfere with async I/O latency.");
    println!("We test with EXACTLY 4 total threads in two configurations:");
    println!("  1. All-Async: 4 Tokio threads (compute blocks async work)");
    println!("  2. Hybrid: 2 Tokio + 2 Rayon threads (compute offloaded)");
    println!();
    println!("Lower overhead numbers mean better async responsiveness.\n");

    // Test 1: All Async (4 Tokio threads, no Rayon)
    println!("\n{:=<70}", "");
    println!("Configuration 1: All-Async (4 Tokio threads, no Rayon)");
    println!("{:=<70}", "");
    println!("⚠️  Compute tasks run INLINE on Tokio threads, blocking async work!");

    let all_async_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .thread_name("tokio-worker")
        .enable_all()
        .build()?;

    all_async_rt.block_on(async {
        // No compute pool for all-async mode
        // All compute work will run inline on Tokio threads
        run_all_tests(None).await
    })?;

    // Test 2: Hybrid (2 Tokio + 2 Rayon)
    println!("\n{:=<70}", "");
    println!("Configuration 2: Hybrid (2 Tokio + 2 Rayon threads)");
    println!("{:=<70}", "");
    println!("✅ Compute tasks offloaded to Rayon, keeping async threads free!");

    let hybrid_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("tokio-worker")
        .enable_all()
        .build()?;

    // Create Rayon pool with 2 threads
    let compute_pool = Arc::new(ComputePool::new(ComputeConfig {
        num_threads: Some(2),
        stack_size: Some(2 * 1024 * 1024),
        thread_prefix: "rayon".to_string(),
        pin_threads: false,
    })?);

    hybrid_rt.block_on(async { run_all_tests(Some(compute_pool)).await })?;

    // Summary
    println!("\n{:=<70}", "");
    println!("📈 Analysis & Recommendations");
    println!("{:=<70}", "");
    println!();
    println!("Expected Results:");
    println!("• All-Async mode:");
    println!("  - Small tasks: Minimal interference (tasks complete quickly)");
    println!("  - Medium/Large tasks: SEVERE interference (all threads blocked)");
    println!();
    println!("• Hybrid mode:");
    println!("  - All task sizes: Minimal interference (compute offloaded)");
    println!("  - Async threads remain free to handle I/O");
    println!();
    println!("Key Takeaway:");
    println!("When compute tasks take >100μs, offloading to dedicated threads");
    println!("is essential to maintain async responsiveness!");
    println!();
    println!("✅ Benchmark complete!");

    Ok(())
}
