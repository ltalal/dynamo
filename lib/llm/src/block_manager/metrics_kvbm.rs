// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::Router;
use dynamo_runtime::metrics::prometheus_names::{
    kvbm::{
        CONNECTOR_MAYBE_FINISHED_OFFLOADING, CONNECTOR_MAYBE_FINISHED_ONBOARDING,
        CONNECTOR_OFFLOADING_OPERATIONS, MATCHED_TOKENS, OFFLOAD_BLOCKS_D2D,
        OFFLOAD_BLOCKS_D2D_COMPLETED, OFFLOAD_BLOCKS_D2H, OFFLOAD_BLOCKS_D2H_COMPLETED,
        OFFLOAD_BLOCKS_H2D, OFFLOAD_BLOCKS_H2D_COMPLETED, OFFLOAD_QUEUE_D2D, OFFLOAD_QUEUE_D2H,
        OFFLOAD_QUEUE_H2D, OFFLOAD_TRANSFERS_D2D, OFFLOAD_TRANSFERS_D2D_COMPLETED,
        OFFLOAD_TRANSFERS_D2H, OFFLOAD_TRANSFERS_D2H_COMPLETED, OFFLOAD_TRANSFERS_H2D,
        OFFLOAD_TRANSFERS_H2D_COMPLETED, ONBOARD_BLOCKS_D2D, ONBOARD_BLOCKS_D2D_COMPLETED,
        ONBOARD_BLOCKS_H2D, ONBOARD_BLOCKS_H2D_COMPLETED, ONBOARD_TRANSFERS_D2D,
        ONBOARD_TRANSFERS_D2D_COMPLETED, ONBOARD_TRANSFERS_H2D, ONBOARD_TRANSFERS_H2D_COMPLETED,
    },
    sanitize_prometheus_name,
};
use prometheus::{HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts, Registry};
use std::{collections::HashMap, net::SocketAddr, sync::Arc, thread};
use tokio::{net::TcpListener, sync::Notify};

use crate::http::service::{RouteDoc, metrics::router};
use crate::block_manager::offload::MAX_TRANSFER_BATCH_SIZE;

#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    // number of blocks offloaded from device to host
    pub offload_blocks_d2h: IntCounter,

    // number of blocks offloaded from host to disk
    pub offload_blocks_h2d: IntCounter,

    // number of blocks offloaded from device to disk (bypassing host memory)
    pub offload_blocks_d2d: IntCounter,

    // number of blocks onboarded from host to device
    pub onboard_blocks_h2d: IntCounter,

    // number of blocks onboarded from disk to device
    pub onboard_blocks_d2d: IntCounter,

    // number of completed offload blocks from device to host
    pub offload_blocks_d2h_completed: IntCounter,

    // number of completed offload blocks from host to disk
    pub offload_blocks_h2d_completed: IntCounter,

    // number of completed offload blocks from device to disk (bypassing host memory)
    pub offload_blocks_d2d_completed: IntCounter,

    // number of completed onboard blocks from host to device
    pub onboard_blocks_h2d_completed: IntCounter,

    // number of completed onboard blocks from disk to device
    pub onboard_blocks_d2d_completed: IntCounter,

    // number of offload transfers from device to host
    pub offload_transfers_d2h: IntCounter,

    // number of offload transfers from host to disk
    pub offload_transfers_h2d: IntCounter,

    // number of offload transfers from device to disk (bypassing host memory)
    pub offload_transfers_d2d: IntCounter,

    // number of onboard transfers from host to device
    pub onboard_transfers_h2d: IntCounter,

    // number of onboard transfers from disk to device
    pub onboard_transfers_d2d: IntCounter,

    // number of completed offload transfers from device to host
    pub offload_transfers_d2h_completed: IntCounter,

    // number of completed offload transfers from host to disk
    pub offload_transfers_h2d_completed: IntCounter,

    // number of completed offload transfers from device to disk (bypassing host memory)
    pub offload_transfers_d2d_completed: IntCounter,

    // number of completed onboard transfers from host to device
    pub onboard_transfers_h2d_completed: IntCounter,

    // number of completed onboard transfers from disk to device
    pub onboard_transfers_d2d_completed: IntCounter,

    // number of matched tokens from KVBM
    pub matched_tokens: IntCounter,

    // size of offload queue from device to host
    pub offload_queue_d2h: IntGauge,

    // size of offload queue from host to disk
    pub offload_queue_h2d: IntGauge,

    // size of offload queue from device to disk (bypassing host memory)
    pub offload_queue_d2d: IntGauge,

    shutdown_notify: Option<Arc<Notify>>,
}

/// Worker-specific metrics that should only be tracked by workers, not the leader.
#[derive(Clone, Debug)]
pub struct KvbmWorkerMetrics {
    // number of requests in maybe_finished_onboarding set
    pub connector_maybe_finished_onboarding: IntGauge,

    // number of requests in maybe_finished_offloading set
    pub connector_maybe_finished_offloading: IntGauge,

    // number of pending offloading operations
    pub connector_offloading_operations: IntGauge,

    /// transfers started by worker (labels: direction=offload|onboard, pools=d2h|d2d|h2d)
    pub worker_transfers_started: IntCounterVec,
    /// transfers completed by worker (labels: direction=offload|onboard, pools=d2h|d2d|h2d)
    pub worker_transfers_completed: IntCounterVec,
    /// transfer size (blocks) observed by worker (labels: direction=offload|onboard, pools=d2h|d2d|h2d)
    pub worker_transfers_size_in_blocks: HistogramVec,

    shutdown_notify: Option<Arc<Notify>>,
}

impl KvbmMetrics {
    /// Create raw metrics and (once per process) spawn an axum server exposing `/metrics` at metrics_port.
    /// Non-blocking: the HTTP server runs on a background task.
    pub fn new(mr: &KvbmMetricsRegistry, create_endpoint: bool, metrics_port: u16) -> Self {
        // 1) register kvbm metrics
        let offload_blocks_d2h = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2H,
                "The number of offload blocks from device to host",
                &[],
            )
            .unwrap();
        let offload_blocks_h2d = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_H2D,
                "The number of offload blocks from host to disk",
                &[],
            )
            .unwrap();
        let offload_blocks_d2d = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2D,
                "The number of offload blocks from device to disk (bypassing host memory)",
                &[],
            )
            .unwrap();
        let onboard_blocks_h2d = mr
            .create_intcounter(
                ONBOARD_BLOCKS_H2D,
                "The number of onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d = mr
            .create_intcounter(
                ONBOARD_BLOCKS_D2D,
                "The number of onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let offload_blocks_d2h_completed = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2H_COMPLETED,
                "The number of completed offload blocks from device to host",
                &[],
            )
            .unwrap();
        let offload_blocks_h2d_completed = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_H2D_COMPLETED,
                "The number of completed offload blocks from host to disk",
                &[],
            )
            .unwrap();
        let offload_blocks_d2d_completed = mr
            .create_intcounter(
                OFFLOAD_BLOCKS_D2D_COMPLETED,
                "The number of completed offload blocks from device to disk (bypassing host memory)",
                &[],
            )
            .unwrap();
        let onboard_blocks_h2d_completed = mr
            .create_intcounter(
                ONBOARD_BLOCKS_H2D_COMPLETED,
                "The number of completed onboard blocks from host to device",
                &[],
            )
            .unwrap();
        let onboard_blocks_d2d_completed = mr
            .create_intcounter(
                ONBOARD_BLOCKS_D2D_COMPLETED,
                "The number of completed onboard blocks from disk to device",
                &[],
            )
            .unwrap();
        let offload_transfers_d2h = mr
            .create_intcounter(
                OFFLOAD_TRANSFERS_D2H,
                "The number of offload transfers from device to host",
                &[],
            )
            .unwrap();
        let offload_transfers_h2d = mr
            .create_intcounter(
                OFFLOAD_TRANSFERS_H2D,
                "The number of offload transfers from host to disk",
                &[],
            )
            .unwrap();
        let offload_transfers_d2d = mr
            .create_intcounter(
                OFFLOAD_TRANSFERS_D2D,
                "The number of offload transfers from device to disk (bypassing host memory)",
                &[],
            )
            .unwrap();
        let onboard_transfers_h2d = mr
            .create_intcounter(
                ONBOARD_TRANSFERS_H2D,
                "The number of onboard transfers from host to device",
                &[],
            )
            .unwrap();
        let onboard_transfers_d2d = mr
            .create_intcounter(
                ONBOARD_TRANSFERS_D2D,
                "The number of onboard transfers from disk to device",
                &[],
            )
            .unwrap();
        let offload_transfers_d2h_completed = mr
            .create_intcounter(
                OFFLOAD_TRANSFERS_D2H_COMPLETED,
                "The number of completed offload transfers from device to host",
                &[],
            )
            .unwrap();
        let offload_transfers_h2d_completed = mr
            .create_intcounter(
                OFFLOAD_TRANSFERS_H2D_COMPLETED,
                "The number of completed offload transfers from host to disk",
                &[],
            )
            .unwrap();
        let offload_transfers_d2d_completed = mr
            .create_intcounter(
                OFFLOAD_TRANSFERS_D2D_COMPLETED,
                "The number of completed offload transfers from device to disk (bypassing host memory)",
                &[],
            )
            .unwrap();
        let onboard_transfers_h2d_completed = mr
            .create_intcounter(
                ONBOARD_TRANSFERS_H2D_COMPLETED,
                "The number of completed onboard transfers from host to device",
                &[],
            )
            .unwrap();
        let onboard_transfers_d2d_completed = mr
            .create_intcounter(
                ONBOARD_TRANSFERS_D2D_COMPLETED,
                "The number of completed onboard transfers from disk to device",
                &[],
            )
            .unwrap();
        let matched_tokens = mr
            .create_intcounter(MATCHED_TOKENS, "The number of matched tokens", &[])
            .unwrap();
        let offload_queue_d2h = mr
            .create_intgauge(
                OFFLOAD_QUEUE_D2H,
                "The size of offload queue from device to host",
                &[],
            )
            .unwrap();
        let offload_queue_h2d = mr
            .create_intgauge(
                OFFLOAD_QUEUE_H2D,
                "The size of offload queue from host to disk",
                &[],
            )
            .unwrap();
        let offload_queue_d2d = mr
            .create_intgauge(
                OFFLOAD_QUEUE_D2D,
                "The size of offload queue from device to disk (bypassing host memory)",
                &[],
            )
            .unwrap();

        // Initialize all metrics with 0 to ensure they appear in metrics endpoint
        offload_blocks_d2h.inc_by(0);
        offload_blocks_h2d.inc_by(0);
        offload_blocks_d2d.inc_by(0);
        onboard_blocks_h2d.inc_by(0);
        onboard_blocks_d2d.inc_by(0);
        offload_blocks_d2h_completed.inc_by(0);
        offload_blocks_h2d_completed.inc_by(0);
        offload_blocks_d2d_completed.inc_by(0);
        onboard_blocks_h2d_completed.inc_by(0);
        onboard_blocks_d2d_completed.inc_by(0);
        offload_transfers_d2h.inc_by(0);
        offload_transfers_h2d.inc_by(0);
        offload_transfers_d2d.inc_by(0);
        onboard_transfers_h2d.inc_by(0);
        onboard_transfers_d2d.inc_by(0);
        offload_transfers_d2h_completed.inc_by(0);
        offload_transfers_h2d_completed.inc_by(0);
        offload_transfers_d2d_completed.inc_by(0);
        onboard_transfers_h2d_completed.inc_by(0);
        onboard_transfers_d2d_completed.inc_by(0);
        matched_tokens.inc_by(0);
        offload_queue_d2h.set(0);
        offload_queue_h2d.set(0);
        offload_queue_d2d.set(0);

        // early return if no endpoint is needed
        if !create_endpoint {
            return Self {
                offload_blocks_d2h,
                offload_blocks_h2d,
                offload_blocks_d2d,
                onboard_blocks_h2d,
                onboard_blocks_d2d,
                offload_blocks_d2h_completed,
                offload_blocks_h2d_completed,
                offload_blocks_d2d_completed,
                onboard_blocks_h2d_completed,
                onboard_blocks_d2d_completed,
                offload_transfers_d2h,
                offload_transfers_h2d,
                offload_transfers_d2d,
                onboard_transfers_h2d,
                onboard_transfers_d2d,
                offload_transfers_d2h_completed,
                offload_transfers_h2d_completed,
                offload_transfers_d2d_completed,
                onboard_transfers_h2d_completed,
                onboard_transfers_d2d_completed,
                matched_tokens,
                offload_queue_d2h,
                offload_queue_h2d,
                offload_queue_d2d,
                shutdown_notify: None,
            };
        }

        // 2) start HTTP server in background with graceful shutdown via Notify
        let registry = mr.inner(); // Arc<Registry>
        let notify = Arc::new(Notify::new());
        let notify_for_task = notify.clone();

        let addr = SocketAddr::from(([0, 0, 0, 0], metrics_port));
        let (_route_docs, app): (Vec<RouteDoc>, Router) = router(
            (*registry).clone(), // take owned Registry (Clone) for router to wrap in Arc
            None,                // or Some("/metrics".to_string()) to override the path
        );

        let run_server = async move {
            let listener = match TcpListener::bind(addr).await {
                Ok(listener) => listener,
                Err(err) => {
                    panic!("failed to bind metrics server to {addr}: {err}");
                }
            };

            if let Err(err) = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    // wait for shutdown signal
                    notify_for_task.notified().await;
                })
                .await
            {
                tracing::error!("[kvbm] metrics server error: {err}");
            }
        };

        // Spawn on existing runtime if present, otherwise start our own.
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(run_server);
        } else {
            thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("build tokio runtime");
                rt.block_on(run_server);
            });
        }

        Self {
            offload_blocks_d2h,
            offload_blocks_h2d,
            offload_blocks_d2d,
            onboard_blocks_h2d,
            onboard_blocks_d2d,
            offload_blocks_d2h_completed,
            offload_blocks_h2d_completed,
            offload_blocks_d2d_completed,
            onboard_blocks_h2d_completed,
            onboard_blocks_d2d_completed,
            offload_transfers_d2h,
            offload_transfers_h2d,
            offload_transfers_d2d,
            onboard_transfers_h2d,
            onboard_transfers_d2d,
            offload_transfers_d2h_completed,
            offload_transfers_h2d_completed,
            offload_transfers_d2d_completed,
            onboard_transfers_h2d_completed,
            onboard_transfers_d2d_completed,
            matched_tokens,
            offload_queue_d2h,
            offload_queue_h2d,
            offload_queue_d2d,
            shutdown_notify: Some(notify),
        }
    }
}

impl KvbmWorkerMetrics {
    /// Create worker-specific metrics and (once per process) spawn an axum server exposing `/metrics` at metrics_port.
    /// Non-blocking: the HTTP server runs on a background task.
    pub fn new(mr: &KvbmMetricsRegistry, create_endpoint: bool, metrics_port: u16) -> Self {
        let connector_maybe_finished_onboarding = mr
            .create_intgauge(
                CONNECTOR_MAYBE_FINISHED_ONBOARDING,
                "The number of requests in maybe_finished_onboarding set",
                &[],
            )
            .unwrap();
        let connector_maybe_finished_offloading = mr
            .create_intgauge(
                CONNECTOR_MAYBE_FINISHED_OFFLOADING,
                "The number of requests in maybe_finished_offloading set",
                &[],
            )
            .unwrap();
        let connector_offloading_operations = mr
            .create_intgauge(
                CONNECTOR_OFFLOADING_OPERATIONS,
                "The number of pending offloading operations",
                &[],
            )
            .unwrap();
        // Worker transfer metrics (Vec with labels)
        let registry = mr.inner();
        let worker_transfers_started = {
            let name = sanitize_prometheus_name("kvbm_worker_transfers_started")
                .expect("valid metric name");
            let opts = Opts::new(
                name,
                "Transfers started by worker (labels: direction, pools)",
            );
            let v =
                IntCounterVec::new(opts, &["direction", "pools"]).expect("create IntCounterVec");
            registry
                .register(Box::new(v.clone()))
                .expect("register IntCounterVec");
            v
        };
        let worker_transfers_completed = {
            let name = sanitize_prometheus_name("kvbm_worker_transfers_completed")
                .expect("valid metric name");
            let opts = Opts::new(
                name,
                "Transfers completed by worker (labels: direction, pools)",
            );
            let v =
                IntCounterVec::new(opts, &["direction", "pools"]).expect("create IntCounterVec");
            registry
                .register(Box::new(v.clone()))
                .expect("register IntCounterVec");
            v
        };
        let worker_transfers_size_in_blocks = {
            let name = sanitize_prometheus_name("kvbm_worker_transfers_size_in_blocks")
                .expect("valid metric name");
            // Custom buckets: 1..=MAX_TRANSFER_BATCH_SIZE (inclusive)
            let mut opts =
                HistogramOpts::new(name, "Transfer size in blocks (labels: direction, pools)");
            let buckets: Vec<f64> = (1..=MAX_TRANSFER_BATCH_SIZE)
                .map(|v| v as f64)
                .collect();
            opts.buckets = buckets;
            let v = HistogramVec::new(opts, &["direction", "pools"]).expect("create HistogramVec");
            registry
                .register(Box::new(v.clone()))
                .expect("register HistogramVec");
            v
        };

        // Initialize with 0
        connector_maybe_finished_onboarding.set(0);
        connector_maybe_finished_offloading.set(0);
        connector_offloading_operations.set(0);
        // Initialize all label combinations to appear in the metrics endpoint
        for dir in ["offload", "onboard"] {
            for pools in ["d2h", "d2d", "h2d"] {
                worker_transfers_started
                    .with_label_values(&[dir, pools])
                    .inc_by(0);
                worker_transfers_completed
                    .with_label_values(&[dir, pools])
                    .inc_by(0);
                // Do not observe 0 to avoid skewing histogram; just create the child by calling get_metric_with_label_values
                let _ = worker_transfers_size_in_blocks
                    .get_metric_with_label_values(&[dir, pools]);
            }
        }

        // early return if no endpoint is needed
        if !create_endpoint {
            return Self {
                connector_maybe_finished_onboarding,
                connector_maybe_finished_offloading,
                connector_offloading_operations,
                worker_transfers_started,
                worker_transfers_completed,
                worker_transfers_size_in_blocks,
                shutdown_notify: None,
            };
        }

        // start HTTP server in background with graceful shutdown via Notify
        let registry = mr.inner(); // Arc<Registry>
        let notify = Arc::new(Notify::new());
        let notify_for_task = notify.clone();

        let addr = SocketAddr::from(([0, 0, 0, 0], metrics_port));
        let (_route_docs, app): (Vec<RouteDoc>, Router) = router(
            (*registry).clone(), // take owned Registry (Clone) for router to wrap in Arc
            None,                // or Some("/metrics".to_string()) to override the path
        );

        let run_server = async move {
            let listener = match TcpListener::bind(addr).await {
                Ok(listener) => listener,
                Err(err) => {
                    panic!("failed to bind worker metrics server to {addr}: {err}");
                }
            };

            if let Err(err) = axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    // wait for shutdown signal
                    notify_for_task.notified().await;
                })
                .await
            {
                tracing::error!("[kvbm] worker metrics server error: {err}");
            }
        };

        // Spawn on existing runtime if present, otherwise start our own.
        if tokio::runtime::Handle::try_current().is_ok() {
            tokio::spawn(run_server);
        } else {
            thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("build tokio runtime");
                rt.block_on(run_server);
            });
        }

        Self {
            connector_maybe_finished_onboarding,
            connector_maybe_finished_offloading,
            connector_offloading_operations,
            worker_transfers_started,
            worker_transfers_completed,
            worker_transfers_size_in_blocks,
            shutdown_notify: Some(notify),
        }
    }
}

impl Drop for KvbmMetrics {
    fn drop(&mut self) {
        if let Some(n) = &self.shutdown_notify {
            // (all KvbmMetrics clones) + 1 (held by server task)
            // strong_count == 2 means this is the last metrics instance
            if Arc::strong_count(n) == 2 {
                n.notify_waiters();
            }
        }
    }
}

impl Drop for KvbmWorkerMetrics {
    fn drop(&mut self) {
        if let Some(n) = &self.shutdown_notify {
            // (all KvbmWorkerMetrics clones) + 1 (held by server task)
            // strong_count == 2 means this is the last metrics instance
            if Arc::strong_count(n) == 2 {
                n.notify_waiters();
            }
        }
    }
}

/// A raw, standalone Prometheus metrics registry implementation using the fixed prefix: `kvbm_`
#[derive(Debug, Clone)]
pub struct KvbmMetricsRegistry {
    registry: Arc<Registry>,
    prefix: String,
}

impl KvbmMetricsRegistry {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Registry::new()),
            prefix: "kvbm".to_string(),
        }
    }

    pub fn create_intcounter(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<IntCounter> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let const_labels: HashMap<String, String> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let opts = Opts::new(metrics_name, description).const_labels(const_labels);
        let c = IntCounter::with_opts(opts)?;
        self.registry.register(Box::new(c.clone()))?;
        Ok(c)
    }

    pub fn create_intgauge(
        &self,
        name: &str,
        description: &str,
        labels: &[(&str, &str)],
    ) -> anyhow::Result<IntGauge> {
        let metrics_name = sanitize_prometheus_name(&format!("{}_{}", self.prefix, name))?;
        let const_labels: HashMap<String, String> = labels
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let opts = Opts::new(metrics_name, description).const_labels(const_labels);
        let g = IntGauge::with_opts(opts)?;
        self.registry.register(Box::new(g.clone()))?;
        Ok(g)
    }

    pub fn inner(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }
}

impl Default for KvbmMetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}
