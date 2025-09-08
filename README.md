# 🧠 Citadel Triton Mesh — Distributed Master · Worker · Trainer

**Build**: `2025-09-07 03:52:33`
**Extension of**: NVIDIA Triton Inference Server
**Use Case**: Scalable inference and training across heterogeneous, air-gapped, and enterprise environments.

---

## 🧭 Introduction

NVIDIA Triton provides a robust base for serving AI models but lacks out-of-the-box support for:

* Seamless multi-node, multi-GPU, and cross-OS inference.
* Dynamic load balancing and failover in real-world latency scenarios.
* Integrated fine-tuning/training alongside inference.
* Operation in air-gapped or disconnected enterprise environments.

**Citadel Triton Mesh** solves these with a distributed fabric that introduces a **Master–Worker–Trainer** architecture:

* Meshes **Windows, Linux, macOS (CPU, GPU, TPU)** nodes.
* Predicts load using **seconds-of-work estimates**.
* Orchestrates fine-tuning with **soft GPU leasing**.
* Offers deep **observability** (telemetry, GPU memory snapshots).
* Deployable in **cloud, on-prem, edge, or classified** environments.

---

## 📦 Overview

**Citadel Triton Mesh** is a distributed inference and training framework.

* **Master Node**: Exposes public gRPC API and Registry.
* **Workers**: Register capabilities, stream telemetry, and run Triton with Citadel extensions.
* **Trainer Router**: Schedules fine-tuning jobs based on GPU load and availability.

### ✨ Key Capabilities

* **Cross-OS Workers**: Windows, Linux, macOS (CPU-only).
* **Load Prediction**: Seconds-of-work estimation using *Predicted Power-of-Two-Choices (P2C)*.
* **Replica-Aware Execution**: Chooses lowest-latency/backlog replica.
* **Streaming I/O Support**.
* **Soft-Leasing Training Scheduler**.
* **Scalable Architecture**: Both horizontally (workers) and vertically (replicas).
* **Telemetry & Logging**: Throughput, latency, GPU snapshots, logs, audit.

---

## 🏗️ Architecture

```
Clients
  │
  ├─ gRPC :9000 → Master (Inference + Trainer Router)
  │
  └─ gRPC :9100 → Registry/Admin (internal)
                     │
                     ├─ Worker A (Windows)
                     ├─ Worker B (Linux)
                     └─ Trainer daemons (optional)
```

### 🔌 Master Ports

* `9000`: Client ingress for inference/training.
* `9100`: Worker registry and admin interface.

### ⚙️ Worker Behavior

* Register with Master.
* Stream replica telemetry (queue depth, SPU, memory).
* Perform latency-aware routing.

### 🧑‍🏫 Trainer Behavior

* Advertise per-GPU training availability.
* Accept soft leases for fine-tuning jobs.

---

## 🚀 Starting Services

### 🪟 Windows (PowerShell)

```powershell
.\start_master.ps1
.\start_worker.ps1
```

### 🐧 Linux (bash)

```bash
./start_master.sh
./start_worker.sh
```

> Ensure workers can reach master on **port 9100**. Port **9000** is the public client ingress.

---

## 📈 Routing and Load Balancing

### 🔁 Inference — Predicted P2C

```python
inc_units = ceil(B / 32_000)
predicted(worker) = (queued * SPU) + (inc_units * SPU)
```

Two workers sampled → route to lower `predicted(worker)` value.
Failover and client deadlines enforced.

### ⚙️ Worker-Local Replica Scoring

```python
score = (A * EWMA_latency) + (B * queue_depth)
        + (queued + running) * SPU
```

Concurrency limits and backpressure handled via token pools.

### 🧠 Trainer Routing

* Estimate job size from payload.
* Rank `(worker, GPU)` by `seconds-of-work + free VRAM`.
* Soft lease best candidate.
* Failover if job fails to start.

---

## 🧩 Models and Replicas

### 📁 Directory Layout

```
citadel_worker/models/<ModelName>/
  ├─ config.json
  └─ model.py  # implements CitadelModel
```

### 🔧 Execution Modes

* **In-process**: Fastest, no isolation.
* **Sub-process**: Isolated via Conda/venv, using `replica_server.py` with socket IPC.

### 🔄 CitadelModel API (Summary)

```python
init()              # Load weights
execute(input)      # Inference (str/bytes/dict)
execute_batch(list) # Optional batching
stream_execute()    # Optional streaming
finalize()          # Cleanup
```

---

## 📡 Streaming Inputs

* Clients send chunked JSON (e.g., `audio_data` with `session_id`, `chunk_index`).
* Server reassembles into a full payload.
* Streaming responses include `meta_json` with timing and device stats.

---

## 🏋️ Training

### 🔁 Via Inference Channel

```json
{
  "level": "train",
  "knowledge": ["/path/file.pdf"],
  "hyperparams": {"epochs": "1"}
}
```

Or stream chunks with `{file_name, chunk_index, chunk_b64}`.

### 🔗 Direct Trainer API

```python
TrainerStub.StartJob(TrainRequest)
```

Master router selects `(worker, gpu)` and streams progress.

---

## 📊 Scaling

### ➕ Horizontal

* Add workers on any reachable OS.
* Auto-registration and telemetry.

### ⬆️ Vertical

* Increase per-GPU replicas and concurrency via `config.json`.

---

## 🛡️ Fault Tolerance

* **Inference**: Retry other workers if error or timeout.
* **Training**: Failover if no initial progress.

---

## 🔍 Observability

* Request lifecycle metrics (queued, started, completed).
* Replica telemetry: queue depth, SPU, latency.
* GPU stats: free/total memory.
* Structured logs.
* gRPC stats surface for internal tools.

---

## 🧪 Client Examples (Python)

### 📥 Inference

```python
import grpc
from proto.python import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel("MASTER_HOST:9000")
stub = inference_pb2_grpc.InferenceStub(channel)

req = inference_pb2.InferenceRequest(model="ModelA", raw=b"hello world")
for rsp in stub.Infer(iter([req])):
    if rsp.error:
        raise RuntimeError(rsp.error)
    print(bytes(rsp.raw or b""))
```

### 🧠 Start Training

```python
import grpc
from proto.python import trainer_pb2, trainer_pb2_grpc

channel = grpc.insecure_channel("MASTER_HOST:9000")
stub = trainer_pb2_grpc.TrainerStub(channel)

req = trainer_pb2.TrainRequest(
    model_name="llm",
    name="demo-run",
    knowledge=["C:\\data\\doc1.pdf"],
    hyperparams={"epochs": "1", "lr": "1e-4"},
)

for prog in stub.StartJob(req):
    print(prog.stage, prog.step, prog.loss, prog.message)
```

---

## 🔐 Security & Networking

### 🔒 Port Scoping

* `9100`: Registry/Admin (worker ↔ master).
* `9000`: Public inference/training ingress.

### 📡 TLS + Proxies

* Recommended: TLS-terminating proxy in front of `:9000`.

### 🛰️ Air-Gapped Support

* No external services required. Fully internal mesh ops.

### 🧾 Access Control

* SQL execution (if enabled) gated by token.
* Optional RBAC, audit logging for classified setups.

---

## 🧰 Troubleshooting

| Symptom                          | Resolution                                                                |
| -------------------------------- | ------------------------------------------------------------------------- |
| Workers not visible              | Check port `9100` connectivity and script configs                         |
| GPU memory shows `0`             | Ensure `nvidia-smi` is available; CPU-only continues without VRAM metrics |
| Timeouts on large uploads        | Increase timeouts or switch to streaming                                  |
| Sub-process replica startup fail | Validate environment path and dependencies                                |

---

## 📚 Glossary

| Term               | Definition                                                          |
| ------------------ | ------------------------------------------------------------------- |
| **Triton Server**  | NVIDIA’s inference server base                                      |
| **Worker**         | Triton + Citadel agent (routing, telemetry, scheduling)             |
| **Master**         | Coordinator: gRPC API, Registry, job router                         |
| **Trainer Daemon** | Optional fine-tuning orchestrator colocated with Worker             |
| **Work-unit**      | `ceil(bytes / 32 KiB)`                                              |
| **SPU**            | Seconds-per-unit, estimated per replica                             |
| **Predicted P2C**  | Two-choice worker selection using load prediction                   |
| **Soft lease**     | Temporary job reservation on a (worker, gpu)                        |
| **Mesh**           | Citadel’s overlay for distributed routing, telemetry, orchestration |

