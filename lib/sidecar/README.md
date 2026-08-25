# Sidecars

Rust sidecars connect Dynamo workers to inference engines over their native
gRPC APIs. Dynamo owns worker registration and request handling; the engine
runs in a separate process.

```text
Dockerfile  CPU-only image containing all three sidecar executables
common/  Shared gRPC arguments, transport, and errors
sglang/  SGLang sidecar
trtllm/  TensorRT-LLM sidecar
vllm/    vLLM sidecar
```

Engine protocols and request conversion remain in each engine's crate.

Build the unified image from the repository root:

```bash
docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:1.3.0 .
```

The image deliberately has no default entrypoint. Select the executable for
the colocated engine with the container command:

```text
dynamo-vllm-sidecar
dynamo-sglang-sidecar
dynamo-trtllm-sidecar
```
