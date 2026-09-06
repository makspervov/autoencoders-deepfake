import argparse
import time
import os
import torch
import openvino as ov

# Import all models
from model_vae import VAE
from model_dae import DAE
from model_sae import SAE


def export_to_openvino(
    model: torch.nn.Module, dummy_input: torch.Tensor, model_name: str
):
    """
    Directly converts a PyTorch model to OpenVINO Intermediate Representation (IR) without ONNX.
    """
    print(f"Converting PyTorch model '{model_name}' directly to OpenVINO...")
    model.eval()
    
    # Прямая конвертация с жесткой фиксацией Shape
    ov_model = ov.convert_model(
        model, 
        example_input=dummy_input, 
        input=[1, 3, dummy_input.shape[2], dummy_input.shape[3]]
    )
    ov_model.reshape([1, 3, dummy_input.shape[2], dummy_input.shape[3]])
    
    return ov_model


def run_benchmark(
    model_name: str,
    device: str,
    image_size: int = 128,
    warmup_runs: int = 20,
    production_runs: int = 500,
):
    print(f"--- Starting Benchmark ---")
    print(f"Model: {model_name.upper()}")
    print(f"Device: {device.upper()}")
    print(f"Image Size: {image_size}x{image_size}")
    print(f"Warm-up Runs: {warmup_runs}")
    print(f"Production Runs: {production_runs}\n")

    # 1. Initialize the model
    print("1. Initializing model...")
    if model_name.lower() == "vae":
        model = VAE()
    elif model_name.lower() == "dae":
        model = DAE()
    elif model_name.lower() == "sae":
        model = SAE()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()

    # 2. Select device and prepare model/dummy input
    print("2. Preparing device and dummy data...")
    is_openvino = False
    ov_compiled_model = None

    if device == "npu":
        is_openvino = True
        torch_device = torch.device("cpu")
        dummy_input = torch.randn(1, 3, image_size, image_size)

        core = ov.Core()
        available_devices = core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")

        target_device = "NPU" if "NPU" in available_devices else "CPU"
        if target_device == "CPU":
            print("Warning: NPU not found in OpenVINO available devices. Falling back to OpenVINO CPU.")

        ov_model = export_to_openvino(model, dummy_input, model_name)
        print(f"Compiling OpenVINO model for {target_device} with LATENCY hint...")
        
        # Сжимаем до FP16 прямо в памяти (без сохранения на диск) и задаем приоритет Latency
        ov_compiled_model = core.compile_model(
            ov_model, 
            target_device,
            config={
                "PERFORMANCE_HINT": "LATENCY",
                "INFERENCE_PRECISION_HINT": "f16" 
            }
        )
        dummy_input_np = dummy_input.numpy()

    else:
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        torch_device = torch.device(device)
        model.to(torch_device)
        dummy_input = torch.randn(1, 3, image_size, image_size).to(torch_device)

    # 3. Perform "warm-up" runs
    print("3. Performing warm-up runs...")
    if is_openvino:
        for _ in range(warmup_runs):
            _ = ov_compiled_model([dummy_input_np])
    else:
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(dummy_input)

    # 4. Perform production runs and measure time
    print("4. Performing production runs...")

    # Synchronize GPU before starting the timer
    if not is_openvino and torch_device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    if is_openvino:
        for _ in range(production_runs):
            _ = ov_compiled_model([dummy_input_np])
    else:
        with torch.no_grad():
            for _ in range(production_runs):
                _ = model(dummy_input)

    # Synchronize GPU after runs to get accurate time
    if not is_openvino and torch_device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency = (total_time / production_runs) * 1000  # in milliseconds
    fps = production_runs / total_time

    # 5. Output Results
    print("\n--- Benchmark Results ---")
    print(f"Total Time for {production_runs} runs: {total_time:.4f} seconds")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performance Benchmark Script")
    parser.add_argument(
        "--model",
        type=str,
        default="vae",
        choices=["vae", "dae", "sae"],
        help="Model architecture to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "npu"],
        help="Processing backend",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Input image resolution",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warm-up runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=500,
        help="Number of production runs to measure",
    )

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        device=args.device,
        image_size=args.image_size,
        warmup_runs=args.warmup,
        production_runs=args.runs,
    )
