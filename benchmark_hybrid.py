import os
import time
import argparse
import torch
import numpy as np
import openvino as ov
import joblib

from model_dae import DAE

def export_encoder_to_openvino(model: torch.nn.Module, dummy_input: torch.Tensor):
    """Экспортируем только блок сжатия (энкодер) в OpenVINO"""
    print("Converting DAE Encoder directly to OpenVINO...")
    model.eval()
    
    # Забираем только энкодер
    encoder_module = model.encoder 
    
    ov_model = ov.convert_model(
        encoder_module, 
        example_input=dummy_input, 
        input=[1, 3, dummy_input.shape[2], dummy_input.shape[3]]
    )
    ov_model.reshape([1, 3, dummy_input.shape[2], dummy_input.shape[3]])
    
    return ov_model

def run_hybrid_benchmark(
    dae_path: str, clf_dir: str, device: str, image_size: int = 128, 
    warmup_runs: int = 20, production_runs: int = 500
):
    print(f"--- Starting Hybrid Pipeline Benchmark ---")
    print(f"Pipeline: DAE Encoder ({device.upper()}) + XGBoost (CPU)")
    
    # 1. Загружаем модели
    torch_device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    
    dae = DAE()
    if os.path.exists(dae_path):
        dae.load_state_dict(torch.load(dae_path, map_location=torch_device))
    else:
        print("Warning: DAE weights not found, using random.")
    dae.to(torch_device)
    dae.eval()

    scaler_path = os.path.join(clf_dir, "scaler.pkl")
    xgb_path = os.path.join(clf_dir, "xgb_model.pkl")
    
    print("Loading Scaler and XGBoost...")
    scaler = joblib.load(scaler_path)
    xgb_clf = joblib.load(xgb_path)

    # 2. Подготовка устройств
    is_openvino = False
    ov_compiled_encoder = None
    dummy_input = torch.randn(1, 3, image_size, image_size)

    if device == "npu":
        is_openvino = True
        core = ov.Core()
        target_device = "NPU" if "NPU" in core.available_devices else "CPU"
        
        ov_model = export_encoder_to_openvino(dae, dummy_input)
        print(f"Compiling Encoder for {target_device} with LATENCY hint...")
        
        ov_compiled_encoder = core.compile_model(
            ov_model, 
            target_device,
            config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f16"}
        )
        dummy_input_np = dummy_input.numpy()
    else:
        dummy_input = dummy_input.to(torch_device)

    print("3. Performing warm-up runs...")
    for _ in range(warmup_runs):
        if is_openvino:
            # Извлекаем признаки на NPU
            res = ov_compiled_encoder([dummy_input_np])
            latent = res[ov_compiled_encoder.output(0)]
        else:
            with torch.no_grad():
                latent = dae.encode(dummy_input).cpu().numpy()
        
        # Классификация на CPU
        latent_flat = latent.reshape(1, -1)
        scaled = scaler.transform(latent_flat)
        _ = xgb_clf.predict(scaled)

    print("4. Performing production runs...")
    if not is_openvino and torch_device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for _ in range(production_runs):
        if is_openvino:
            res = ov_compiled_encoder([dummy_input_np])
            latent = res[ov_compiled_encoder.output(0)]
        else:
            with torch.no_grad():
                latent = dae.encode(dummy_input).cpu().numpy()
        
        latent_flat = latent.reshape(1, -1)
        scaled = scaler.transform(latent_flat)
        _ = xgb_clf.predict(scaled)

    if not is_openvino and torch_device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency = (total_time / production_runs) * 1000 
    fps = production_runs / total_time

    print("\n--- Hybrid Benchmark Results ---")
    print(f"Total Time for {production_runs} runs: {total_time:.4f} seconds")
    print(f"Average Latency (End-to-End): {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dae_path", type=str, default="./models/dae_model.pth")
    parser.add_argument("--clf_dir", type=str, default="./results_supervised")
    parser.add_argument("--device", type=str, default="npu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()

    run_hybrid_benchmark(args.dae_path, args.clf_dir, args.device, production_runs=args.runs)