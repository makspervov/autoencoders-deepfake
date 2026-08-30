import os
import argparse
import glob
import multiprocessing
import torch
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Global variable to hold the MTCNN instance for each worker process.
worker_mtcnn = None

def init_worker(image_size, margin, device):
    """
    Initialization function for each worker process.
    """
    global worker_mtcnn
    
    torch.set_num_threads(1)
    
    worker_mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        keep_all=False,
        device=device,
        post_process=False,
    )


def process_single_image(args):
    """
    Process a single image. Accepts an image in its original resolution
    to avoid discarding high-resolution photos.
    """
    img_path, input_dir, output_dir = args
    try:
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)

        if os.path.exists(out_path):
            return True

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        img_cropped = worker_mtcnn(img)

        if img_cropped is not None:
            img_cropped = img_cropped.permute(1, 2, 0).cpu().numpy().astype("uint8")
            res_img = Image.fromarray(img_cropped)
            res_img.save(out_path)
            return True
        return False
        
    except Exception:
        return False


def get_optimal_device(requested_device):
    # Automatically determine the best device to use based on availability and user preference.
    if requested_device != "auto":
        return requested_device
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn' to avoid issues on certain platforms (like macOS).
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Universal Face Extraction (CPU/GPU)")
    parser.add_argument("--input_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', 'mps'")
    parser.add_argument("--workers", type=int, default=0, help="0 = auto, >0 = number of parallel threads")
    args = parser.parse_args()

    active_device = get_optimal_device(args.device)

    # Determine the number of worker processes based on the device and user input.
    if args.workers == 0:
        if active_device in ["cuda", "mps"]:
            # GPU: set the number of workers to the minimum of 4 or the number of CPU cores to avoid overloading the system.
            num_workers = min(4, multiprocessing.cpu_count())
        else:
            # CPU: use all cores minus 1, to keep the system responsive.
            num_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_workers = args.workers

    print(f"Extracting faces from '{args.input_dir}' to '{args.output_dir}'")
    print(f"Device: {active_device.upper()} | Parallel Threads: {num_workers}")

    image_paths = []
    for ext in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(args.input_dir, ext), recursive=True))

    if not image_paths:
        print(f"Images not found in {args.input_dir}.")
        exit()

    tasks = [(path, args.input_dir, args.output_dir) for path in image_paths]
    
    successful = 0
    failed = 0

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.image_size, 20, active_device)
    ) as pool:
        iterator = pool.imap_unordered(process_single_image, tasks)
        
        for success in tqdm(iterator, total=len(tasks), desc="Processing"):
            if success:
                successful += 1
            else:
                failed += 1

    print("\n--- Results ---")
    print(f"Total files: {len(image_paths)}")
    print(f"Successful: {successful}")
    print(f"Not found/Errors: {failed}")