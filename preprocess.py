import os
import argparse
import glob
from tqdm import tqdm
from PIL import Image
from facenet_pytorch import MTCNN


def extract_faces(
    input_dir: str,
    output_dir: str,
    image_size: int = 128,
    margin: int = 20,
    device: str = "cpu",
):
    """
    Extract faces from all images in input_dir and save them to output_dir.
    Keeps directory structure intact.
    """
    print(
        f"Extracting faces from '{input_dir}' to '{output_dir}' using {device.upper()}..."
    )

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        keep_all=False,
        device=device,
        post_process=False,  # We want raw pixel values, not pre-whitened for Facenet
    )

    # Find all image files
    # The FaceForensics dataset typically has standard image extensions if frames are extracted,
    # or if we are processing extracted frames.
    image_paths = []
    for ext in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext), recursive=True))

    if not image_paths:
        print(f"No images found in {input_dir}. Please check your dataset path.")
        return

    print(f"Found {len(image_paths)} images. Starting processing...")

    successful_extractions = 0
    failed_extractions = 0

    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Recreate directory structure
            rel_path = os.path.relpath(img_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)

            # Skip if already exists
            if os.path.exists(out_path):
                successful_extractions += 1
                continue

            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Open image
            img = Image.open(img_path).convert("RGB")

            # Detect and extract face
            # MTCNN returns a cropped and resized tensor if face is found
            # However, since post_process=False, it returns values in range [0, 255]
            img_cropped = mtcnn(img)

            if img_cropped is not None:
                # Convert back to PIL Image and save
                # img_cropped is (C, H, W). Permute to (H, W, C), convert to numpy, then Image
                img_cropped = img_cropped.permute(1, 2, 0).cpu().numpy().astype("uint8")
                res_img = Image.fromarray(img_cropped)
                res_img.save(out_path)
                successful_extractions += 1
            else:
                failed_extractions += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            failed_extractions += 1

    print("\n--- Extraction Summary ---")
    print(f"Total images processed: {len(image_paths)}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions (No face detected or error): {failed_extractions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract faces from images using MTCNN"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/raw",
        help="Input directory containing original images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for cropped faces",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Output size of cropped faces (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run MTCNN on",
    )

    args = parser.parse_args()

    extract_faces(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        device=args.device,
    )
