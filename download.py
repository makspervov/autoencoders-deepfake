import os
import argparse
import zipfile
import glob
from huggingface_hub import snapshot_download


def download_dataset(dataset_name: str, download_path: str):
    """
    Downloads the dataset from Hugging Face Hub using huggingface_hub.
    Extracts any downloaded zip files automatically.
    """
    print(f"Downloading dataset '{dataset_name}' to '{download_path}'...")

    # Ensure download directory exists
    os.makedirs(download_path, exist_ok=True)

    try:
        # Download the dataset using snapshot_download
        downloaded_path = snapshot_download(
            repo_id=dataset_name,
            repo_type="dataset",
            local_dir=download_path,
            local_dir_use_symlinks=False
        )
        print("Dataset downloaded successfully!")

        # Unzip any zip files in the downloaded directory
        print("Checking for zip files to extract...")
        zip_files = glob.glob(os.path.join(download_path, "**/*.zip"), recursive=True)
        for zip_file in zip_files:
            print(f"Extracting {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Extract into the same directory as the zip file
                extract_dir = os.path.dirname(zip_file)
                zip_ref.extractall(extract_dir)
            # Remove the zip file after extraction to save space
            os.remove(zip_file)

        print("Extraction complete!")
    except Exception as e:
        print(f"Error downloading or extracting the dataset: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Dataset from Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="OpenRL/DeepFakeFace",
        help="Hugging Face dataset name (default: OpenRL/DeepFakeFace)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/raw",
        help="Path to download the dataset to (default: ./data/raw)",
    )

    args = parser.parse_args()
    download_dataset(args.dataset, args.path)
