import os
import argparse
import subprocess


def download_dataset(dataset_name: str, download_path: str):
    """
    Downloads the dataset from Kaggle using the Kaggle API.
    Assumes Kaggle credentials (kaggle.json) are configured properly.
    """
    print(f"Downloading dataset '{dataset_name}' to '{download_path}'...")

    # Ensure download directory exists
    os.makedirs(download_path, exist_ok=True)

    try:
        # Use subprocess to run the kaggle cli command
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_name,
                "-p",
                download_path,
                "--unzip",
            ],
            check=True,
        )
        print("Dataset downloaded and extracted successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the dataset: {e}")
        print(
            "Please ensure your Kaggle credentials are set up correctly in ~/.kaggle/kaggle.json"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FaceForensics++ Dataset from Kaggle"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="xdxd003/ff-c23",
        help="Kaggle dataset name (default: xdxd003/ff-c23)",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data/raw",
        help="Path to download the dataset to (default: ./data/raw)",
    )

    args = parser.parse_args()
    download_dataset(args.dataset, args.path)
