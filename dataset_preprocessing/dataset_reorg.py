"""
Badminton Video Dataset Reorganization Tool

Reorganizes raw badminton video datasets containing match folders into standard
machine learning training format. Automatically extracts video frames, renames
CSV annotation files, and generates data structure suitable for deep learning models.

Usage Examples:
    python dataset_reorg.py --source dataset --output dataset_ml     # Basic: input folder -> output folder
    python dataset_reorg.py --source /path/to/data --output /path/to/output --force     # Custom paths, overwrite existing
    python dataset_reorg.py --source dataset --check-only     # Validate structure only, no conversion

Dependencies:
    pip install opencv-python
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import cv2


# Files and folders to ignore during processing
INVALID_FILES = {'.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'}
INVALID_FOLDERS = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules'}


def is_valid_item(item_name):
    """Check if file or folder name is valid for processing."""
    if item_name.startswith('.') and item_name not in {'.', '..'}:
        return False
    return item_name not in INVALID_FILES and item_name not in INVALID_FOLDERS


def validate_source_structure(source_folder):
    """Validate source folder structure and return summary."""
    print(f"🔍 Checking source folder: {source_folder}")

    if not os.path.exists(source_folder):
        return False, f"Source folder does not exist: {source_folder}"

    # Find match folders
    all_items = [item for item in os.listdir(source_folder) if is_valid_item(item)]
    match_folders = [
        item for item in all_items
        if item.startswith("match") and os.path.isdir(os.path.join(source_folder, item))
    ]

    if not match_folders:
        return False, "No match folders found"

    # Validate each match folder structure
    valid_matches = 0
    total_videos = 0
    total_csvs = 0

    for match_folder in match_folders:
        match_path = os.path.join(source_folder, match_folder)
        csv_path = os.path.join(match_path, "csv")
        video_path = os.path.join(match_path, "video")

        if os.path.exists(csv_path) and os.path.exists(video_path):
            valid_matches += 1

            # Count CSV files
            csv_files = [f for f in os.listdir(csv_path)
                         if f.endswith('_ball.csv') and is_valid_item(f)]
            total_csvs += len(csv_files)

            # Count video files
            video_files = [f for f in os.listdir(video_path)
                           if f.endswith('.mp4') and is_valid_item(f)]
            total_videos += len(video_files)

    if valid_matches == 0:
        return False, "No valid match folders found (must contain both csv and video)"

    summary = f"✅ Found {valid_matches} match folders, {total_videos} videos, {total_csvs} CSV files"
    return True, summary


def extract_frames_from_video(video_path, output_folder):
    """Extract frames from video file and save as JPG sequence."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ❌ Cannot open: {os.path.basename(video_path)}")
        return False

    frame_count = 0
    jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]  # High quality JPG

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame numbering starts from 0
        frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame, jpg_quality)
        frame_count += 1

    cap.release()
    print(f"    ✅ {os.path.basename(video_path)} -> {frame_count} frames")
    return True


def process_match_folder(match_folder_path, output_match_folder, current_idx, total_count):
    """Process a single match folder."""
    match_name = os.path.basename(match_folder_path)
    print(f"\n📁 [{current_idx}/{total_count}] Processing {match_name}")

    # Create output folder structure
    inputs_folder = os.path.join(output_match_folder, "inputs")
    labels_folder = os.path.join(output_match_folder, "labels")
    os.makedirs(inputs_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # Process video folder
    video_count = 0
    video_folder = os.path.join(match_folder_path, "video")
    if os.path.exists(video_folder):
        mp4_files = [f for f in os.listdir(video_folder)
                     if f.endswith('.mp4') and is_valid_item(f)]

        if mp4_files:
            print(f"  🎬 Converting {len(mp4_files)} videos:")
            for mp4_file in mp4_files:
                video_path = os.path.join(video_folder, mp4_file)
                video_name = Path(mp4_file).stem
                video_output_folder = os.path.join(inputs_folder, video_name)

                if extract_frames_from_video(video_path, video_output_folder):
                    video_count += 1

    # Process CSV folder
    csv_count = 0
    csv_folder = os.path.join(match_folder_path, "csv")
    if os.path.exists(csv_folder):
        csv_files = [f for f in os.listdir(csv_folder)
                     if f.endswith('_ball.csv') and is_valid_item(f)]

        if csv_files:
            print(f"  📄 Copying {len(csv_files)} CSV files:")
            for csv_file in csv_files:
                csv_path = os.path.join(csv_folder, csv_file)
                original_name = Path(csv_file).stem
                new_name = original_name.replace("_ball", "") + ".csv"

                destination_path = os.path.join(labels_folder, new_name)
                shutil.copy2(csv_path, destination_path)
                print(f"    ✅ {csv_file} -> {new_name}")
                csv_count += 1

    print(f"  ✅ Completed: {video_count} videos, {csv_count} CSV files")


def reorganize_dataset(source_folder, output_folder, force=False):
    """Reorganize the entire dataset."""
    # Validate source folder structure
    is_valid, message = validate_source_structure(source_folder)
    if not is_valid:
        print(f"❌ {message}")
        return False

    print(message)

    # Handle existing output folder
    if os.path.exists(output_folder):
        if force:
            print(f"🗑️  Removing existing folder: {output_folder}")
            shutil.rmtree(output_folder)
        else:
            response = input(f"⚠️  Output folder exists: {output_folder}\n   Delete and rebuild? (y/n): ")
            if response.lower() != 'y':
                print("❌ Operation cancelled")
                return False
            shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 Created output folder: {output_folder}")

    # Find all valid match folders
    all_items = [item for item in os.listdir(source_folder) if is_valid_item(item)]
    match_folders = [
        item for item in all_items
        if item.startswith("match") and os.path.isdir(os.path.join(source_folder, item))
    ]

    # Filter valid match folders
    valid_matches = []
    for match_folder_name in match_folders:
        source_match_path = os.path.join(source_folder, match_folder_name)
        csv_exists = os.path.exists(os.path.join(source_match_path, "csv"))
        video_exists = os.path.exists(os.path.join(source_match_path, "video"))

        if csv_exists and video_exists:
            valid_matches.append(match_folder_name)
        else:
            print(f"⚠️  Skipping {match_folder_name}: missing csv or video folder")

    if not valid_matches:
        print("❌ No valid match folders found")
        return False

    print(f"🚀 Processing {len(valid_matches)} match folders...")

    # Process each match folder
    for idx, match_folder_name in enumerate(valid_matches, 1):
        source_match_path = os.path.join(source_folder, match_folder_name)
        target_match_path = os.path.join(output_folder, match_folder_name)

        os.makedirs(target_match_path, exist_ok=True)
        process_match_folder(source_match_path, target_match_path, idx, len(valid_matches))

    print(f"\n🎉 Reorganization completed!")
    print(f"   Source folder: {source_folder}")
    print(f"   Output folder: {output_folder}")
    return True


def main():
    """Main function - handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Badminton Video Dataset Reorganization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Structure:
    dataset/
    ├── match1/
    │   ├── csv/
    │   │   └── rally1_ball.csv
    │   └── video/
    │       └── rally1.mp4
    └── match2/...

Output Structure:
    dataset_ml/
    ├── match1/
    │   ├── inputs/
    │   │   └── rally1/
    │   │       ├── 0.jpg
    │   │       ├── 1.jpg
    │   │       └── ...
    │   └── labels/
    │       └── rally1.csv
    └── match2/...
        """
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source folder path containing match1, match2, etc. subfolders"
    )

    parser.add_argument(
        "--output",
        help="Output folder path (default: source_folder + '_ml')"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing output folder"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate source folder structure, don't execute conversion"
    )

    args = parser.parse_args()

    print("🎬 Badminton Video Dataset Reorganization Tool")
    print("=" * 50)

    # Check OpenCV installation
    try:
        print(f"📦 OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed: pip install opencv-python")
        sys.exit(1)

    # Set default output folder if not provided
    if not args.output:
        args.output = f"{args.source}_ml"

    # Only check structure
    if args.check_only:
        is_valid, message = validate_source_structure(args.source)
        print(message)
        sys.exit(0 if is_valid else 1)

    # Execute reorganization
    success = reorganize_dataset(args.source, args.output, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()