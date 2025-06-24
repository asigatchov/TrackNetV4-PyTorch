"""
Dataset Reorganization Processor

Processes reorganized badminton dataset for training. Resizes images to 512×288 resolution
(maintaining aspect ratio), transforms CSV annotation coordinates, and generates heatmaps
for shuttlecock position detection training.

Usage Examples:
    python dataset_reorg_processor.py --source dataset_reorg     # Basic: auto output to dataset_reorg_train
    python dataset_reorg_processor.py --source /path/to/data --output /path/to/train     # Custom output path
    python dataset_reorg_processor.py --source dataset_reorg --sigma 5     # Custom heatmap sigma (default: 3)

Dependencies:
    pip install opencv-python pandas numpy scipy
"""

import argparse
import glob
import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def create_gaussian_heatmap(center_x, center_y, width=512, height=288, sigma=3):
    """Generate 2D Gaussian heatmap centered at specified coordinates."""
    x = np.arange(0, width)
    y = np.arange(0, height)
    xx, yy = np.meshgrid(x, y)
    pos = np.dstack((xx, yy))

    mean = [center_x, center_y]
    cov = [[sigma ** 2, 0], [0, sigma ** 2]]

    rv = multivariate_normal(mean, cov)
    heatmap = rv.pdf(pos)

    # Normalize to 0-255 range
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap


def resize_image_keep_ratio(image, target_width=512, target_height=288):
    """Resize image to target size while maintaining aspect ratio."""
    h, w = image.shape[:2]

    # Calculate scaling ratio
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)  # Choose smaller ratio to maintain aspect ratio

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas and center the image
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    start_x = (target_width - new_w) // 2
    start_y = (target_height - new_h) // 2
    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized

    return canvas, scale, start_x, start_y


def transform_coordinates(x, y, scale, offset_x, offset_y):
    """Transform coordinates based on image scaling ratio."""
    new_x = x * scale + offset_x
    new_y = y * scale + offset_y
    return new_x, new_y


def process_video(input_dir, label_file, output_inputs_dir, output_heatmaps_dir, video_name, sigma):
    """Process all frames of a single video."""
    print(f"    Processing {video_name}...")

    # Create output directories
    video_inputs_dir = os.path.join(output_inputs_dir, video_name)
    video_heatmaps_dir = os.path.join(output_heatmaps_dir, video_name)
    os.makedirs(video_inputs_dir, exist_ok=True)
    os.makedirs(video_heatmaps_dir, exist_ok=True)

    # Read annotation file
    if not os.path.exists(label_file):
        print(f"      Warning: Label file not found {label_file}")
        return

    try:
        df = pd.read_csv(label_file)
    except Exception as e:
        print(f"      Error: Cannot read label file {label_file}: {e}")
        return

    # Get all image files
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    image_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    # Get frame numbers from images and CSV
    image_frames = {int(os.path.splitext(os.path.basename(f))[0]) for f in image_files}
    csv_frames = set(df['Frame'].values)

    # Find strictly matching frames
    matched_frames = image_frames & csv_frames
    only_image_frames = image_frames - csv_frames
    only_csv_frames = csv_frames - image_frames

    processed_count = 0

    # Process only matched frames
    for frame_num in sorted(matched_frames):
        image_file = os.path.join(input_dir, f"{frame_num}.jpg")

        # Read image
        image = cv2.imread(image_file)
        if image is None:
            print(f"      Warning: Cannot read image {image_file}")
            continue

        # Get corresponding annotation
        frame_data = df[df['Frame'] == frame_num]
        frame_row = frame_data.iloc[0]

        # Resize image
        resized_image, scale, offset_x, offset_y = resize_image_keep_ratio(image)

        # Generate heatmap based on visibility
        if frame_row['Visibility'] == 1:
            # Visible frame: generate heatmap with annotation point
            orig_x = frame_row['X']
            orig_y = frame_row['Y']

            if pd.isna(orig_x) or pd.isna(orig_y):
                print(f"      Warning: Frame {frame_num} has invalid coordinates, generating zero heatmap")
                heatmap = np.zeros((288, 512), dtype=np.uint8)
            else:
                # Transform coordinates
                new_x, new_y = transform_coordinates(orig_x, orig_y, scale, offset_x, offset_y)

                # Ensure coordinates are within valid range
                new_x = max(0, min(511, new_x))
                new_y = max(0, min(287, new_y))

                # Generate heatmap
                heatmap = create_gaussian_heatmap(new_x, new_y, sigma=sigma)
        else:
            # Invisible frame: generate zero heatmap
            heatmap = np.zeros((288, 512), dtype=np.uint8)

        # Save files
        output_image_path = os.path.join(video_inputs_dir, f"{frame_num}.jpg")
        output_heatmap_path = os.path.join(video_heatmaps_dir, f"{frame_num}.jpg")

        cv2.imwrite(output_image_path, resized_image)
        cv2.imwrite(output_heatmap_path, heatmap)

        processed_count += 1

    # Summary for each video
    print(f"      Completed {processed_count} frames", end="")

    # Report mismatched cases
    issues = []
    if only_image_frames:
        frames_str = ",".join(map(str, sorted(only_image_frames)))
        issues.append(f"Images without CSV records: {frames_str}")

    if only_csv_frames:
        frames_str = ",".join(map(str, sorted(only_csv_frames)))
        issues.append(f"CSV records without images: {frames_str}")

    if issues:
        print(f", skipped {'; '.join(issues)}")
    else:
        print()


def process_match(match_dir, output_dir, sigma):
    """Process a single match folder."""
    match_name = os.path.basename(match_dir)
    print(f"  Processing {match_name}...")

    # Create output folders (keep original match name)
    output_match_dir = os.path.join(output_dir, match_name)
    output_inputs_dir = os.path.join(output_match_dir, "inputs")
    output_heatmaps_dir = os.path.join(output_match_dir, "heatmaps")

    os.makedirs(output_inputs_dir, exist_ok=True)
    os.makedirs(output_heatmaps_dir, exist_ok=True)

    # Get inputs and labels directories
    inputs_dir = os.path.join(match_dir, "inputs")
    labels_dir = os.path.join(match_dir, "labels")

    if not os.path.exists(inputs_dir):
        print(f"    Warning: inputs directory not found {inputs_dir}")
        return

    if not os.path.exists(labels_dir):
        print(f"    Warning: labels directory not found {labels_dir}")
        return

    # Get all video folders
    video_dirs = [d for d in os.listdir(inputs_dir)
                  if os.path.isdir(os.path.join(inputs_dir, d))]
    video_dirs.sort()

    for video_name in video_dirs:
        video_input_dir = os.path.join(inputs_dir, video_name)
        video_label_file = os.path.join(labels_dir, f"{video_name}.csv")

        process_video(video_input_dir, video_label_file,
                      output_inputs_dir, output_heatmaps_dir, video_name, sigma)


def main():
    """Main function - handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset Reorganization Processor for Badminton Shuttlecock Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input Structure:
    dataset_reorg/
    ├── match1/
    │   ├── inputs/video1/0.jpg,1.jpg...
    │   └── labels/video1.csv
    └── match2/...

Output Structure:
    dataset_reorg_train/
    ├── match1/
    │   ├── inputs/video1/0.jpg,1.jpg... (512×288)
    │   └── heatmaps/video1/0.jpg,1.jpg... (Gaussian heatmaps)
    └── match2/...

CSV Format Example (video1.csv):
    Frame,Visibility,X,Y
    0,1,637.0,346.0
    1,1,639.0,346.0
    2,0,640.0,345.0  # Visibility=0 generates zero heatmap
        """
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Source dataset_reorg folder path"
    )

    parser.add_argument(
        "--output",
        help="Output directory path (default: source_folder + '_train')"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=3.0,
        help="Gaussian heatmap standard deviation (default: 3.0)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.source):
        print(f"❌ Source dataset path does not exist: {args.source}")
        return

    # Set default output directory
    if not args.output:
        dataset_name = os.path.basename(args.source.rstrip('/'))
        parent_dir = os.path.dirname(args.source)
        args.output = os.path.join(parent_dir, f"{dataset_name}_train")

    os.makedirs(args.output, exist_ok=True)

    print("🏸 Dataset Reorganization Processor")
    print("=" * 50)
    print(f"📂 Source: {args.source}")
    print(f"📂 Output: {args.output}")
    print(f"🎯 Heatmap sigma: {args.sigma}")

    # Get all match folders
    match_dirs = [
        os.path.join(args.source, d)
        for d in os.listdir(args.source)
        if os.path.isdir(os.path.join(args.source, d)) and d.startswith('match')
    ]
    match_dirs.sort()

    if not match_dirs:
        print("❌ No match folders found")
        return

    print(f"🏸 Found {len(match_dirs)} match folders")

    # Process each match
    for match_dir in match_dirs:
        try:
            process_match(match_dir, args.output, args.sigma)
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(match_dir)}: {e}")
            continue

    print("\n🏸 Dataset processing completed!")


if __name__ == "__main__":
    main()
