import os
import shutil
import cv2
import glob
import argparse
import sys
from pathlib import Path

# 定义需要过滤的无效文件和文件夹
INVALID_FILES = {'.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'}
INVALID_FOLDERS = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules'}


def is_valid_item(item_name):
    """检查文件或文件夹名是否有效"""
    if item_name.startswith('.') and item_name not in {'.', '..'}:
        return False
    if item_name in INVALID_FILES or item_name in INVALID_FOLDERS:
        return False
    return True


def validate_source_structure(source_folder):
    """验证源文件夹结构"""
    print(f"🔍 检查源文件夹: {source_folder}")

    if not os.path.exists(source_folder):
        return False, f"源文件夹不存在: {source_folder}"

    # 查找match文件夹
    all_items = [item for item in os.listdir(source_folder) if is_valid_item(item)]
    match_folders = [item for item in all_items
                     if item.startswith("match") and
                     os.path.isdir(os.path.join(source_folder, item))]

    if not match_folders:
        return False, "未找到match文件夹"

    # 验证每个match文件夹的结构
    valid_matches = 0
    total_videos = 0
    total_csvs = 0

    for match_folder in match_folders:
        match_path = os.path.join(source_folder, match_folder)
        csv_path = os.path.join(match_path, "csv")
        video_path = os.path.join(match_path, "video")

        has_csv = os.path.exists(csv_path)
        has_video = os.path.exists(video_path)

        if has_csv and has_video:
            valid_matches += 1

            if has_csv:
                csv_files = [f for f in os.listdir(csv_path)
                             if f.endswith('_ball.csv') and is_valid_item(f)]
                total_csvs += len(csv_files)

            if has_video:
                video_files = [f for f in os.listdir(video_path)
                               if f.endswith('.mp4') and is_valid_item(f)]
                total_videos += len(video_files)

    if valid_matches == 0:
        return False, "无有效match文件夹（需同时包含csv和video）"

    summary = f"✅ 找到 {valid_matches} 个match文件夹，{total_videos} 个视频，{total_csvs} 个CSV"
    return True, summary


def extract_frames_from_video(video_path, output_folder):
    """从视频提取帧"""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ❌ 无法打开: {os.path.basename(video_path)}")
        return False

    frame_count = 0
    # JPG质量设置，范围0-100，95为高质量
    jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 帧编号从0开始
        frame_filename = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame, jpg_quality)
        frame_count += 1

    cap.release()
    print(f"    ✅ {os.path.basename(video_path)} -> {frame_count} 帧")
    return True


def process_match_folder(match_folder_path, output_match_folder, current_idx, total_count):
    """处理单个match文件夹"""
    match_name = os.path.basename(match_folder_path)
    print(f"\n📁 [{current_idx}/{total_count}] 处理 {match_name}")

    # 创建输出文件夹结构
    inputs_folder = os.path.join(output_match_folder, "inputs")
    labels_folder = os.path.join(output_match_folder, "labels")
    os.makedirs(inputs_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    # 处理video文件夹
    video_folder = os.path.join(match_folder_path, "video")
    video_count = 0
    if os.path.exists(video_folder):
        mp4_files = [f for f in os.listdir(video_folder)
                     if f.endswith('.mp4') and is_valid_item(f)]

        if mp4_files:
            print(f"  🎬 转换 {len(mp4_files)} 个视频:")
            for mp4_file in mp4_files:
                video_path = os.path.join(video_folder, mp4_file)
                video_name = Path(mp4_file).stem
                video_output_folder = os.path.join(inputs_folder, video_name)

                if extract_frames_from_video(video_path, video_output_folder):
                    video_count += 1

    # 处理csv文件夹
    csv_count = 0
    csv_folder = os.path.join(match_folder_path, "csv")
    if os.path.exists(csv_folder):
        csv_files = [f for f in os.listdir(csv_folder)
                     if f.endswith('_ball.csv') and is_valid_item(f)]

        if csv_files:
            print(f"  📄 复制 {len(csv_files)} 个CSV:")
            for csv_file in csv_files:
                csv_path = os.path.join(csv_folder, csv_file)
                original_name = Path(csv_file).stem
                new_name = original_name.replace("_ball", "") + ".csv"

                destination_path = os.path.join(labels_folder, new_name)
                shutil.copy2(csv_path, destination_path)
                print(f"    ✅ {csv_file} -> {new_name}")
                csv_count += 1

    print(f"  ✅ 完成: {video_count} 视频, {csv_count} CSV")


def reorganize_dataset(source_folder, force=False):
    """重新组织整个数据集"""
    # 验证源文件夹结构
    is_valid, message = validate_source_structure(source_folder)
    if not is_valid:
        print(f"❌ {message}")
        return False

    print(message)

    # 自动生成输出文件夹名：源文件夹名 + _reorg
    output_folder = f"{source_folder}_reorg"

    # 处理已存在的目标文件夹
    if os.path.exists(output_folder):
        if force:
            print(f"🗑️  删除已存在文件夹: {output_folder}")
            shutil.rmtree(output_folder)
        else:
            response = input(f"⚠️  目标文件夹已存在: {output_folder}\n   是否删除重建? (y/n): ")
            if response.lower() != 'y':
                print("❌ 操作取消")
                return False
            shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)
    print(f"📂 创建输出文件夹: {output_folder}")

    # 查找所有有效的match文件夹
    all_items = [item for item in os.listdir(source_folder) if is_valid_item(item)]
    match_folders = [item for item in all_items
                     if item.startswith("match") and
                     os.path.isdir(os.path.join(source_folder, item))]

    # 过滤有效的match文件夹
    valid_matches = []
    for match_folder_name in match_folders:
        source_match_path = os.path.join(source_folder, match_folder_name)
        csv_exists = os.path.exists(os.path.join(source_match_path, "csv"))
        video_exists = os.path.exists(os.path.join(source_match_path, "video"))

        if csv_exists and video_exists:
            valid_matches.append(match_folder_name)
        else:
            print(f"⚠️  跳过 {match_folder_name}: 缺少csv或video文件夹")

    if not valid_matches:
        print("❌ 没有找到有效的match文件夹")
        return False

    print(f"🚀 开始处理 {len(valid_matches)} 个match文件夹...")

    # 处理每个match文件夹
    for idx, match_folder_name in enumerate(valid_matches, 1):
        source_match_path = os.path.join(source_folder, match_folder_name)
        target_match_path = os.path.join(output_folder, match_folder_name)

        os.makedirs(target_match_path, exist_ok=True)
        process_match_folder(source_match_path, target_match_path, idx, len(valid_matches))

    print(f"\n🎉 重组完成!")
    print(f"   源文件夹: {source_folder}")
    print(f"   输出文件夹: {output_folder}")
    return True


def main():
    """主函数 - 处理命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频数据集重组工具 - 将match文件夹结构转换为标准的inputs/labels格式，视频帧保存为JPG(0.jpg开始编号)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--source", "-s",
                        required=True,
                        help="源文件夹路径（包含match1, match2等子文件夹）")

    parser.add_argument("--force", "-f",
                        action="store_true",
                        help="强制覆盖已存在的输出文件夹（自动生成为源文件夹名_reorg）")

    parser.add_argument("--check-only",
                        action="store_true",
                        help="仅检查源文件夹结构，不执行转换")

    args = parser.parse_args()

    print("🎬 视频数据集重组工具")
    print("=" * 50)

    # 检查OpenCV
    try:
        import cv2
        print(f"📦 OpenCV {cv2.__version__}")
    except ImportError:
        print("❌ 未安装OpenCV: pip install opencv-python")
        sys.exit(1)

    # 仅检查结构
    if args.check_only:
        is_valid, message = validate_source_structure(args.source)
        print(message)
        sys.exit(0 if is_valid else 1)

    # 执行重组
    success = reorganize_dataset(args.source, args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

"""
使用方法（视频帧保存为JPG格式，从0.jpg开始编号）：

基本用法：
  python dataset_reorg.py --source dataset          # 处理dataset文件夹，自动输出到dataset_reorg
  python dataset_reorg.py -s /path/to/data          # 处理指定路径，输出到/path/to/data_reorg

强制覆盖：
  python dataset_reorg.py -s dataset --force        # 强制覆盖已存在的dataset_reorg文件夹

仅检查结构：
  python dataset_reorg.py -s dataset --check-only   # 只验证文件夹结构，不执行转换

安装依赖：
  pip install opencv-python

输入结构：
  dataset/
  ├── match1/
  │   ├── csv/
  │   │   └── video1_ball.csv
  │   └── video/
  │       └── video1.mp4
  └── match2/
      ├── csv/
      └── video/

输出结构（自动生成dataset_reorg）：
  dataset_reorg/
  ├── match1/
  │   ├── inputs/
  │   │   └── video1/
  │   │       ├── 0.jpg
  │   │       ├── 1.jpg
  │   │       └── ...
  │   └── labels/
  │       └── video1.csv
  └── match2/
      ├── inputs/
      └── labels/

功能特点：
- 自动生成输出文件夹（源文件夹名_reorg）
- 自动过滤系统文件(.DS_Store等)
- 实时显示处理进度
- 视频转换为JPG帧(0.jpg开始编号，95%质量)
- CSV文件移除_ball后缀
- 源文件夹保持不变
"""
