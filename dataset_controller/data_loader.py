from pathlib import Path

from dataset_controller.ball_tracking_data_reader import BallTrackingDataset
from visualize_dataset_sample import play_dataset


def load_all_matches(professional_dir, config):
    """
    加载professional文件夹中所有match文件夹并拼接
    """
    professional_dir = Path(professional_dir)

    # 获取所有match文件夹并排序
    match_dirs = sorted([d for d in professional_dir.iterdir()
                         if d.is_dir() and d.name.startswith('match')])

    if not match_dirs:
        print("未找到match文件夹")
        return None

    print(f"找到 {len(match_dirs)} 个match文件夹: {[d.name for d in match_dirs]}")

    # 加载第一个数据集
    combined_dataset = None

    for match_dir in match_dirs:
        try:
            dataset = BallTrackingDataset(str(match_dir), config=config)
            if len(dataset) > 0:
                if combined_dataset is None:
                    combined_dataset = dataset
                else:
                    combined_dataset = combined_dataset + dataset  # 使用+操作符
                print(f"已添加 {match_dir.name}: {len(dataset)} 个样本")
            else:
                print(f"跳过空数据集: {match_dir.name}")
        except Exception as e:
            print(f"加载 {match_dir.name} 时出错: {e}")

    if combined_dataset:
        print(f"拼接完成，总共 {len(combined_dataset)} 个样本")

    return combined_dataset


# Usage
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    professional_dir = base_dir / 'dataset' / 'Professional'

    config_3in3out = {
        "input_frames": 3,
        "output_frames": 3,
        "normalize_coords": False,
        "normalize_pixels": False,
        "video_ext": ".mp4",
        "csv_suffix": "_ball.csv"
    }

    # 一行解决所有拼接
    dataset = load_all_matches(professional_dir, config_3in3out)

    if dataset:
        frames, labels = dataset[0]
        print(f"第一个样本 - Frames shape: {frames.shape}, Labels count: {len(labels)}")
        print(f"总样本数: {len(dataset)}")

        # print(dataset[0])  # 打印第一个样本的详细信息
        play_dataset(dataset, delay_ms=10)
