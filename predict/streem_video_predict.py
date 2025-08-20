import argparse
import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
import time
from tqdm import tqdm
import torch

from model.tracknet_v4 import TrackNet
from model.vballnet_v1 import VballNetV1
from model.vballnet_v1a import VballNetV1a
from model.vballnet_v2 import VballNetV2
from model.vballnet_v3 import VballNetV3
from model.vballnet_v1c import VballNetV1c
from model.vballnet_v1d import VballNetV1d
from model.vballnetfast_v1 import VballNetFastV1
from model.vballnetfast_v2 import VballNetFastV2

MODEL_CONFIGS = {
    'VballNetV1c': {
        'class': VballNetV1c,
        'args': {
            'height': 'input_height',
            'width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim',
            'fusion_layer_type': '"TypeA"'
        },
        '_model_type': 'VballNetV1c'
    },
    'VballNetV1d': {
        'class': VballNetV1d,
        'args': {
            'height': 'input_height',
            'width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim'
        },
        '_model_type': 'VballNetV1d'
    },
    'VballNetV2': {
        'class': VballNetV2,
        'args': {
            'height': 'input_height',
            'width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim'
        },
        '_model_type': 'VballNetV2'
    },
    'VballNetV3': {
        'class': VballNetV3,
        'args': {
            'height': 'input_height',
            'width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim'
        },
        '_model_type': 'VballNetV2'
    },
    'VballNetFastV1': {
        'class': VballNetFastV1,
        'args': {
            'input_height': 'input_height',
            'input_width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim'
        },
        '_model_type': 'VballNetFastV1'
    },
    'VballNetFastV2': {
        'class': VballNetFastV2,
        'args': {
            'input_height': 'input_height',
            'input_width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim'
        },
        '_model_type': 'VballNetFastV2'
    },
    'VballNetV1': {
        'class': VballNetV1,
        'args': {
            'height': 'input_height',
            'width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim',
            'fusion_layer_type': '"TypeA"'
        },
        '_model_type': None
    },

    'VballNetV1': {
        'class': VballNetV1a,
        'args': {
            'height': 'input_height',
            'width': 'input_width',
            'in_dim': 'in_dim',
            'out_dim': 'out_dim',
            'fusion_layer_type': '"TypeA"'
        },
        '_model_type': 'VballNetV1a'
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description="Volleyball ball detection and tracking")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--track_length", type=int, default=8, help="Length of the ball track (default: 8 frames)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output video and CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights file (e.g., outputs/exp_20250801_110350/checkpoints/best_model.pth)")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true", default=False, help="Save only CSV, skip video output")
    return parser.parse_args()

def parse_model_params_from_name(model_path):
    basename = os.path.basename(model_path)
    seq = 3
    grayscale = False
    if "seq" in basename:
        import re
        m = re.search(r"seq(\d+)", basename)
        if m:
            seq = int(m.group(1))
    if "grayscale" in basename.lower():
        grayscale = True
    return seq, grayscale

def get_in_out_dim(seq, grayscale):
    if grayscale:
        in_dim = seq
        out_dim = seq
    else:
        in_dim = seq * 3
        out_dim = seq
    return in_dim, out_dim

def load_model(model_path, input_height=288, input_width=512):
    if not os.path.exists(model_path):
        raise ValueError(f"Model weights file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basename = os.path.basename(model_path)
    seq, grayscale = parse_model_params_from_name(model_path)

    model = None
    for key, cfg in MODEL_CONFIGS.items():
        if key in basename:
            in_dim, out_dim = get_in_out_dim(seq, grayscale)
            args = {}
            for arg_name, val in cfg['args'].items():
                if val == 'input_height':
                    args[arg_name] = input_height
                elif val == 'input_width':
                    args[arg_name] = input_width
                elif val == 'in_dim':
                    args[arg_name] = in_dim
                elif val == 'out_dim':
                    args[arg_name] = out_dim
                elif val == '"TypeA"':
                    args[arg_name] = "TypeA"
                else:
                    args[arg_name] = val
            model = cfg['class'](**args).to(device)
            if cfg['_model_type']:
                model._model_type = cfg['_model_type']
            break


    if model is None:
        model = TrackNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    model._seq = seq
    model._grayscale = grayscale
    return model

def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames

def setup_output_writer(video_basename, output_dir, frame_width, frame_height, fps, only_csv):
    if output_dir is None or only_csv:
        return None, None

    output_path = os.path.join(output_dir, f'{video_basename}_predict.mp4')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return out_writer, output_path

def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    csv_path = os.path.join(output_dir, f'{video_basename}_predict_ball.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y']).to_csv(csv_path, index=False)
    return csv_path

def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode='a', header=False, index=False)

def preprocess_frame(frame, input_height=288, input_width=512):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32) / 255.0
    return frame

def preprocess_input(frame_buffer, input_height=288, input_width=512, seq=3, grayscale=False):
    if grayscale:
        gray_frames = [cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for f in frame_buffer]
        input_tensor = np.stack(gray_frames, axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=0)
    else:
        input_tensor = np.concatenate(frame_buffer, axis=2)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
    input_tensor = torch.from_numpy(input_tensor).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    return input_tensor

def postprocess_output(output, threshold=0.55, input_height=288, input_width=512):
    results = []
    seq = output.shape[0]
    for frame_idx in range(seq):
        heatmap = output[frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_contour)
                results.append((1, cx, cy, x, y, w, h))
            else:
                results.append((0, 0, 0, 0, 0, 0, 0))
        else:
            results.append((0, 0, 0, 0, 0, 0, 0))
    return results

def visualize_heatmaps(output, seq=9, input_height=288, input_width=512):
    # Показывает только центральную тепловую карту (для seq=9 — это 5 кадр, индекс 4)
    center_idx = seq // 2
    heatmap = output[center_idx, :, :]
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_uint8 = heatmap_norm.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    cv2.imshow(f'Heatmap Center Frame {center_idx+1}', heatmap_color)
    cv2.waitKey(1)

def draw_track(frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0), center_color=(0,255,0), current_ball_bbox=None):
    points = list(track_points)
    seq = len(points)
    center_idx = seq // 2 if seq > 0 else 0
    for idx, point in enumerate(points):
        if point is None:
            continue
        if idx == center_idx:
            cv2.circle(frame, point, 10, center_color, -1)  # центральный зелёный радиус 6
        elif idx == seq-1:
            cv2.circle(frame, point, 10, current_color, -1) # последний (текущий) — красный
        else:
            cv2.circle(frame, point, 5, history_color, -1)
    return frame

def main():
    args = parse_args()
    input_width, input_height = 512, 288

    model = load_model(args.model_path, input_height, input_width)
    model.eval()
    seq = getattr(model, "_seq", 3)
    grayscale = getattr(model, "_grayscale", False)

    cap, frame_width, frame_height, fps, total_frames = initialize_video(args.video_path)

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_path = setup_csv_file(video_basename, args.output_dir)

    processed_frame_buffer = deque(maxlen=seq)
    frame_buffer = deque(maxlen=seq)
    track_points = deque(maxlen=args.track_length)
    frame_index = 0

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    stop = False
    h0 = None
    use_gru = hasattr(model, '_model_type') and model._model_type == "VballNetV1c"

    print("GRU", use_gru)
    while cap.isOpened() and not stop:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_frame(frame_rgb)
        frame_buffer.append(frame)
        processed_frame_buffer.append(processed_frame)

        if len(processed_frame_buffer) < seq:
            print('len:', len(processed_frame_buffer), seq)
            continue

        if len(processed_frame_buffer) == seq:
            input_tensor = preprocess_input(processed_frame_buffer, input_height, input_width, seq=seq, grayscale=grayscale)
            with torch.no_grad():
                if use_gru:
                    output, hn = model(input_tensor, h0=h0)
                    h0 = hn.detach()
                else:
                    output = model(input_tensor)
                output = output.squeeze(0).cpu().numpy()
            predictions = postprocess_output(output, input_height=input_height, input_width=input_width)
            print('run prediction:', len(predictions), 'frames')

            for idx, pred in enumerate(predictions):
                processed_frame_buffer.popleft()
                frame = frame_buffer.popleft()

                frame_index += 1
                visibility, x, y, bbox_x, bbox_y, bbox_w, bbox_h = pred
                current_ball_bbox = None
                if visibility == 0:
                    x_orig, y_orig = -1, -1
                    if len(track_points) > 0:
                        track_points.popleft()
                else:
                    x_orig = x * frame_width / input_width
                    y_orig = y * frame_height / input_height
                    track_points.append((int(x_orig), int(y_orig)))
                    bbox_x_orig = int(bbox_x * frame_width / input_width)
                    bbox_y_orig = int(bbox_y * frame_height / input_height)
                    bbox_w_orig = int(bbox_w * frame_width / input_width)
                    bbox_h_orig = int(bbox_h * frame_height / input_height)
                    current_ball_bbox = (bbox_x_orig, bbox_y_orig, bbox_w_orig, bbox_h_orig)

                result = {
                    'Frame': frame_index,
                    'Visibility': visibility,
                    'X': int(x_orig),
                    'Y': int(y_orig)
                }
                append_to_csv(result, csv_path)

                if args.visualize or out_writer is not None:
                    vis_frame = frame.copy()
                    vis_frame = draw_track(vis_frame, track_points, current_ball_bbox=current_ball_bbox)
                    if args.visualize:
                        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
                        cv2.imshow('Tracking', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            stop = True
                            break

                    visualize_heatmaps(output, 3)
                    if out_writer is not None:
                        out_writer.write(vis_frame)

        end_time = time.time()
        batch_time = end_time - start_time
        batch_fps = 1 / batch_time if batch_time > 0 else 0
        pbar.update(1)

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

