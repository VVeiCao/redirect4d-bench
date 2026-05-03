"""
数据预处理：交互式 Mask 标注（SAM2 + Gradio）

功能：
    通过 Gradio Web UI 交互式地在视频/图片序列上标注前景 mask。
    使用 SAM2 视频分割模型，用户只需在某一帧上点击前景/背景点，
    即可自动传播到所有帧，生成完整的 mask 序列。

输出文件（对齐 DAVIS 格式，可直接接入后续 pipeline）：
    data/user/images/{scene_name}/
    ├── 00000.jpg
    ├── 00001.jpg
    └── ...
    data/user/masks/{scene_name}/
    ├── 00000.png          # 灰度 PNG, 0(背景)/255(前景)
    ├── 00001.png
    └── ...
    configs/user/{scene_name}.yaml   # 自动生成的配置文件

环境依赖：
    pip install git+https://github.com/facebookresearch/sam2.git
    pip install gradio loguru

SAM2 Checkpoint：
    默认使用: checkpoints/sam2/sam2_hiera_large.pt
    自动下载: bash download_checkpoints.sh sam2
    手动下载: wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

使用示例：
    python scripts/prep_interactive_mask.py
    python scripts/prep_interactive_mask.py --port 8890
    python scripts/prep_interactive_mask.py --checkpoint_dir /path/to/sam2_hiera_large.pt
"""

import torch

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import os
import sys
import subprocess
import shutil
import time
import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import imageio.v2 as iio
import numpy as np
from PIL import Image
from loguru import logger as guru

from sam2.build_sam import build_sam2_video_predictor

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 不兼容的视频编码
INCOMPATIBLE_CODECS = ['av1', 'av01', 'hevc', 'h265', 'vp9', 'vp8']


def get_video_codec(video_path):
    """获取视频编码格式"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip().lower()
    except Exception as e:
        guru.warning(f"获取视频编码失败: {e}")
        return None


def check_and_convert_video(video_path):
    """检查视频编码，必要时转换为 H.264"""
    if not video_path or not os.path.exists(video_path):
        return video_path

    codec = get_video_codec(video_path)
    guru.info(f"视频编码: {codec}")

    if codec and codec in INCOMPATIBLE_CODECS:
        guru.info(f"转换 {codec} 为 H.264 格式...")
        tmp_dir = tempfile.mkdtemp(prefix="freeorbit4d_")
        output_path = os.path.join(tmp_dir, f"converted_{int(time.time())}.mp4")

        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-pix_fmt', 'yuv420p', '-an',
            output_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
                cap.release()
                guru.info(f"视频转换成功: {output_path}")
                return output_path
            cap.release()
        except Exception as e:
            guru.warning(f"视频转换失败: {e}")

    return video_path


def isimage(p):
    return os.path.splitext(p.lower())[-1] in [".png", ".jpg", ".jpeg"]


def draw_points(img, points, labels):
    """在图像上绘制选中的点（绿=正, 红=负）"""
    out = img.copy()
    for p, label in zip(points, labels):
        x, y = int(p[0]), int(p[1])
        color = (0, 255, 0) if label == 1.0 else (255, 0, 0)
        out = cv2.circle(out, (x, y), 10, color, -1)
    return out


def compose_img_mask(img, color_mask, fac=0.5):
    """将 mask 叠加到图像上"""
    out_f = fac * img / 255 + (1 - fac) * color_mask / 255
    return (255 * out_f).astype("uint8")


class MaskAnnotator:
    """SAM2 交互式 Mask 标注器"""

    def __init__(self, checkpoint_dir, model_cfg):
        self.checkpoint_dir = checkpoint_dir
        self.model_cfg = model_cfg
        self.sam_model = None

        self.selected_points = []
        self.selected_labels = []
        self.cur_label_val = 1.0

        self.frame_index = 0
        self.image = None
        self.cur_mask = None
        self.cur_logit = None
        self.masks_all = []

        self.img_dir = ""
        self.img_paths = []
        self.video_name = None
        self.inference_state = None
        self._temp_dirs = []  # 记录所有临时目录，保存后统一清理

        self._init_sam_model()

    def _init_sam_model(self):
        if self.sam_model is not None:
            return
        if not os.path.exists(self.checkpoint_dir):
            error_msg = (
                f"SAM2 checkpoint 不存在: {self.checkpoint_dir}\n\n"
                "下载地址:\n"
                "  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt\n"
                f"请将文件放到: {self.checkpoint_dir}"
            )
            guru.error(error_msg)
            raise FileNotFoundError(error_msg)
        self.sam_model = build_sam2_video_predictor(self.model_cfg, self.checkpoint_dir)
        guru.info(f"SAM2 模型已加载: {self.checkpoint_dir}")

    def extract_scene_name(self, path):
        """从路径提取场景名"""
        if not path:
            return "sequence"
        name = Path(path).stem if Path(path).is_file() else Path(path).name
        if not name or name in ['.', '..']:
            name = Path(path).parent.name
        import re
        clean = re.sub(r'[^\w\-_]', '_', name)
        return clean if clean else "sequence"

    def clear_points(self):
        self.selected_points.clear()
        self.selected_labels.clear()
        return None, None, None, "已清除所有点，请重新选点"

    def _clear_image(self):
        self.image = None
        self.frame_index = 0
        self.cur_mask = None
        self.cur_logit = None
        self.masks_all = []

    def reset(self):
        self._clear_image()
        if self.inference_state is not None:
            self.sam_model.reset_state(self.inference_state)

    def set_img_dir(self, img_dir):
        self._clear_image()
        self.img_dir = img_dir
        if not os.path.exists(img_dir):
            guru.error(f"目录不存在: {img_dir}")
            return 0
        self.img_paths = [
            os.path.abspath(os.path.join(img_dir, p))
            for p in sorted(os.listdir(img_dir)) if isimage(p)
        ]
        guru.info(f"找到 {len(self.img_paths)} 张图片")
        self.video_name = self.extract_scene_name(img_dir)
        return len(self.img_paths)

    def set_input_image(self, i=0):
        if i < 0 or i >= len(self.img_paths):
            return self.image
        self.clear_points()
        self.frame_index = i
        self.image = iio.imread(self.img_paths[i])
        return self.image

    def get_sam_features(self):
        try:
            self.inference_state = self.sam_model.init_state(video_path=self.img_dir)
            self.sam_model.reset_state(self.inference_state)
            guru.info("SAM 特征提取完成")
            return "SAM 特征提取完成，请在图片上点击选点，然后提交开始追踪", self.image
        except Exception as e:
            error_msg = f"SAM 特征提取失败: {e}"
            guru.error(error_msg)
            return error_msg, self.image

    def set_positive(self):
        self.cur_label_val = 1.0
        return "正在选择正向点（前景）"

    def set_negative(self):
        self.cur_label_val = 0.0
        return "正在选择负向点（背景）"

    def add_point(self, frame_idx, i, j):
        self.selected_points.append([j, i])
        self.selected_labels.append(self.cur_label_val)
        mask, logit = self._get_sam_mask(
            frame_idx,
            np.array(self.selected_points, dtype=np.float32),
            np.array(self.selected_labels, dtype=np.int32),
        )
        self.cur_mask = mask
        self.cur_logit = logit
        return mask

    def _get_sam_mask(self, frame_idx, input_points, input_labels):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=0,
                points=input_points,
                labels=input_labels,
            )
        mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
        logit = out_mask_logits[0].squeeze().cpu().numpy()
        return mask, logit

    def run_tracker(self):
        """传播 mask 到所有帧"""
        images = [iio.imread(p)[:, :, :3] for p in self.img_paths]
        self.masks_all = []

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(
                self.inference_state, start_frame_idx=0
            ):
                mask = (out_mask_logits[0] > 0.0).squeeze().cpu().numpy()
                self.masks_all.append(mask)

        # 生成可视化视频
        out_frames = []
        for img, mask in zip(images, self.masks_all):
            colored_mask = np.zeros_like(img)
            colored_mask[mask] = [0, 255, 0]
            out_frames.append(compose_img_mask(img, colored_mask, 0.5))

        tmp_dir = tempfile.mkdtemp(prefix="freeorbit4d_")
        self._temp_dirs.append(tmp_dir)
        out_vidpath = os.path.join(tmp_dir, "tracked_masks.mp4")
        iio.mimwrite(out_vidpath, out_frames)

        msg = f"追踪完成！共 {len(self.masks_all)} 帧。满意的话请输入场景名并保存。"
        return out_vidpath, msg

    def save_to_data_dir(self, scene_name):
        """保存为 data/user/ 格式，对齐 DAVIS 目录结构"""
        if not self.masks_all or len(self.masks_all) == 0:
            return "请先完成 mask 追踪"
        if not scene_name or not scene_name.strip():
            return "请输入场景名"

        scene_name = scene_name.strip()
        images_dir = os.path.join(PROJECT_ROOT, "data", "user", "images", scene_name)
        masks_dir = os.path.join(PROJECT_ROOT, "data", "user", "masks", scene_name)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        for i, (img_path, mask) in enumerate(zip(self.img_paths, self.masks_all)):
            # 保存图像 (JPEG)
            img = Image.open(img_path).convert("RGB")
            img.save(os.path.join(images_dir, f"{i:05d}.jpg"))

            # 保存 mask (灰度 PNG, 0/255)
            mask_uint8 = (mask.astype(np.uint8) * 255)
            Image.fromarray(mask_uint8, mode='L').save(
                os.path.join(masks_dir, f"{i:05d}.png")
            )

        # 生成 config
        config_dir = os.path.join(PROJECT_ROOT, "configs", "user")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"{scene_name}.yaml")

        config_content = f"""_base_: ../default.yaml

project:
  name: {scene_name}
  input_images: data/user/images/{scene_name}
  input_masks: data/user/masks/{scene_name}
  output_root: outputs/user
"""
        with open(config_path, 'w') as f:
            f.write(config_content)

        # 保存选点记录
        meta = {
            "scene_name": scene_name,
            "num_frames": len(self.masks_all),
            "selected_points": self.selected_points,
            "selected_labels": self.selected_labels,
            "frame_index": self.frame_index,
        }
        meta_path = os.path.join(masks_dir, "annotation_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        msg = (
            f"保存完成！\n"
            f"  图像: {images_dir} ({len(self.masks_all)} 帧)\n"
            f"  Mask: {masks_dir}\n"
            f"  配置: {config_path}\n\n"
            f"后续使用:\n"
            f"  python scripts/0_0_gen_multiviews.py --config configs/user/{scene_name}.yaml"
        )
        # 清理所有临时目录
        for d in self._temp_dirs:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
                guru.info(f"已清理临时目录: {d}")
        self._temp_dirs.clear()
        # 如果 img_dir 是系统临时目录（从视频提取的帧），也清理掉
        if self.img_dir and self.img_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.img_dir, ignore_errors=True)
            guru.info(f"已清理临时帧目录: {self.img_dir}")

        guru.info(msg)
        return msg


def resize_and_crop_frame(frame, target_width, target_height):
    """缩放并居中裁剪帧"""
    h, w = frame.shape[:2]
    if w == target_width and h == target_height:
        return frame
    scale = max(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h))
    start_x = (new_w - target_width) // 2
    start_y = (new_h - target_height) // 2
    return frame[start_y:start_y + target_height, start_x:start_x + target_width]


def make_demo(checkpoint_dir, model_cfg):
    annotator = MaskAnnotator(checkpoint_dir, model_cfg)

    with gr.Blocks(title="FreeOrbit4D - Interactive Mask Annotation") as demo:
        gr.Markdown("# FreeOrbit4D - 交互式 Mask 标注 (SAM2)")
        instruction = gr.Textbox(
            "上传视频或选择图片目录，然后在图片上点击标注前景物体。",
            label="说明", interactive=False,
        )

        # ===== 输入 =====
        with gr.Tab("上传视频"):
            input_video_field = gr.File(
                label="上传视频文件",
                file_types=[".mp4", ".avi", ".mov", ".mkv"],
            )
            with gr.Row():
                video_target_width = gr.Number(0, label="目标宽度 (0=原始)")
                video_target_height = gr.Number(0, label="目标高度 (0=原始)")
            load_video_button = gr.Button("Step 1: 加载视频")

            # Step 2: 选择起始帧
            with gr.Group(visible=False) as frame_select_group:
                gr.Markdown("### 选择起始帧与采样参数")
                with gr.Row():
                    video_stride = gr.Number(1, label="采样步长", minimum=1, step=1)
                    video_num_frames = gr.Number(45, label="帧数 (精确)", minimum=1, step=1)
                preview_slider = gr.Slider(label="起始帧 (拖动选择并预览)", minimum=0, maximum=0, value=0, step=1)
                preview_image = gr.Image(label="帧预览")
                video_info = gr.Textbox(label="信息", interactive=False)
                extract_button = gr.Button("Step 2: 确认并提取", variant="primary")

        with gr.Tab("选择图片目录"):
            with gr.Row():
                img_dir_field = gr.Text(None, label="图片目录路径", placeholder="输入包含图片帧的目录")
                load_dir_button = gr.Button("加载目录")

        # ===== 标注 =====
        frame_index = gr.Slider(label="帧索引", minimum=0, maximum=0, value=0, step=1)

        with gr.Row():
            with gr.Column():
                reset_button = gr.Button("重置")
                input_image = gr.Image(None, label="输入帧")
                with gr.Row():
                    pos_button = gr.Button("正向点 (前景)")
                    neg_button = gr.Button("负向点 (背景)")
                clear_button = gr.Button("清除选点")

            with gr.Column():
                output_img = gr.Image(label="当前选择")
                submit_button = gr.Button("提交 Mask 并追踪")
                final_video = gr.Video(label="Mask 追踪结果")

        # ===== 保存 =====
        with gr.Row():
            scene_name_field = gr.Text(
                "my_scene", label="场景名 (scene_name)",
                info="将保存到 data/user/images/{scene_name}/ 和 data/user/masks/{scene_name}/",
            )
            save_button = gr.Button("保存到 data/user/", variant="primary")

        # ===== 状态 =====
        # 存储预加载的所有帧（load_video 后填充）
        _all_frames_dir = gr.State(None)    # 临时目录路径（所有帧 JPG）
        _all_frames_total = gr.State(0)     # 总帧数

        # ===== 事件绑定 =====

        def load_image_directory(img_dir):
            if not img_dir or not os.path.isdir(img_dir):
                return gr.Slider(), "请输入有效的目录路径", None, ""
            num_imgs = annotator.set_img_dir(img_dir)
            if num_imgs == 0:
                return gr.Slider(), "目录中没有找到图片文件", None, ""
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
            first_image = annotator.set_input_image(0)
            sam_msg, sam_img = annotator.get_sam_features()
            scene = annotator.extract_scene_name(img_dir)
            return slider, sam_msg, sam_img if sam_img is not None else first_image, scene

        def load_video(video_file, target_width, target_height):
            """Step 1: 提取视频所有帧到临时目录，显示预览 slider"""
            if video_file is None:
                return (
                    gr.Group(visible=False),  # frame_select_group
                    gr.Slider(),              # preview_slider
                    None,                     # preview_image
                    "请先上传视频",            # video_info
                    None,                     # _all_frames_dir
                    0,                        # _all_frames_total
                )

            converted = check_and_convert_video(video_file)
            cap = cv2.VideoCapture(converted)
            if not cap.isOpened():
                return (
                    gr.Group(visible=False), gr.Slider(), None,
                    "无法打开视频文件", None, 0,
                )

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            target_w = int(target_width) if int(target_width) > 0 else orig_w
            target_h = int(target_height) if int(target_height) > 0 else orig_h

            guru.info(f"加载视频: {total} 帧, {orig_w}x{orig_h} -> {target_w}x{target_h}")

            temp_dir = tempfile.mkdtemp(prefix="freeorbit4d_allframes_")
            count = 0
            for idx in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                if orig_w != target_w or orig_h != target_h:
                    frame = resize_and_crop_frame(frame, target_w, target_h)
                cv2.imwrite(os.path.join(temp_dir, f"{idx:05d}.jpg"), frame)
                count += 1
            cap.release()

            guru.info(f"已提取 {count} 帧到 {temp_dir}")

            # 读取第一帧作为预览
            first_frame_path = os.path.join(temp_dir, "00000.jpg")
            first_img = iio.imread(first_frame_path) if os.path.exists(first_frame_path) else None

            # 默认 stride=1, num_frames=45 → 最后一帧 = start + 44 < count → start <= count-45
            default_nframes = 45
            default_stride = 1
            slider_max = max(0, count - (default_nframes - 1) * default_stride - 1)
            info_msg = (
                f"视频已加载: {count} 帧, {target_w}x{target_h}。\n"
                f"拖动滑块选择起始帧 (范围 0~{slider_max})，选好后点击「确认并提取」"
            )

            return (
                gr.Group(visible=True),                                        # frame_select_group
                gr.Slider(minimum=0, maximum=slider_max, value=0, step=1),     # preview_slider
                first_img,                                                      # preview_image
                info_msg,                                                       # video_info
                temp_dir,                                                       # _all_frames_dir
                count,                                                          # _all_frames_total
            )

        def preview_frame(slider_val, frames_dir):
            """滑块拖动时显示对应帧"""
            if not frames_dir or not os.path.isdir(frames_dir):
                return None
            frame_path = os.path.join(frames_dir, f"{int(slider_val):05d}.jpg")
            if os.path.exists(frame_path):
                return iio.imread(frame_path)
            return None

        def update_slider_range(stride, num_frames, total_frames):
            """当步长或帧数改变时，更新起始帧 slider 的范围"""
            total = int(total_frames)
            if total <= 0:
                return gr.Slider(), ""
            stride_val = max(1, int(stride))
            nframes = max(1, int(num_frames))
            # 需要的最后一帧索引: start + (nframes-1)*stride < total
            # → start < total - (nframes-1)*stride
            slider_max = max(0, total - (nframes - 1) * stride_val - 1)
            info_msg = f"可选起始帧范围: 0 ~ {slider_max} (共 {total} 帧, 步长 {stride_val}, 取 {nframes} 帧)"
            return gr.Slider(minimum=0, maximum=slider_max, step=1), info_msg

        def extract_frames(video_file, start_frame, stride, num_frames, frames_dir, total_frames):
            """Step 2: 从预加载帧中按 start/stride/num 选取子集，送入 SAM2"""
            if not frames_dir or not os.path.isdir(frames_dir):
                return "请先点击「加载视频」", gr.Slider(), None, ""

            start = max(0, int(start_frame))
            stride_val = max(1, int(stride))
            nframes = int(num_frames)
            total = int(total_frames)

            # 计算要选取的帧索引
            selected_indices = list(range(start, total, stride_val))

            if len(selected_indices) < nframes:
                return (
                    f"帧数不足！从第 {start} 帧开始、步长 {stride_val}，"
                    f"只能取 {len(selected_indices)} 帧，但需要 {nframes} 帧。"
                    f"请调整参数。(视频共 {total} 帧)",
                    gr.Slider(), None, ""
                )

            selected_indices = selected_indices[:nframes]

            # 复制选中帧到新临时目录（重新编号 00000, 00001, ...）
            temp_dir = tempfile.mkdtemp(prefix="freeorbit4d_frames_")
            for i, idx in enumerate(selected_indices):
                src = os.path.join(frames_dir, f"{idx:05d}.jpg")
                dst = os.path.join(temp_dir, f"{i:05d}.jpg")
                shutil.copy2(src, dst)

            num_imgs = annotator.set_img_dir(os.path.abspath(temp_dir))
            slider = gr.Slider(minimum=0, maximum=num_imgs - 1, value=0, step=1)
            first_image = annotator.set_input_image(0)
            sam_msg, sam_img = annotator.get_sam_features()

            scene = annotator.extract_scene_name(video_file)
            msg = (
                f"提取 {nframes} 帧 (起始帧 {start}, 步长 {stride_val})。{sam_msg}"
            )
            return msg, slider, sam_img if sam_img is not None else first_image, scene

        def get_select_coords(frame_idx, img, evt: gr.SelectData):
            if img is None:
                return None
            i = evt.index[1]
            j = evt.index[0]
            binary_mask = annotator.add_point(frame_idx, i, j)
            colored_mask = np.zeros_like(img)
            colored_mask[binary_mask] = [0, 255, 0]
            out = compose_img_mask(img, colored_mask, 0.5)
            out = draw_points(out, annotator.selected_points, annotator.selected_labels)
            return out

        def run_tracker_with_message():
            vid, msg = annotator.run_tracker()
            return vid, msg

        def save_data(scene_name):
            return annotator.save_to_data_dir(scene_name)

        # ===== 绑定: 视频 Tab =====
        load_video_button.click(
            load_video,
            [input_video_field, video_target_width, video_target_height],
            [frame_select_group, preview_slider, preview_image, video_info,
             _all_frames_dir, _all_frames_total],
        )
        preview_slider.change(
            preview_frame,
            [preview_slider, _all_frames_dir],
            [preview_image],
        )
        # 步长/帧数改变时更新 slider 范围
        video_stride.change(
            update_slider_range,
            [video_stride, video_num_frames, _all_frames_total],
            [preview_slider, video_info],
        )
        video_num_frames.change(
            update_slider_range,
            [video_stride, video_num_frames, _all_frames_total],
            [preview_slider, video_info],
        )
        # 确认并提取：用 preview_slider 的值作为起始帧
        extract_button.click(
            extract_frames,
            [input_video_field, preview_slider, video_stride, video_num_frames,
             _all_frames_dir, _all_frames_total],
            [instruction, frame_index, input_image, scene_name_field],
        )

        # ===== 绑定: 目录 Tab =====
        load_dir_button.click(
            load_image_directory,
            [img_dir_field],
            [frame_index, instruction, input_image, scene_name_field],
        )

        # ===== 绑定: 标注 =====
        frame_index.change(annotator.set_input_image, [frame_index], [input_image])
        input_image.select(get_select_coords, [frame_index, input_image], [output_img])

        reset_button.click(annotator.reset)
        clear_button.click(annotator.clear_points, outputs=[output_img, final_video, instruction, instruction])
        pos_button.click(annotator.set_positive, outputs=[instruction])
        neg_button.click(annotator.set_negative, outputs=[instruction])
        submit_button.click(run_tracker_with_message, outputs=[final_video, instruction])
        save_button.click(save_data, [scene_name_field], [instruction])

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FreeOrbit4D - 交互式 Mask 标注 (SAM2)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--checkpoint_dir", type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "sam2", "sam2_hiera_large.pt"),
    )
    parser.add_argument("--model_cfg", type=str, default="sam2_hiera_l.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        print(f"\nSAM2 checkpoint 不存在: {args.checkpoint_dir}")
        print("\n下载:")
        print("  wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt")
        print(f"\n放到: {args.checkpoint_dir}")
        exit(1)

    demo = make_demo(args.checkpoint_dir, args.model_cfg)
    demo.launch(server_name="127.0.0.1", server_port=args.port or None)
