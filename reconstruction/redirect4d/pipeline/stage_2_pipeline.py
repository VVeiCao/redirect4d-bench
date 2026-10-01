"""
阶段2流程编排：视频生成

流程：
  2_0 Wan2.2 视频生成（自动Caption + 深度控制）

输入：
  - outputs/rendering/{scene}/arc_{type}_{angle}/inference/
    ├── reference_image.png
    ├── rendered_depths.mp4
    └── original_images.mp4

输出：
  - outputs/rendering/{scene}/arc_{type}_{angle}/inference/
    ├── generated_prompt.txt
    └── output_video.mp4
"""

import subprocess
from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger

logger = setup_logger('stage_2_video')


class Stage2VideoPipeline:
    """阶段2：视频生成"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        # 使用场景级别的rendering目录，而不是包含轨迹名称的完整路径
        self.rendering_dir = Path(config.get('project.output_rendering_base'))
    
    def run(self, trajectory_name: Optional[str] = None):
        """
        运行视频生成流程
        
        Args:
            trajectory_name: 轨迹子目录名（如arc_yaw_-120或arc_pitch_-120），
                           默认使用配置的arc_type和arc_angle生成
        """
        logger.info("=" * 60)
        logger.info("阶段2：视频生成")
        logger.info("=" * 60)
        
        # 确定数据目录
        if trajectory_name:
            data_dir = self.rendering_dir / trajectory_name
        else:
            # 从配置读取 arc_type 和 arc_angle 生成目录名
            arc_type = self.config.get('stage_1.rendering.arc_type', 'yaw')
            arc_angle = self.config.get('stage_1.rendering.arc_angle', 90)
            trajectory_name = f"arc_{arc_type}_{int(arc_angle)}"
            data_dir = self.rendering_dir / trajectory_name
        
        logger.info(f"📁 渲染目录: {data_dir}")
        
        # 检查输入文件是否存在
        inference_dir = data_dir / "inference"
        required_files = [
            inference_dir / "reference_image.png",
            inference_dir / "rendered_depths.mp4",
            inference_dir / "original_images.mp4"
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少必需的输入文件:\n" + 
                "\n".join(f"  - {f}" for f in missing) +
                f"\n💡 提示: 请先运行 stage_1b 生成渲染结果"
            )
        
        logger.info("✓ 输入文件检查通过")
        
        # 检查是否已生成过
        output_video = inference_dir / "output_video.mp4"
        if output_video.exists():
            logger.info(f"⚠️  输出视频已存在: {output_video}")
            logger.info(f"   将覆盖重新生成...")
        
        # 调用视频生成脚本
        logger.info("\n🎬 开始视频生成...")
        logger.info(f"   使用 Wan2.2-VACE-Fun-A14B 模型")
        logger.info(f"   自动生成 Caption (Qwen3-VL-2B)")
        
        # 获取参数
        seed = self.config.get('stage_2.seed', 1)
        num_inference_steps = self.config.get('stage_2.num_inference_steps', 50)
        sigma_shift = self.config.get('stage_2.sigma_shift', 16.0)
        cfg_scale = self.config.get('stage_2.cfg_scale', 5.0)
        
        logger.info(f"   - Seed: {seed}")
        logger.info(f"   - 推理步数: {num_inference_steps}")
        logger.info(f"   - Sigma shift: {sigma_shift}")
        logger.info(f"   - CFG scale: {cfg_scale}")
        
        # 构建命令
        cmd = [
            'python', 'scripts/2_0_Wan2.2-VACE-Fun-A14B.py',
            '--data_dir', str(data_dir),
            '--seed', str(seed),
            '--num_inference_steps', str(num_inference_steps),
            '--sigma_shift', str(sigma_shift),
            '--cfg_scale', str(cfg_scale)
        ]
        
        # 执行命令
        result = subprocess.run(cmd, check=True)
        
        if result.returncode == 0:
            logger.info("\n" + "=" * 60)
            logger.info("🎉 视频生成完成！")
            logger.info("=" * 60)
            logger.info(f"📹 输出视频: {output_video}")
            logger.info(f"📝 生成的Caption: {inference_dir / 'generated_prompt.txt'}")
        else:
            raise RuntimeError("视频生成失败")


if __name__ == '__main__':
    print("🧪 测试阶段2 Pipeline...")
    
    from utils.config import Config
    
    config = Config('configs/scenes/camel.yaml')
    pipeline = Stage2VideoPipeline(config)
    
    print(f"\n✅ Pipeline创建成功")
    print(f"   - 渲染目录: {pipeline.rendering_dir}")
    print(f"   - 默认轨迹: arc_{int(config.get('stage_1.rendering.arc_angle'))}")
    print(f"\n使用方法:")
    print(f"   pipeline.run()  # 使用默认arc_angle")
    print(f"   pipeline.run(trajectory_name='arc_90')  # 指定轨迹")


