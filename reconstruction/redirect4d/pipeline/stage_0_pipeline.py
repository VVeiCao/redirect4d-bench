"""
阶段0流程编排：数据准备

流程：
  0_0 多视角生成 → 0_1 数据预处理

输入：
  - 原始图像目录
  - Mask目录

输出：
  - outputs/multiview/    (中间产物)
  - outputs/prepared/     (最终输出)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from core.data_processor import DataProcessor

logger = setup_logger('stage_0_pipeline')


class Stage0Pipeline:
    """阶段0完整流程：数据准备"""
    
    STEPS = ['multiview', 'preparation']
    
    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 路径配置
        self.input_images = Path(config.get('project.input_images'))
        self.input_masks = Path(config.get('project.input_masks'))
        self.output_multiview = Path(config.get('project.output_multiview'))
        self.output_prepared = Path(config.get('project.output_prepared'))
    
    def run(self, resume_from: Optional[str] = None):
        """
        运行完整流程
        
        Args:
            resume_from: 从哪个步骤继续（None则从头开始）
        """
        start_idx = self._get_start_index(resume_from)
        
        # 执行步骤
        for i, step_name in enumerate(self.STEPS[start_idx:], start_idx + 1):
            logger.info("=" * 60)
            logger.info(f"步骤 {i}/{len(self.STEPS)}: {step_name}")
            logger.info("=" * 60)
            
            try:
                # 调用对应的步骤方法
                step_method = getattr(self, f'_run_{step_name}')
                step_method()
                
                logger.info(f"✅ {step_name} 完成\n")
                
            except Exception as e:
                logger.error(f"❌ {step_name} 失败: {e}")
                raise
        
        logger.info("=" * 60)
        logger.info("🎉 阶段0完整流程完成！")
        logger.info("=" * 60)
        logger.info(f"📁 输出目录: {self.output_prepared}")
    
    def _run_multiview(self):
        """步骤1：多视角生成（0_0）"""
        logger.info(f"输入图像: {self.input_images}")
        logger.info(f"输入mask: {self.input_masks}")
        logger.info(f"输出目录: {self.output_multiview}")
        
        if not self.input_images.exists():
            raise FileNotFoundError(f"图像目录不存在: {self.input_images}")
        if not self.input_masks.exists():
            raise FileNotFoundError(f"Mask目录不存在: {self.input_masks}")
        
        script_path = Path(__file__).parent.parent / 'scripts' / '0_0_gen_multiviews.py'
        
        cmd = [
            sys.executable,
            str(script_path),
            '--input_images_dir', str(self.input_images),
            '--input_masks_dir', str(self.input_masks),
            '--output_folder', str(self.output_multiview),
            '--model_type', self.config.get('stage_0.multiview.model_type', 'sv4d2'),
            '--num_steps', str(self.config.get('stage_0.multiview.num_steps', 50)),
            '--seed', str(self.config.get('common.seed', 23))
        ]
        
        num_frames = self.config.get('common.num_frames')
        if num_frames:
            cmd.extend(['--num_frames', str(num_frames)])
        
        logger.info(f"执行命令: {' '.join(cmd[:3])} ...")
        result = subprocess.run(cmd, check=True)
        
        if result.returncode != 0:
            raise RuntimeError("多视角生成失败")
        
        logger.info(f"✓ 多视角数据已生成: {self.output_multiview}")
    
    def _run_preparation(self):
        """步骤2：数据预处理（0_1）"""
        logger.info(f"输入目录: {self.output_multiview}")
        logger.info(f"输出目录: {self.output_prepared}")
        
        # 验证输入
        multiview_dir = self.output_multiview / 'multiview_images'
        if not multiview_dir.exists():
            raise FileNotFoundError(
                f"多视角数据不存在: {multiview_dir}\n"
                f"请先运行 multiview 步骤"
            )
        
        # 创建处理器
        processor = DataProcessor.from_config(self.config)
        
        # 批量处理
        processor.process_scene(
            input_dir=str(self.output_multiview),
            output_dir=str(self.output_prepared),
            num_frames=self.config.get('common.num_frames')
        )
        
        logger.info(f"✓ 预处理数据已生成: {self.output_prepared}")
        
        # 统计输出
        frame_dirs = list(self.output_prepared.glob('[0-9]*'))
        logger.info(f"✓ 共处理 {len(frame_dirs)} 帧")
    
    def _get_start_index(self, resume_from: Optional[str]) -> int:
        """获取起始步骤索引"""
        if resume_from is None:
            return 0
        
        if resume_from not in self.STEPS:
            raise ValueError(
                f"无效的步骤名: {resume_from}. "
                f"可选: {', '.join(self.STEPS)}"
            )
        
        return self.STEPS.index(resume_from)


if __name__ == '__main__':
    print("🧪 测试阶段0 Pipeline...")
    
    from utils.config import Config
    
    # 测试创建
    config = Config()
    pipeline = Stage0Pipeline(config)
    
    print(f"\n✅ Pipeline创建成功")
    print(f"   - 输入图像: {pipeline.input_images}")
    print(f"   - 输入mask: {pipeline.input_masks}")
    print(f"   - 输出multiview: {pipeline.output_multiview}")
    print(f"   - 输出prepared: {pipeline.output_prepared}")
    print(f"\n流程步骤: {' → '.join(pipeline.STEPS)}")


