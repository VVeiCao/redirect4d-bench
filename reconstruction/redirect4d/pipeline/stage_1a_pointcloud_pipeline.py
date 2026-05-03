"""
阶段1A流程编排：点云生成和对齐

流程：
  1_0 前景点云生成 → 1_1 背景点云生成 → 1_2 点云对齐

输入：
  - outputs/prepared/

输出：
  - {frame_id}/pointcloud/*.ply
  - global_background.ply
"""

from pathlib import Path
from typing import Optional, List

from utils.config import Config
from utils.logging import setup_logger
from utils.file_io import find_frame_dirs
from core.pointcloud import ForegroundPointCloudGenerator, BackgroundPointCloudGenerator
from core.alignment import PointCloudAligner

logger = setup_logger('stage_1a_pointcloud')


class Stage1APointCloudPipeline:
    """阶段1A：点云生成和对齐"""
    
    STEPS = ['foreground', 'background', 'align']
    
    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        self.data_dir = Path(config.get('project.output_prepared'))
    
    def run(self, resume_from: Optional[str] = None, stop_at: Optional[str] = None):
        """
        运行流程

        Args:
            resume_from: 从哪个步骤继续（None表示从头开始）
            stop_at: 执行完该步骤后停止（None表示跑到最后）
        """
        logger.info("=" * 60)
        logger.info("阶段1A：点云生成和对齐")
        logger.info("=" * 60)
        logger.info(f"数据目录: {self.data_dir}")
        logger.info("=" * 60)

        start_idx = self._get_start_index(resume_from)

        if resume_from:
            logger.info(f"\n→ 从步骤 '{resume_from}' 开始\n")
        if stop_at:
            if stop_at not in self.STEPS:
                raise ValueError(f"Invalid stop_at: {stop_at}. Options: {self.STEPS}")
            logger.info(f"\n→ 跑到 '{stop_at}' 就停\n")

        # 执行步骤
        for i in range(start_idx, len(self.STEPS)):
            step_name = self.STEPS[i]
            logger.info(f"\n{'='*60}")
            logger.info(f"步骤 {i+1}/{len(self.STEPS)}: {step_name}")
            logger.info(f"{'='*60}")

            try:
                step_method = getattr(self, f'_run_{step_name}')
                step_method()
                logger.info(f"✅ {step_name} 完成\n")

            except Exception as e:
                logger.error(f"❌ {step_name} 失败: {e}")
                logger.info(f"💡 修复后可以继续: --resume_from {step_name}")
                raise

            if stop_at == step_name:
                logger.info(f"\n→ stop_at='{stop_at}' 已到,跳过后续步骤\n")
                break

        logger.info("=" * 60)
        logger.info("🎉 阶段1A完成！")
        logger.info("=" * 60)
    
    def _run_foreground(self):
        """步骤1：前景点云生成（1_0）"""
        logger.info("使用 ForegroundPointCloudGenerator")
        
        generator = ForegroundPointCloudGenerator.from_config(self.config)
        generator.load_model()
        generator.batch_process(
            folder=str(self.data_dir),
            num_frames=self.config.get('common.num_frames')
        )
    
    def _run_background(self):
        """步骤2：背景点云生成（1_1）"""
        method = self.config.get('stage_1.background.method', 'dpg')

        if method == 'vipe':
            logger.info("使用 VIPeBackgroundGenerator (vipe)")
            from core.vipe_background import VIPeBackgroundGenerator
            generator = VIPeBackgroundGenerator.from_config(self.config)
            generator.generate_global_background(data_dir=str(self.data_dir))
        elif method == 'megasam':
            logger.info("使用 MegaSaMBackgroundGenerator (megasam)")
            from core.megasam_background import MegaSaMBackgroundGenerator
            generator = MegaSaMBackgroundGenerator.from_config(self.config)
            generator.generate_global_background(data_dir=str(self.data_dir))
        else:
            logger.info("使用 BackgroundPointCloudGenerator (DPG)")
            generator = BackgroundPointCloudGenerator.from_config(self.config)
            generator.load_model()
            generator.generate_global_background(data_dir=str(self.data_dir))
    
    def _run_align(self):
        """步骤3：点云对齐（1_2）"""
        logger.info("使用 PointCloudAligner")
        
        aligner = PointCloudAligner.from_config(self.config)
        aligner.align_all_frames(
            folder=str(self.data_dir),
            num_frames=self.config.get('common.num_frames')
        )
    
    def _get_start_index(self, resume_from: Optional[str]) -> int:
        """获取起始步骤索引"""
        if resume_from is None:
            return 0
        if resume_from not in self.STEPS:
            raise ValueError(
                f"无效的步骤名: {resume_from}\n"
                f"可选: {', '.join(self.STEPS)}"
            )
        return self.STEPS.index(resume_from)


if __name__ == '__main__':
    print("🧪 测试阶段1A Pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = Stage1APointCloudPipeline(config)
    
    print(f"\n✅ Pipeline创建成功")
    print(f"   - 数据目录: {pipeline.data_dir}")
    print(f"\n流程步骤: {' → '.join(pipeline.STEPS)}")
    print(f"\n支持断点续传: --resume_from {step}" for step in pipeline.STEPS)
