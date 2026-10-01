"""
完整流程编排：从原始图像到最终视频

流程：
  阶段0: 数据准备（0_0 → 0_1）
  阶段1: 点云处理 + 渲染（1_0 → 1_1 → 1_2 → 1_4）
  阶段2: 视频生成（2_0）

输入：原始图像 + mask
输出：最终生成视频
"""

from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from pipeline.stage_0_pipeline import Stage0Pipeline
from pipeline.stage_1_pipeline import Stage1Pipeline
from pipeline.stage_2_pipeline import Stage2VideoPipeline

logger = setup_logger('full_pipeline')


class FullPipeline:
    """端到端完整流程"""
    
    # 定义所有步骤（跨阶段）
    ALL_STEPS = [
        # 阶段0
        'multiview', 'preparation',
        # 阶段1
        'foreground', 'background', 'align', 'render',
        # 阶段2
        'video'
    ]
    
    STAGE_0_STEPS = ['multiview', 'preparation']
    STAGE_1_STEPS = ['foreground', 'background', 'align', 'render']
    STAGE_2_STEPS = ['video']
    
    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
    
    def run(self, 
            resume_from: Optional[str] = None,
            trajectory_json: Optional[str] = None,
            trajectory_name: Optional[str] = None):
        """
        运行完整流程
        
        Args:
            resume_from: 从哪个步骤继续
                - 'multiview' / 'preparation': 从阶段0的某步开始
                - 'foreground' / 'background' / 'align' / 'render': 跳过阶段0，从阶段1开始
                - 'video': 跳过阶段0和阶段1，只运行阶段2
            trajectory_json: 轨迹JSON文件（用于阶段1渲染）
            trajectory_name: 轨迹子目录名（用于阶段2视频生成）
        """
        logger.info("=" * 60)
        logger.info("🚀 完整流程：端到端")
        logger.info("=" * 60)
        logger.info(f"场景: {self.config.get('project.name', 'default')}")
        if resume_from:
            logger.info(f"续传模式: 从 '{resume_from}' 开始")
        if trajectory_json:
            logger.info(f"使用轨迹: {trajectory_json}")
        logger.info("=" * 60)
        
        # 如果没有指定trajectory_name，但指定了trajectory_json，提前从json文件名提取
        # 这样可以在整个流程中使用
        if not trajectory_name and trajectory_json:
            from pathlib import Path
            trajectory_name = Path(trajectory_json).stem
            logger.info(f"从trajectory_json提取轨迹名称: {trajectory_name}")
        
        # 保存实际使用的轨迹名称（用于最后的输出显示）
        self._actual_trajectory_name = trajectory_name
        
        # 确定起始阶段和步骤
        if resume_from is None:
            # 从阶段0开始
            run_stage_0 = True
            run_stage_1 = True
            run_stage_2 = True
            stage_0_resume_from = None
            stage_1_resume_from = None
        elif resume_from in self.STAGE_0_STEPS:
            # 从阶段0的某步开始
            run_stage_0 = True
            run_stage_1 = True
            run_stage_2 = True
            stage_0_resume_from = resume_from
            stage_1_resume_from = None
        elif resume_from in self.STAGE_1_STEPS:
            # 跳过阶段0，从阶段1的某步开始
            run_stage_0 = False
            run_stage_1 = True
            run_stage_2 = True
            stage_1_resume_from = resume_from
        elif resume_from in self.STAGE_2_STEPS:
            # 跳过阶段0和1，只运行阶段2
            run_stage_0 = False
            run_stage_1 = False
            run_stage_2 = True
            stage_1_resume_from = None
        else:
            raise ValueError(
                f"无效的步骤名: {resume_from}\n"
                f"可选: {', '.join(self.ALL_STEPS)}"
            )
        
        # 执行阶段0
        if run_stage_0:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# 阶段0: 数据准备")
            logger.info(f"{'#'*60}\n")
            
            stage_0 = Stage0Pipeline(self.config)
            stage_0.run(resume_from=stage_0_resume_from if resume_from in self.STAGE_0_STEPS else None)
        else:
            logger.info("\n⏭️  跳过阶段0")
        
        # 执行阶段1
        if run_stage_1:
            logger.info(f"\n{'#'*60}")
            logger.info(f"# 阶段1: 点云处理 + 渲染")
            logger.info(f"{'#'*60}\n")
            
            stage_1 = Stage1Pipeline(self.config)
            stage_1.run(
                resume_from=stage_1_resume_from,
                trajectory_json=trajectory_json
            )
        else:
            logger.info("\n⏭️  跳过阶段1")
        
        # 执行阶段2
        logger.info(f"\n{'#'*60}")
        logger.info(f"# 阶段2: 视频生成")
        logger.info(f"{'#'*60}\n")
        
        # trajectory_name在前面已经提取好了
        stage_2 = Stage2VideoPipeline(self.config)
        stage_2.run(trajectory_name=trajectory_name)
        
        # 完成
        logger.info("\n" + "=" * 60)
        logger.info("🎉 完整流程完成！")
        logger.info("=" * 60)
        self._print_outputs()
    
    def _print_outputs(self):
        """打印输出文件位置"""
        scene_name = self.config.get('project.name', 'default')
        output_root = self.config.get('project.output_root', 'outputs')
        
        # 使用实际的轨迹名称（如果有的话），否则使用配置文件中的默认值
        if hasattr(self, '_actual_trajectory_name') and self._actual_trajectory_name:
            trajectory_name = self._actual_trajectory_name
        else:
            arc_type = self.config.get('stage_1.rendering.arc_type', 'yaw')
            arc_angle = self.config.get('stage_1.rendering.arc_angle', 90)
            trajectory_name = f"arc_{arc_type}_{int(arc_angle)}"
        
        logger.info("\n📁 输出文件位置:")
        logger.info(f"  - 多视角: {output_root}/multiview/{scene_name}")
        logger.info(f"  - 点云: {output_root}/prepared/{scene_name}")
        logger.info(f"  - 渲染: {output_root}/rendering/{scene_name}/{trajectory_name}")
        logger.info(f"  - 视频: {output_root}/rendering/{scene_name}/{trajectory_name}/inference/output_video.mp4")


if __name__ == '__main__':
    print("🧪 测试完整Pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = FullPipeline(config)
    
    print(f"\n✅ Pipeline创建成功")
    print(f"   - 场景: {config.get('project.name', 'default')}")
    
    print(f"\n完整流程:")
    print(f"  阶段0: {' → '.join(pipeline.STAGE_0_STEPS)}")
    print(f"  阶段1: {' → '.join(pipeline.STAGE_1_STEPS)}")
    
    print(f"\n支持的resume_from步骤:")
    print(f"  {', '.join(pipeline.ALL_STEPS)}")
