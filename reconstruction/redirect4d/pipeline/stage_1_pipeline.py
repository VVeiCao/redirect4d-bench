"""
阶段1完整流程：点云处理 + 渲染

流程：
  阶段1A: 1_0 → 1_1 → 1_2（点云）
  阶段1B: 1_4（渲染）

注意：
  - 1_3 作为独立的交互式工具，不在自动化流程中
  - 渲染默认使用自动生成的arc轨迹
  - 也可以使用预先通过1_3生成的轨迹
"""

from typing import Optional
from utils.config import Config
from utils.logging import setup_logger
from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline
from pipeline.stage_1b_rendering_pipeline import Stage1BRenderingPipeline

logger = setup_logger('stage_1')


class Stage1Pipeline:
    """阶段1完整流程"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
    
    def run(self, 
            resume_from: Optional[str] = None,
            trajectory_json: Optional[str] = None):
        """
        运行完整的阶段1流程
        
        Args:
            resume_from: 从哪个步骤继续
                - 'foreground', 'background', 'align': 从阶段1A的某步开始
                - 'render': 跳过1A，只运行1B
            trajectory_json: 渲染用的轨迹文件（None则自动生成arc轨迹）
        """
        logger.info("=" * 60)
        logger.info("阶段1：点云处理 + 渲染")
        logger.info("=" * 60)
        
        # 阶段1A：点云处理
        if resume_from != 'render':
            logger.info("\n### 阶段1A：点云生成和对齐 ###\n")
            stage_1a = Stage1APointCloudPipeline(self.config)
            stage_1a.run(resume_from=resume_from)
        else:
            logger.info("⏭️  跳过阶段1A（从render开始）")
        
        # 阶段1B：渲染
        logger.info("\n### 阶段1B：渲染 ###\n")
        stage_1b = Stage1BRenderingPipeline(self.config)
        stage_1b.run(trajectory_json=trajectory_json)
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 阶段1完整流程完成！")
        logger.info("=" * 60)


if __name__ == '__main__':
    print("🧪 测试阶段1 Pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = Stage1Pipeline(config)
    
    print(f"\n✅ Pipeline创建成功")
    print(f"\n完整流程:")
    print(f"  阶段1A:")
    print(f"    1. foreground - 前景点云")
    print(f"    2. background - 背景点云")
    print(f"    3. align - 点云对齐")
    print(f"  阶段1B:")
    print(f"    4. render - 渲染（arc轨迹）")
    print(f"\n支持断点续传:")
    print(f"  --resume_from foreground|background|align|render")


