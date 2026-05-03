#!/usr/bin/env python3
"""
Pipeline执行工具 - 自动化流程管理

Pipeline选择：
  stage_0    阶段0：数据准备（0_0 → 0_1）
  stage_1a   阶段1A：点云处理（1_0 → 1_1 → 1_2）
  stage_1b   阶段1B：渲染（1_4）
  stage_1    阶段1完整（1A + 1B）
  stage_2    阶段2：视频生成（2_0）
  full       端到端（0 → 1 → 2）

--resume_from 参数（断点续传）：
  stage_0:  multiview | preparation
  stage_1a: foreground | background | align
  stage_1:  foreground | background | align | render
  full:     multiview | preparation | foreground | background | align | render | video

使用示例：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 点云处理
   python run_pipeline.py stage_1a --config configs/scenes/camel.yaml

2. 渲染
   python run_pipeline.py stage_1b --config configs/scenes/camel.yaml

3. 视频生成
   python run_pipeline.py full --config configs/scenes/robot.yaml --resume_from background

4. 完整流程（点云 + 渲染）
   python run_pipeline.py stage_1 --config configs/scenes/car-turn_yaw_-120.yaml --resume_from background

5. 端到端流程（数据 + 点云 + 渲染 + 视频）
python run_pipeline.py full --config configs/scenes/unitree6.yaml && python run_pipeline.py full --config configs/scenes/unitree7.yaml
export CUDA_VISIBLE_DEVICES=1
   python run_pipeline.py full --config configs/eval/car-roundabout_yaw_-120.yaml --resume_from video --trajectory_json original_global_camera.json
   cp outputs/rendering/lecun2/arc_yaw_-120/inference/reference_image.png outputs/rendering/lecun2/test2/inference/reference_image.png
   python run_pipeline.py full --config configs/scenes/car-roundabout-camel.yaml --resume_from render --trajectory_json test2.json

   python run_pipeline.py full --config configs/scenes/lecun4.yaml --resume_from render --trajectory_json test1.json
   python run_pipeline.py full --config configs/eval/bear_yaw_-120.yaml --resume_from render --trajectory_json test_1.json

   python run_pipeline.py full --config configs/eval/parkour_yaw_-120.yaml --resume_from render --trajectory_json round_1.json


6. 断点续传
   python run_pipeline.py stage_1a --config configs/scenes/camel.yaml --resume_from align

7. 调试模式（少量帧）
   python run_pipeline.py stage_1a --config configs/scenes/camel.yaml --num_frames 3
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.config import Config
from utils.logging import setup_logger
from pipeline.stage_0_pipeline import Stage0Pipeline
from pipeline.stage_1a_pointcloud_pipeline import Stage1APointCloudPipeline
from pipeline.stage_1b_rendering_pipeline import Stage1BRenderingPipeline
from pipeline.stage_1_pipeline import Stage1Pipeline
from pipeline.stage_2_pipeline import Stage2VideoPipeline
from pipeline.full_pipeline import FullPipeline

logger = setup_logger('pipeline')


def create_parser():
    """创建参数解析器"""
    parser = argparse.ArgumentParser(
        description='Pipeline执行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 点云处理
  python run_pipeline.py stage_1a --config configs/scenes/camel.yaml
  
  # 渲染
  python run_pipeline.py stage_1b --config configs/scenes/camel.yaml
  
  # 视频生成
  python run_pipeline.py stage_2 --config configs/scenes/camel.yaml
  
  # 完整流程（点云 + 渲染 + 视频）
  python run_pipeline.py full --config configs/scenes/camel.yaml

更多帮助请查看脚本顶部的详细文档。
        """
    )
    
    # Pipeline选择
    parser.add_argument(
        'pipeline',
        choices=['stage_0', 'stage_1a', 'stage_1b', 'stage_1', 'stage_2', 'full'],
        help='''要执行的Pipeline:
  stage_0  - 阶段0（0_0→0_1）
  stage_1a - 阶段1A（1_0→1_1→1_2）
  stage_1b - 阶段1B（1_4渲染）
  stage_1  - 阶段1完整（1A+1B）
  stage_2  - 阶段2（2_0视频生成）
  full     - 端到端（0→1→2）'''
    )
    
    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径（YAML格式）'
    )
    
    # 断点续传
    parser.add_argument(
        '--resume_from',
        type=str,
        metavar='STEP',
        help='''从指定步骤继续:
  stage_0:  multiview | preparation
  stage_1a: foreground | background | align
  stage_1:  foreground | background | align | render
  full:     multiview | preparation | foreground | background | align | render | video'''
    )
    
    # 通用参数
    parser.add_argument('--num_frames', type=int, help='处理帧数（调试用）')
    
    # 阶段1B/阶段1特定参数
    parser.add_argument('--trajectory_json', type=str, 
                       help='轨迹JSON文件（用于stage_1b/stage_1）')
    parser.add_argument('--arc_angle', type=float,
                       help='Arc轨迹角度（度，覆盖配置文件）')
    
    # 阶段2特定参数
    parser.add_argument('--trajectory_name', type=str,
                       help='轨迹子目录名（用于stage_2，如arc_-120）')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 覆盖参数
    if args.num_frames is not None:
        config.update('common.num_frames', args.num_frames)
    if args.arc_angle is not None:
        config.update('stage_1.rendering.arc_angle', args.arc_angle)
    
    logger.info("=" * 60)
    logger.info(f"Pipeline: {args.pipeline}")
    logger.info(f"配置文件: {args.config}")
    if args.resume_from:
        logger.info(f"续传模式: 从 '{args.resume_from}' 开始")
    logger.info("=" * 60)
    
    # 执行对应的Pipeline
    try:
        if args.pipeline == 'stage_0':
            pipeline = Stage0Pipeline(config)
            pipeline.run(resume_from=args.resume_from)
            
        elif args.pipeline == 'stage_1a':
            pipeline = Stage1APointCloudPipeline(config)
            pipeline.run(resume_from=args.resume_from)
            
        elif args.pipeline == 'stage_1b':
            if args.resume_from:
                logger.warning("stage_1b只有一个步骤，忽略--resume_from")
            pipeline = Stage1BRenderingPipeline(config)
            pipeline.run(trajectory_json=args.trajectory_json)
            
        elif args.pipeline == 'stage_1':
            pipeline = Stage1Pipeline(config)
            pipeline.run(
                resume_from=args.resume_from,
                trajectory_json=args.trajectory_json
            )
            
        elif args.pipeline == 'stage_2':
            if args.resume_from:
                logger.warning("stage_2只有一个步骤，忽略--resume_from")
            pipeline = Stage2VideoPipeline(config)
            pipeline.run(trajectory_name=args.trajectory_name)
            
        elif args.pipeline == 'full':
            pipeline = FullPipeline(config)
            pipeline.run(
                resume_from=args.resume_from,
                trajectory_json=args.trajectory_json,
                trajectory_name=args.trajectory_name
            )
        
        logger.info("\n✅ Pipeline执行完成！")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline执行失败: {e}")
        logger.info("\n💡 提示:")
        logger.info("  - 检查配置文件是否正确")
        logger.info("  - 检查输入数据是否存在")
        raise


if __name__ == '__main__':
    main()
