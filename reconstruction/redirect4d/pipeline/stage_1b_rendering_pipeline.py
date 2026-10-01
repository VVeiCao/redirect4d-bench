"""
阶段1B流程编排：渲染

流程：
  1_4 渲染（支持两种轨迹模式）

轨迹模式：
  1. 自动模式（默认）：自动生成arc轨迹
  2. 手动模式：使用预先生成的轨迹json（来自1_3）

输入：
  - outputs/prepared/（点云数据）
  - 轨迹JSON（可选）

输出：
  - outputs/rendering/{scene}/{trajectory_name}/
"""

from pathlib import Path
from typing import Optional

from utils.config import Config
from utils.logging import setup_logger
from utils.file_io import load_json
from core.rendering import PointCloudRenderer

logger = setup_logger('stage_1b_rendering')


class Stage1BRenderingPipeline:
    """阶段1B：渲染"""
    
    def __init__(self, config: Config):
        """
        Args:
            config: 配置对象
        """
        self.config = config
        self.data_dir = Path(config.get('project.output_prepared'))
    
    def run(self, trajectory_json: Optional[str] = None):
        """
        运行渲染流程
        
        Args:
            trajectory_json: 
                - None: 自动生成arc轨迹（默认）
                - "path/to/trajectory.json": 使用预先生成的轨迹
        """
        logger.info("=" * 60)
        logger.info("阶段1B：渲染")
        logger.info("=" * 60)
        
        # 确定轨迹模式
        if trajectory_json is None:
            mode = "自动模式（arc轨迹）"
            trajectory_source = self._auto_generate_trajectory()
        else:
            mode = "手动模式（使用预先生成的轨迹）"
            trajectory_source = self._validate_trajectory_json(trajectory_json)
        
        logger.info(f"轨迹模式: {mode}")
        logger.info(f"轨迹文件: {trajectory_source}")
        
        # 渲染
        logger.info("\n开始渲染...")
        renderer = PointCloudRenderer.from_config(self.config)
        
        output_dir = renderer.render_trajectory(
            data_dir=str(self.data_dir),
            trajectory_json=str(trajectory_source)
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 渲染完成！")
        logger.info("=" * 60)
        logger.info(f"📁 输出目录: {output_dir}")
        logger.info(f"📹 视频文件:")
        logger.info(f"  - {output_dir}/videos/rendered_images.mp4")
        logger.info(f"  - {output_dir}/videos/rendered_depths.mp4")
        logger.info(f"\n💡 推理文件（用于视频生成）:")
        logger.info(f"  - {output_dir}/inference/")
    
    def _auto_generate_trajectory(self) -> Path:
        """
        自动生成arc轨迹
        
        Returns:
            生成的轨迹json文件路径
        """
        logger.info("\n🔄 自动生成arc轨迹...")
        
        # 从配置读取参数
        arc_type = self.config.get('stage_1.rendering.arc_type', 'yaw')
        arc_angle = self.config.get('stage_1.rendering.arc_angle', 90)
        
        logger.info(f"  - 圆弧类型: {arc_type}")
        logger.info(f"  - 圆弧角度: {arc_angle}°")
        logger.info(f"  - 半径和仰角: 自动从点云数据计算")
        
        # 生成轨迹文件路径（包含类型和角度）
        trajectory_name = f"arc_{arc_type}_{int(arc_angle)}"
        trajectory_path = self.data_dir / f"{trajectory_name}.json"
        
        # 总是重新生成轨迹（确保帧数与当前数据一致）
        if trajectory_path.exists():
            logger.info(f"⚠️  删除旧轨迹: {trajectory_path}")
            trajectory_path.unlink()
        
        # 生成新轨迹
        logger.info(f"✓ 生成新轨迹...")
        renderer = PointCloudRenderer.from_config(self.config)
        
        trajectory_json = renderer.generate_arc_trajectory(
            data_dir=str(self.data_dir),
            arc_type=arc_type,
            arc_angle=arc_angle,
            num_frames=None,  # 自动检测
            save_json_path=trajectory_path.name
        )
        
        logger.info(f"✓ 轨迹已生成: {trajectory_json}")
        return Path(trajectory_json)
    
    def _validate_trajectory_json(self, trajectory_json: str) -> Path:
        """
        验证手动提供的轨迹json
        
        Args:
            trajectory_json: 轨迹文件路径（相对或绝对）
        
        Returns:
            验证后的路径
        """
        logger.info(f"\n✓ 使用手动生成的轨迹: {trajectory_json}")
        
        # 转换为Path对象
        traj_path = Path(trajectory_json)
        
        # 如果是相对路径，相对于data_dir解析
        if not traj_path.is_absolute():
            traj_path = self.data_dir / traj_path
        
        # 验证文件存在
        if not traj_path.exists():
            raise FileNotFoundError(
                f"轨迹文件不存在: {traj_path}\n"
                f"💡 提示:\n"
                f"  1. 检查文件路径是否正确\n"
                f"  2. 或使用 1_3_cam_traj.py 生成轨迹:\n"
                f"     python scripts/1_3_cam_traj.py --data_dir {self.data_dir}"
            )
        
        # 验证JSON格式
        try:
            data = load_json(str(traj_path))
            
            if 'camera_path' not in data:
                raise ValueError("轨迹JSON缺少 'camera_path' 字段")
            
            logger.info(f"  - 轨迹帧数: {len(data['camera_path'])}")
            
        except Exception as e:
            raise ValueError(f"轨迹JSON格式错误: {e}")
        
        return traj_path


if __name__ == '__main__':
    print("🧪 测试阶段1B Pipeline...")
    
    from utils.config import Config
    
    config = Config()
    pipeline = Stage1BRenderingPipeline(config)
    
    print(f"\n✅ Pipeline创建成功")
    print(f"   - 数据目录: {pipeline.data_dir}")
    print(f"\n支持两种轨迹模式:")
    print(f"   1. 自动模式: 生成arc轨迹（默认）")
    print(f"   2. 手动模式: 使用预先生成的JSON")
    print(f"\n默认arc参数:")
    print(f"   - 角度: {config.get('stage_1.rendering.arc_angle')}°")
    print(f"   - 半径/仰角: 自动从点云计算")
