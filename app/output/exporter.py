import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class SecondaryExporter:
    """
    Layer 8: 二级格式化文件导出器
    负责为财务/审计系统输出标准兼容的、带有安全编码（UTF-8-SIG）的数据资产快照。
    """

    @staticmethod
    def to_csv(clean_df: pd.DataFrame, file_path: str, delimiter: str = ",") -> str:
        """
        导出为完美的、带 Excel BOM 头防护的 CSV 文件
        """
        if clean_df.empty:
            logger.warning("Clean dataset is empty. CSV export generated an empty shell.")
            
        try:
            # 确保目标输出文件夹存在
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                
            # 使用 utf-8-sig 注入 BOM 字节头，彻底消灭 Windows Excel 直接双击打开时产生的中文乱码灾难
            clean_df.to_csv(
                file_path,
                index=False,
                sep=delimiter,
                encoding="utf-8-sig"
            )
            logger.info(f"Secondary CSV Snapshot exported successfully to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export CSV file to {file_path}: {e}")
            raise RuntimeError(f"Layer 8 File Exporter Panic: {str(e)}")