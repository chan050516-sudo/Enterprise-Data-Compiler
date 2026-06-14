import sqlite3
import pandas as pd
import sys
import os

# 请将这里替换为你的 northwind.db 文件的完整路径
DB_PATH = "northwind.db"

def export_all_tables_to_csv(db_path, output_dir="northwind_csv"):
    """从 SQLite 数据库中导出所有表为 CSV 文件"""
    if not os.path.exists(db_path):
        print(f"错误：找不到数据库文件 {db_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"数据库文件：{db_path}")
    print(f"CSV 文件将保存在：{output_dir}/")
    print("-" * 30)

    conn = sqlite3.connect(db_path)
    
    # 获取数据库中所有表的名字
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)
    
    if tables.empty:
        print("数据库中未找到任何表。")
        conn.close()
        return

    for table_name in tables['name']:
        print(f"正在导出表：{table_name} ...", end=' ', flush=True)
        try:
            df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)
            csv_file_path = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
            print(f"成功，已导出 {len(df)} 行。")
        except Exception as e:
            print(f"失败，原因：{e}")

    conn.close()
    print("-" * 30)
    print("所有表导出完成！")

if __name__ == "__main__":
    export_all_tables_to_csv(DB_PATH)