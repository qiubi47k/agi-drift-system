#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init_pool.py - 辅助脚本：批量导入素材到缓冲池

用法：
    # 导入单个文件
    python init_pool.py --file path/to/file.md
    
    # 导入目录下所有md文件
    python init_pool.py --dir path/to/directory
    
    # 从stdin读取（管道输入）
    echo "素材内容" | python init_pool.py --stdin
    
    # 交互式输入（逐行输入，空行结束）
    python init_pool.py --input
    
    # 指定来源类型（默认raw）
    python init_pool.py --dir path/to/dir --source feedback
"""

import os
import sys
import argparse
import logging

# 确保模块路径正确
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from buffer import BufferPool


def setup_logging():
    """配置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("drift.init_pool")


def import_file(filepath: str, pool: BufferPool, source: str = "raw") -> str:
    """导入单个文件
    
    Args:
        filepath: 文件路径
        pool: 缓冲池实例
        source: 来源标记（"raw" 或 "feedback"）
        
    Returns:
        素材ID
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    content = content.strip()
    if not content:
        return None
    
    if source == "feedback":
        return pool.write_feedback(content)
    else:
        return pool.write_raw(content)


def import_directory(dirpath: str, pool: BufferPool, source: str = "raw") -> int:
    """导入目录下所有文本文件
    
    支持的文件格式：.md, .txt, .text
    递归扫描子目录。
    
    Args:
        dirpath: 目录路径
        pool: 缓冲池实例
        source: 来源标记
        
    Returns:
        成功导入的文件数
    """
    import_count = 0
    supported_extensions = {".md", ".txt", ".text"}
    
    for root, dirs, files in os.walk(dirpath):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                continue
            
            filepath = os.path.join(root, filename)
            try:
                item_id = import_file(filepath, pool, source)
                if item_id:
                    import_count += 1
                    print(f"  ✓ {filepath} -> {item_id[:8]}...")
                else:
                    print(f"  ⚠ {filepath} (内容为空，跳过)")
            except Exception as e:
                print(f"  ✗ {filepath} 导入失败: {e}")
    
    return import_count


def main():
    """主入口"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description="批量导入素材到漂移系统缓冲池",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python init_pool.py --file notes.md
  python init_pool.py --dir ./my_materials --source feedback
  python init_pool.py --input
  echo "素材内容" | python init_pool.py --stdin
        """
    )
    
    # 输入源（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", "-f", help="导入单个文件")
    input_group.add_argument("--dir", "-d", help="导入目录下所有文本文件")
    input_group.add_argument("--stdin", "-s", action="store_true", 
                            help="从标准输入读取")
    input_group.add_argument("--input", "-i", action="store_true",
                            help="交互式输入模式")
    
    # 其他选项
    parser.add_argument("--source", choices=["raw", "feedback"], default="raw",
                       help="素材来源标记（默认: raw）")
    
    args = parser.parse_args()
    
    # 初始化缓冲池
    pool = BufferPool()
    
    print(f"\n{'='*50}")
    print(f"  📥 素材导入工具")
    print(f"  来源标记: {args.source}")
    print(f"{'='*50}\n")
    
    imported_count = 0
    
    if args.file:
        # 导入单个文件
        filepath = os.path.abspath(args.file)
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在 - {filepath}")
            sys.exit(1)
        
        try:
            item_id = import_file(filepath, pool, args.source)
            if item_id:
                print(f"  ✓ 导入成功: {filepath}")
                print(f"    ID: {item_id}")
                imported_count = 1
            else:
                print(f"  ⚠ 文件内容为空: {filepath}")
        except Exception as e:
            print(f"  ✗ 导入失败: {e}")
    
    elif args.dir:
        # 导入目录
        dirpath = os.path.abspath(args.dir)
        if not os.path.isdir(dirpath):
            print(f"错误: 目录不存在 - {dirpath}")
            sys.exit(1)
        
        print(f"  扫描目录: {dirpath}")
        imported_count = import_directory(dirpath, pool, args.source)
    
    elif args.stdin:
        # 从标准输入读取
        print("  从标准输入读取（Ctrl+D结束）:")
        content = sys.stdin.read().strip()
        if content:
            if args.source == "feedback":
                item_id = pool.write_feedback(content)
            else:
                item_id = pool.write_raw(content)
            print(f"  ✓ 导入成功，ID: {item_id}")
            imported_count = 1
        else:
            print("  ⚠ 输入内容为空")
    
    elif args.input:
        # 交互式输入
        print("  交互式输入模式（输入空行结束一条，输入 'EOF' 结束全部）:")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            except EOFError:
                break
        
        content = "\n".join(lines).strip()
        if content:
            if args.source == "feedback":
                item_id = pool.write_feedback(content)
            else:
                item_id = pool.write_raw(content)
            print(f"  ✓ 导入成功，ID: {item_id}")
            imported_count = 1
        else:
            print("  ⚠ 输入内容为空")
    
    # 打印导入结果
    print(f"\n{'='*50}")
    print(f"  导入完成: {imported_count}条素材")
    stats = pool.get_stats()
    print(f"  缓冲池状态: 总计{stats['total']}条, "
          f"未消化{stats['undigested']}条")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
