"""运行官方 TACO 转换。"""
import argparse
import json
from pathlib import Path
from .pipeline import run_pipeline


def main() -> None:
    """无参数；解析命令行并打印实际统计，返回 None；I/O 或非法标注抛异常，框为显示像素 xywh。"""
    parser = argparse.ArgumentParser(description='仅从官方 annotations.json 生成三模型共用 COCO；原图只读、不复制。')
    parser.add_argument('--input',type=Path,default=Path('data/raw/taco'),help='原图根目录，包含官方 annotations.json')
    parser.add_argument('--output',type=Path,default=Path('data/raw/taco/annotations_project.json'),help='唯一输出 JSON 文件；不能覆盖官方标注')
    parser.add_argument('--seed',type=int,default=42,help='固定整组 70/15/15 分区种子（默认42），无K折')
    args = parser.parse_args()
    data = run_pipeline(args.input,args.output,args.seed)
    print(json.dumps(data['project']['statistics'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
