"""清洁机器人视觉系统命令行入口。"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="robot-cleaning-vision",
        description="清洁机器人视觉系统。",
        epilog="架构与各模块说明见 docs/架构设计.md。",
    )
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
