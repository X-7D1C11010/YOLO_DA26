"""YOLO26 光学-SAR 域对抗训练入口。"""

from da_training import main


if __name__ == "__main__":
    main(default_version="26", default_size="s")
