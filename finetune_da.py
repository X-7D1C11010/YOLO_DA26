"""兼容入口。

原 Deep CORAL 实现会在无检测约束下更新全部网络，并在现有日志中反复 OOM。
项目现统一使用 finetune_uda.py 的“伪标签 + 有标签锚定”安全微调流程。
"""

from finetune_uda import main


if __name__ == "__main__":
    main()
