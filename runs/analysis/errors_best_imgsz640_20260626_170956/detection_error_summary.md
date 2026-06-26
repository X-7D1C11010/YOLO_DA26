# 独立测试集检测错误分析报告

- 权重：`/ssd_data/lixiang_data/YOLO_DA/runs/detect/YOLO26s_DA_DANN-4/weights/best.pt`
- 数据：`/ssd_data/lixiang_data/YOLO_DA/test.yaml` split=`val`
- imgsz=640，conf=0.001，NMS IoU=0.7，匹配 IoU=0.5

## 总体结果

- 图像数：275
- GT 数：1501
- 预测数：14920
- TP/FP/FN：1251 / 13669 / 250
- 重复 FP：1023
- 当前 conf 下 Precision：0.0838
- 当前 conf 下 Recall：0.8334
- 当前 conf 下 F1：0.1524
- GT 最佳 IoU 中位数：0.7332920046413489

## 按目标尺度召回

| 尺度 | GT | 命中 | Recall | 最佳 IoU 中位数 |
|---|---:|---:|---:|---:|
| large(>=0.02) | 273 | 172 | 0.6300 | 0.6753373690877353 |
| medium(0.005-0.02) | 893 | 765 | 0.8567 | 0.7392858999615521 |
| small(0.001-0.005) | 335 | 314 | 0.9373 | 0.7589766546342253 |

## 自动判断

- 误检明显多于漏检，优先提升背景抑制：加入 hard negative、提高伪标签阈值、增强 SAR 背景多样性。
- 重复预测较多，建议测试更高 NMS IoU/更低 max_det，或检查密集目标标注与 NMS 参数。
- GT 最佳 IoU 中位数低于 0.75，AP75 低主要来自定位不准或框尺度不匹配。

## 最严重漏检图 Top 10

| image | GT | TP | FP | FN | DUP |
|---|---:|---:|---:|---:|---:|
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/263_B-52_0008.png` | 8 | 2 | 13 | 6 | 1 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/265_KC-135_0001.png` | 11 | 5 | 5 | 6 | 0 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/263_C-130_0006.png` | 7 | 2 | 11 | 5 | 2 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/263_B-52_0005.png` | 7 | 2 | 9 | 5 | 2 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/BC2-SP-ORG-2SVV-20220729T133508-001682-000001-000692_C-130_0007.png` | 5 | 1 | 187 | 4 | 3 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/265_KC-135_0003.png` | 11 | 7 | 20 | 4 | 1 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/263_B-52_0002.png` | 7 | 3 | 18 | 4 | 0 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/263_B-1_0003.png` | 7 | 3 | 13 | 4 | 1 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/263_B-52_0007.png` | 8 | 4 | 9 | 4 | 2 |
| `/ssd_data/lixiang_data/Datasets/data_sar_plane_coarsness/images/261_B-52_0010.png` | 4 | 0 | 8 | 4 | 0 |

完整机器可读结果见 `detection_error_report.json`。
