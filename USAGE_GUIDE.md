# Debate-to-Detect 使用指南

本文档介绍如何使用批量检测功能进行 AI 文本检测。

## 功能概述

1. **批量数据加载** - 支持多种数据源（包括测试集）
2. **批量检测** - 自动处理多条文本并保存结果
3. **指标计算** - AUROC、Accuracy、F1 等指标
4. **结构化输出** - 层级化的结果文件组织

## 使用方式

### 方式一：使用 main.py（推荐）

#### 单文本模式

```bash
# 直接输入文本
python main.py --single --text "这里输入要检测的文本"

# 从文件读取
python main.py --single --file input.txt

# 指定输出文件
python main.py --single --text "xxx" --output result.txt

# 使用不同模型
python main.py --single --text "xxx" --model gpt-4o
```

#### 批量模式

```bash
# 使用测试集（小样本）
python main.py --batch --data-source test --max-samples 10

# 使用 main 数据源
python main.py --batch --data-source main --dataset xsum --source-model gpt4o --max-samples 50

# 使用 m4 数据集
python main.py --batch --data-source m4 --max-samples 100

# 使用 text_attack 数据集
python main.py --batch --data-source text_attack --attack-type delete --max-samples 20
```

### 方式二：直接使用 batch_detect.py

```bash
# 基本用法
python batch_detect.py --data-source test --max-samples 10

# 完整参数
python batch_detect.py \
    --data-source main \
    --dataset xsum \
    --source-model gpt4o \
    --max-samples 50 \
    --model gpt-4o-mini \
    --temperature 1.0 \
    --sleep 1.0 \
    --threshold 0.5 \
    --output-dir Results
```

## 支持的数据源

| 数据源 | 说明 | 参数 |
|--------|------|------|
| `test` | 小样本测试集 | --max-samples |
| `main` | 主要数据集 | --dataset, --source-model |
| `m4` | M4 数据集 | --max-samples |
| `detectrl_multidomain` | DetectRL 多域 | --max-samples |
| `detectrl_multillm` | DetectRL 多模型 | --max-samples |
| `raid` | RAID 数据集 | --max-samples |
| `text_attack` | 文本攻击数据集 | --attack-type, --source-model |
| `realdet` | RealDet 数据集 | --max-samples |
| `base` | Base 模型数据集 | --base-dataset, --base-source-model |

## 输出结构

批量检测结果会保存到 `Results/batch_<timestamp>/` 目录：

```
Results/batch_20240312_143022/
├── config.json              # 运行配置
├── individual_results/      # 单个结果
│   ├── human_0000.json
│   ├── human_0001.json
│   ├── ai_0000.json
│   └── ...
├── metrics.json             # 评估指标
└── summary_report.txt       # 总结报告
```

## 输出指标说明

### metrics.json 包含：

- **auroc**: ROC 曲线下面积（0-1，越高越好）
- **accuracy**: 准确率（在指定阈值下的分类准确率）
- **f1**: F1 分数（在指定阈值下）
- **best_f1**: 最佳 F1 分数
- **best_f1_threshold**: 最佳 F1 对应的阈值
- **threshold**: 当前使用的阈值

### summary_report.txt 包含：

- 运行配置信息
- 处理样本总数
- Verdict 分布（AI_GENERATED/HUMAN_WRITTEN/UNCERTAIN）
- 正确预测数量和比例
- 所有评估指标

## 创建测试集

要创建自己的测试集，在数据目录下创建以下文件：

```
/data1/wujunxi/kailin/data/Test/
├── test_human.json    # 人类文本列表
└── test_machine.json  # AI生成文本列表
```

JSON 格式：

```json
[
    "第一段文本...",
    "第二段文本...",
    ...
]
```

## API 使用示例

```python
from batch_detect import DebateBatchProcessor

# 初始化处理器
processor = DebateBatchProcessor(
    model_name="gpt-4o-mini",
    temperature=1.0,
    sleep=1.0
)

# 加载数据
data, n_samples = processor.load_data(
    data_source="test",
    max_samples=10
)

# 运行批量检测
from pathlib import Path
batch_dir = Path("Results/test_batch")
processor.run_batch(data, batch_dir)

# 计算指标
metrics = processor.calculate_metrics(threshold=0.5)

# 保存汇总
config = {
    "data_source": "test",
    "model": "gpt-4o-mini"
}
processor.save_summary(batch_dir, config, metrics)
```

## 常见问题

### Q: 如何处理中文文本？
A: 系统已支持中文，无需额外配置。

### Q: 内存不足怎么办？
A: 使用 `--max-samples` 限制处理的样本数量。

### Q: 如何调整检测敏感度？
A: 使用 `--threshold` 参数，值越高越倾向于判定为人类文本。

### Q: 单文本模式结果保存在哪里？
A: 默认保存在 `Results/` 目录，文件名格式为 `<stem>_timestamp.json`。

## 更新日志

### v1.0 (2024-03-12)
- 添加测试集数据源支持
- 实现批量检测功能
- 扩展指标计算（ACC, F1）
- 实现结构化输出
- 支持命令行参数
