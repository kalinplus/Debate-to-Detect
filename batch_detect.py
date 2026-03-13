"""
Batch detection script for processing multiple texts
"""
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import traceback

from engine import Debate
from dataloader import load_data


# Verdict constants for type safety
VERDICT_AI_GENERATED = "AI_GENERATED"
VERDICT_HUMAN_WRITTEN = "HUMAN_WRITTEN"
VERDICT_UNCERTAIN = "UNCERTAIN"

# Label constants
LABEL_HUMAN = "human"
LABEL_AI = "ai"


class DebateBatchProcessor:
    """批量处理辩论检测"""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 1.0, sleep: float = 1.0):
        """
        初始化批量处理器

        Args:
            model_name: 使用的模型名称
            temperature: 温度参数
            sleep: API调用间隔时间（秒）
        """
        self.model_name = model_name
        self.temperature = temperature
        self.sleep = sleep
        self.results: List[Dict] = []

    def load_data(self, data_source: str, max_samples: int = -1, **kwargs) -> Tuple[dict, int]:
        """
        加载数据

        Args:
            data_source: 数据源类型
            max_samples: 最大样本数
            **kwargs: 其他参数

        Returns:
            (data, n_samples) 数据字典和样本数量
        """
        print(f"[INFO] Loading data from {data_source}...")

        # 创建临时参数对象
        class Args:
            def __init__(self, data_source, max_samples, **kwargs):
                self.data_source = data_source
                self.max_samples = max_samples
                self.__dict__.update(kwargs)

        args = Args(data_source, max_samples, **kwargs)
        data, n_samples = load_data(args)

        print(f"[INFO] Loaded {n_samples} samples")
        print(f"[INFO] Human texts: {len(data['original'])}, AI texts: {len(data['sampled'])}")

        return data, n_samples

    def run_batch(self, data: dict, batch_dir: Path):
        """
        批量运行检测

        Args:
            data: 数据字典，包含 'original' 和 'sampled'
            batch_dir: 批次输出目录
        """
        print(f"\n[INFO] Starting batch detection...")
        print(f"[INFO] Output directory: {batch_dir}")

        # 创建单个结果目录
        individual_dir = batch_dir / "individual_results"
        individual_dir.mkdir(parents=True, exist_ok=True)

        # 处理人类文本
        print(f"\n[INFO] Processing {len(data['original'])} human texts...")
        for idx, text in enumerate(data['original']):
            self._process_single(text, idx, "human", individual_dir)

        # 处理AI文本
        print(f"\n[INFO] Processing {len(data['sampled'])} AI texts...")
        for idx, text in enumerate(data['sampled']):
            self._process_single(text, idx, "ai", individual_dir)

        print(f"\n[INFO] Batch detection completed!")

    def _process_single(self, text: str, idx: int, label: str, output_dir: Path):
        """
        处理单个文本

        Args:
            text: 待检测文本
            idx: 索引
            label: 真实标签 ("human" 或 "ai")
            output_dir: 输出目录
        """
        try:
            # 初始化新的 debate 实例
            debate = Debate(model_name=self.model_name, T=self.temperature, sleep=self.sleep)

            # 生成输出文件路径 (必须在调用run之前创建)
            filename = f"{label}_{idx:04d}.json"
            output_path = output_dir / filename

            # 运行检测 (传入output_path参数)
            result = debate.run(news_text=text, output_path=output_path)

            # 添加标签信息
            result['true_label'] = label
            result['index'] = idx

            # 保存单个结果
            self._save_individual_result(result, output_path)

            # 保存到结果列表
            self.results.append(result)

            print(f"[OK] {label}_{idx:04d} | Verdict: {result['verdict']} | Score: {result['detection_score']:.3f}")

        except Exception as e:
            print(f"[ERROR] Failed to process {label}_{idx:04d}: {str(e)}")
            traceback.print_exc()

    def _save_individual_result(self, result: Dict, output_path: Path):
        """保存单个结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _verdict_to_pred_label(self, verdict: str) -> Optional[str]:
        """
        将verdict转换为预测标签

        Args:
            verdict: 判定结果 (AI_GENERATED, HUMAN_WRITTEN, UNCERTAIN)

        Returns:
            "ai", "human", 或 None (UNCERTAIN)
        """
        if verdict == VERDICT_AI_GENERATED:
            return LABEL_AI
        elif verdict == VERDICT_HUMAN_WRITTEN:
            return LABEL_HUMAN
        else:
            return None

    def calculate_metrics(self) -> Dict:
        """
        计算评估指标（基于verdict直接判定）

        Returns:
            指标字典
        """
        print(f"\n[INFO] Calculating metrics based on verdicts...")

        if not self.results:
            print("[WARNING] No results for metrics calculation")
            return {}

        # 统计混淆矩阵
        tp = fp = tn = fn = 0

        for result in self.results:
            true_label = result['true_label']
            pred_label = self._verdict_to_pred_label(result['verdict'])

            if pred_label == LABEL_AI:
                if true_label == LABEL_AI:
                    tp += 1
                else:
                    fp += 1
            elif pred_label == LABEL_HUMAN:
                if true_label == LABEL_HUMAN:
                    tn += 1
                else:
                    fn += 1

        # 计算指标
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0

        # 避免除零
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

        print(f"\n[INFO] Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        return metrics

    def save_summary(self, batch_dir: Path, config: Dict, metrics: Dict):
        """
        保存汇总结果

        Args:
            batch_dir: 批次目录
            config: 配置信息
            metrics: 评估指标
        """
        # 保存配置
        config_path = batch_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 保存指标
        if metrics:
            metrics_path = batch_dir / "metrics.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

        # 保存总结报告
        summary_path = batch_dir / "summary_report.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Debate-to-Detect Batch Detection Report\n")
            f.write("=" * 60 + "\n\n")

            f.write("Configuration:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Results Summary:\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total samples processed: {len(self.results)}\n")

            # 统计verdict分布
            verdict_counts = {}
            for result in self.results:
                verdict = result['verdict']
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

            f.write("\nVerdict Distribution:\n")
            for verdict, count in verdict_counts.items():
                f.write(f"  {verdict}: {count}\n")

            # 统计正确率
            correct = 0
            for result in self.results:
                true_label = result['true_label']
                verdict = result['verdict']
                if true_label == 'ai' and verdict == 'AI_GENERATED':
                    correct += 1
                elif true_label == 'human' and verdict == 'HUMAN_WRITTEN':
                    correct += 1

            accuracy = correct / len(self.results) if self.results else 0
            f.write(f"\nCorrect predictions: {correct}/{len(self.results)} ({accuracy:.2%})\n")

            if metrics:
                f.write("\n" + "=" * 60 + "\n")
                f.write("Evaluation Metrics:\n")
                f.write("=" * 60 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")

        print(f"\n[INFO] Summary saved to {batch_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Batch AI text detection using Debate-to-Detect")

    # 数据参数
    parser.add_argument("--data-source", type=str, default="test",
                       choices=["test"],
                    #    choices=["main", "m4", "detectrl_multidomain", "detectrl_multillm", "raid", "text_attack", "realdet", "base", "test"],
                       help="Data source to use")
    parser.add_argument("--max-samples", type=int, default=-1,
                       help="Maximum number of samples to process (default: all)")

    # # Main数据源特定参数
    # parser.add_argument("--dataset", type=str, default="xsum",
    #                    help="Dataset name for main data source")
    # parser.add_argument("--source-model", type=str, default="gpt4o",
    #                    help="Source model for main data source")

    # # Text Attack数据源特定参数
    # parser.add_argument("--attack-type", type=str, default="delete",
    #                    choices=["delete", "dipper", "insert", "replace"],
    #                    help="Attack type for text_attack data source")

    # # Base数据源特定参数
    # parser.add_argument("--base-dataset", type=str, default="xsum",
    #                    help="Dataset name for base data source")
    # parser.add_argument("--base-source-model", type=str, default="gpt-j-6B",
    #                    help="Source model for base data source")

    # 模型参数
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                       help="Model to use for detection")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for generation")
    parser.add_argument("--sleep", type=float, default=1.0,
                       help="Sleep time between API calls (seconds)")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default="Results",
                       help="Base output directory")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建批次目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.output_dir) / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Debate-to-Detect Batch Detection")
    print("=" * 60)
    print(f"Data source: {args.data_source}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max samples: {args.max_samples if args.max_samples > 0 else 'All'}")
    print("=" * 60)

    # 初始化处理器
    processor = DebateBatchProcessor(
        model_name=args.model,
        temperature=args.temperature,
        sleep=args.sleep
    )

    # 准备配置参数
    load_kwargs = {}
    if args.data_source == 'main':
        load_kwargs['dataset'] = args.dataset
        load_kwargs['source_model'] = args.source_model
    elif args.data_source == 'text_attack':
        load_kwargs['attack_type'] = args.attack_type
        load_kwargs['source_model'] = args.source_model
    elif args.data_source == 'base':
        load_kwargs['dataset'] = args.base_dataset
        load_kwargs['source_model'] = args.base_source_model

    # 加载数据
    data, n_samples = processor.load_data(args.data_source, args.max_samples, **load_kwargs)

    # 运行批量检测
    processor.run_batch(data, batch_dir)

    # 计算指标
    metrics = processor.calculate_metrics()

    # 保存汇总
    config = {
        "data_source": args.data_source,
        "model": args.model,
        "temperature": args.temperature,
        "max_samples": args.max_samples if args.max_samples > 0 else n_samples,
        "actual_samples": len(processor.results),
        "load_kwargs": load_kwargs
    }

    processor.save_summary(batch_dir, config, metrics)

    print(f"\n{'=' * 60}")
    print("Batch detection completed!")
    print(f"Results saved to: {batch_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
