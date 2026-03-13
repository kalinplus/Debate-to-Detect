from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, accuracy_score, f1_score
import numpy as np

# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

def evaluate(human_scores, ai_scores):
    scores = human_scores + ai_scores
    # Use consistent labeling: human=0, ai=1 to match get_roc_metrics and get_precision_recall_metrics
    labels = [0] * len(human_scores) + [1] * len(ai_scores)

    auroc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    best_f1_threshold = thresholds[best_f1_index]
    return auroc, best_f1, best_f1_threshold

def get_roc_metrics(real_preds, sample_preds):
    # y_true: 0 for real (human) text, 1 for sampled (AI) text
    # y_score: prediction scores for each sample
    print('real_preds:', len(real_preds), 'sample_preds:', len(sample_preds))
    print('y_true 中 1 的个数:', np.sum([0]*len(real_preds) + [1]*len(sample_preds)))
    y_true = [0] * len(real_preds) + [1] * len(sample_preds)
    y_score = real_preds + sample_preds
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    # y_true: 0 for real (human) text, 1 for sampled (AI) text
    # y_score: prediction scores for each sample
    y_true = [0] * len(real_preds) + [1] * len(sample_preds)
    y_score = real_preds + sample_preds
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)

def calculate_accuracy(y_true, y_pred, threshold=0.5):
    """
    计算准确率

    Args:
        y_true: 真实标签列表 (0=human, 1=AI)
        y_pred: 预测分数列表
        threshold: 判定阈值，默认0.5

    Returns:
        float: 准确率
    """
    y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
    return accuracy_score(y_true, y_pred_binary)

def calculate_f1(y_true, y_pred, threshold=0.5):
    """
    计算 F1 分数

    Args:
        y_true: 真实标签列表 (0=human, 1=AI)
        y_pred: 预测分数列表
        threshold: 判定阈值，默认0.5

    Returns:
        float: F1 分数
    """
    y_pred_binary = [1 if score >= threshold else 0 for score in y_pred]
    return f1_score(y_true, y_pred_binary)

def evaluate_all(human_scores, ai_scores, threshold=0.5):
    """
    综合评估：返回 AUROC, F1, Accuracy

    Args:
        human_scores: 人类文本的预测分数列表
        ai_scores: AI文本的预测分数列表
        threshold: 判定阈值，默认0.5

    Returns:
        dict: 包含所有指标的字典
    """
    scores = human_scores + ai_scores
    labels = [0] * len(human_scores) + [1] * len(ai_scores)

    # 计算 AUROC
    auroc = roc_auc_score(labels, scores)

    # 计算 Accuracy
    accuracy = calculate_accuracy(labels, scores, threshold)

    # 计算 F1
    f1 = calculate_f1(labels, scores, threshold)

    # 计算最佳 F1 和对应阈值
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    best_f1_threshold = thresholds[best_f1_index] if best_f1_index < len(thresholds) else 1.0

    return {
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(best_f1_threshold),
        "threshold": threshold
    }

def verdict_to_label(verdict: str) -> int:
    """
    将 verdict 映射为二分类标签

    Args:
        verdict: "AI_GENERATED", "HUMAN_WRITTEN", 或 "UNCERTAIN"

    Returns:
        int: 1 表示 AI 生成，0 表示人类编写
    """
    verdict_map = {
        "AI_GENERATED": 1,
        "HUMAN_WRITTEN": 0,
        "UNCERTAIN": 0.5  # 可配置，这里使用 0.5 作为不确定值
    }
    return verdict_map.get(verdict, 0.5)

def verdict_to_score(verdict: str, scores: dict = None) -> float:
    """
    将 verdict 转换为数值分数（用于计算 AUROC）

    Args:
        verdict: "AI_GENERATED", "HUMAN_WRITTEN", 或 "UNCERTAIN"
        scores: 可选的原始分数字典 {"Affirmative": X, "Negative": Y}

    Returns:
        float: 0-1 之间的分数，越高越可能是 AI 生成
    """
    if verdict == "AI_GENERATED":
        return 1.0
    elif verdict == "HUMAN_WRITTEN":
        return 0.0
    elif verdict == "UNCERTAIN":
        # 如果有原始分数，使用分数比例
        if scores:
            total = scores.get("Affirmative", 0) + scores.get("Negative", 0)
            if total > 0:
                return scores.get("Affirmative", 0) / total
        return 0.5
    return 0.5
