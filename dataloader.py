"""
load data from where DNA-DetectLLM prepared
"""
import json
import os
from abc import ABC, abstractmethod
from typing import List, Tuple

# Dataset paths configuration
DATA_ROOT = "/data1/wujunxi/kailin/data"
DATASET_CONFIG = {
    "m4": {
        "root_dir": f"{DATA_ROOT}/M4",
        "human_file": "m4_human.json",
        "machine_file": "m4_machine.json"
    },
    "main": {
        "root_dir": f"{DATA_ROOT}/Collected data",
    },
    "detectrl": {
        "root_dir": f"{DATA_ROOT}/DetectRL"
    },
    "raid": {
        "root_dir": f"{DATA_ROOT}/RAID",
        "human_file": "raid_human.json",
        "machine_file": "raid_machine.json"
    },
    "realdet": {
        "root_dir": f"{DATA_ROOT}/RealDet",
        "human_file": "RealDet_human_test.json",
        "machine_file": "RealDet_machine_test.json"
    },
    "text_attack": {
        "root_dir": f"{DATA_ROOT}/Text_attack",
        "human_file": "human_texts.json"
    },
    "base": {
        "root_dir": f"{DATA_ROOT}/Base"
    },
    "test": {
        "root_dir": f"{DATA_ROOT}/Test",
        "human_file": "test_human.json",
        "machine_file": "test_machine.json"
    }
}

class DataSourceStrategy(ABC):
    """数据源加载策略抽象基类"""

    @abstractmethod
    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """加载数据"""
        pass

    @abstractmethod
    def get_available_domains(self) -> List[str]:
        """获取可用数据域"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """获取可用模型"""
        pass

class BaseJSONDataSource(DataSourceStrategy):
    """基于JSON文件的基类数据源加载策略"""

    def __init__(self, human_file: str, machine_file: str):
        self.human_file = human_file
        self.machine_file = machine_file

    def _load_json_file(self, file_path: str) -> List[str]:
        """加载JSON文件并提取文本列表"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取文本列表
        if "human_text" in data:
            return data["human_text"]
        elif "machine_text" in data:
            return data["machine_text"]
        else:
            # 如果是直接的文本列表
            return data if isinstance(data, list) else [data]

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """通用数据加载逻辑"""
        human_texts = self._load_json_file(self.human_file)
        ai_texts = self._load_json_file(self.machine_file)

        # 确保两个列表长度一致
        min_length = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_length]
        ai_texts = ai_texts[:min_length]

        # 限制样本数量
        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]

        return human_texts, ai_texts

class M4DataSource(BaseJSONDataSource):
    """M4数据集加载策略 - 使用已准备好的数据文件"""

    def __init__(self, data_root: str = DATASET_CONFIG['m4']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['m4']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['m4']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class DetectRLDataSource(BaseJSONDataSource):
    """DetectRL数据集加载策略"""

    def __init__(self, data_root: str = DATASET_CONFIG['detectrl']['root_dir'], dataset_type: str = "multidomain"):
        if dataset_type not in ["multidomain", "multillm"]:
            raise ValueError(f"dataset_type must be 'multidomain' or 'multillm', got: {dataset_type}")

        self.dataset_type = dataset_type
        human_file = os.path.join(data_root, f"DetectRL_{dataset_type}_human_test.json")
        machine_file = os.path.join(data_root, f"DetectRL_{dataset_type}_machine_test.json")
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return [self.dataset_type]

    def get_available_models(self) -> List[str]:
        return []

class RAIDDataSource(BaseJSONDataSource):
    """RAID数据集加载策略 - 使用已准备好的数据文件"""

    def __init__(self, data_root: str = DATASET_CONFIG['raid']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['raid']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['raid']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class RealDetDataSource(BaseJSONDataSource):
    """RealDet数据集加载策略"""

    def __init__(self, data_root: str = DATASET_CONFIG['realdet']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['realdet']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['realdet']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["mixed"]

class TextAttackDataSource(BaseJSONDataSource):
    """Text Attack数据集加载策略 - 支持多种攻击类型"""

    def __init__(self, data_root: str = DATASET_CONFIG['text_attack']['root_dir']):
        self.data_root = data_root
        self.human_file = os.path.join(data_root, DATASET_CONFIG['text_attack']['human_file'])
        self.models = ["Claude", "Gemini", "GPT4"]
        self.attack_types = ["delete", "dipper", "insert", "replace"]

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """加载Text Attack数据"""
        model = kwargs.get("model", "GPT4")
        attack_type = kwargs.get("attack_type", "delete")

        if model not in self.models:
            raise ValueError(f"Unsupported model: {model}. Available: {self.models}")
        if attack_type not in self.attack_types:
            raise ValueError(f"Unsupported attack type: {attack_type}. Available: {self.attack_types}")

        # 构造机器生成文件路径
        machine_file = os.path.join(self.data_root, f"{model}_machine_test_{attack_type}.json")

        human_texts = self._load_json_file(self.human_file)
        ai_texts = self._load_json_file(machine_file)

        # 确保两个列表长度一致
        min_length = min(len(human_texts), len(ai_texts))
        human_texts = human_texts[:min_length]
        ai_texts = ai_texts[:min_length]

        # 限制样本数量
        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]

        return human_texts, ai_texts

    def get_available_domains(self) -> List[str]:
        """获取可用的攻击类型"""
        return self.attack_types

    def get_available_models(self) -> List[str]:
        """获取可用的模型"""
        return self.models

class MainDataSource(BaseJSONDataSource):
    """主要数据集加载策略"""

    def __init__(self, data_root: str = DATASET_CONFIG['main']['root_dir']):
        self.data_root = data_root
        self.datasets = ["xsum", "wp", "arxiv"]
        self.source_models = ["gpt4o", "claude3.7", "gemini2.0"]

    def _load_json_file(self, file_path: str, max_samples: int = -1):
        """加载JSON文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """主要数据集加载"""
        dataset = kwargs.get("dataset", "xsum")
        source_model = kwargs.get("source_model", "gpt4o")

        human_file = os.path.join(self.data_root, dataset, f"{dataset}_human.json")
        ai_file = os.path.join(self.data_root, dataset, f"{dataset}_{source_model}.json")

        human_data = self._load_json_file(human_file)
        ai_data = self._load_json_file(ai_file)

        human_texts = human_data.get("human_text", human_data)
        ai_texts = ai_data.get("machine_text", ai_data)

        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]
        # debug for lastde
        # human_texts = human_texts[205:210]
        # ai_texts = ai_texts[205:210]
        return human_texts, ai_texts

    def get_available_domains(self) -> List[str]:
        return self.datasets

    def get_available_models(self) -> List[str]:
        return self.source_models

class BaseDataSource(BaseJSONDataSource):
    """Base模型数据集（来自fast-detect-gpt）加载策略"""
    def __init__(self, data_root: str = DATASET_CONFIG['base']['root_dir']):
        self.data_root = data_root
        self.datasets = ["xsum", "writing", "pubmed", "squad"]
        self.source_models = ["davinci", "gpt-4", "gpt-j-6B", "gpt2-xl"]
        
    def _load_json_file(self, file_path: str, max_samples: int = -1):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def load(self, max_samples: int = -1, **kwargs):
        """Base数据集加载"""
        dataset = kwargs.get("dataset", "xsum")
        source_model = kwargs.get("source_model", "gpt-j-6B")
        
        file_path = os.path.join(self.data_root, dataset, f"{dataset}_{source_model}.raw_data.json")
        if not os.path.exists(file_path):
            print(f"[INFO] file path {file_path} not exist as expected")
            return [], []

        data = self._load_json_file(file_path)
        human_texts = data['original']
        ai_texts = data['sampled']
        
        if max_samples > 0:
            human_texts = human_texts[:max_samples]
            ai_texts = ai_texts[:max_samples]
        return human_texts, ai_texts
    
    def get_available_domains(self) -> List[str]:
        return self.datasets

    def get_available_models(self) -> List[str]:
        return self.source_models

class TestDataSource(BaseJSONDataSource):
    """测试数据集加载策略 - 用于小样本快速测试"""

    def __init__(self, data_root: str = DATASET_CONFIG['test']['root_dir']):
        human_file = os.path.join(data_root, DATASET_CONFIG['test']['human_file'])
        machine_file = os.path.join(data_root, DATASET_CONFIG['test']['machine_file'])
        super().__init__(human_file, machine_file)

    def get_available_domains(self) -> List[str]:
        return []

    def get_available_models(self) -> List[str]:
        return ["test"]

class DataLoader:
    """统一数据加载器"""

    # 数据源策略映射
    _strategies = {
        "m4": M4DataSource,
        "main": MainDataSource,
        "detectrl_multidomain": lambda **kwargs: DetectRLDataSource(dataset_type="multidomain", **kwargs),
        "detectrl_multillm": lambda **kwargs: DetectRLDataSource(dataset_type="multillm", **kwargs),
        "raid": RAIDDataSource,
        "text_attack": TextAttackDataSource,
        "realdet": RealDetDataSource,
        "base": BaseDataSource,
        "test": TestDataSource,
    }

    def __init__(self, data_source: str = "main", **kwargs):
        """
        初始化数据加载器

        Args:
            data_source: 数据源类型 ("m4", "main", "detectrl_multidomain", "detectrl_multillm", "realdet")
            **kwargs: 传递给数据源策略的参数
        """
        if data_source not in self._strategies:
            raise ValueError(f"Unsupported data source: {data_source}. Available: {list(self._strategies.keys())}")

        self.data_source = data_source
        strategy_class = self._strategies[data_source]

        # 处理lambda函数
        if callable(strategy_class) and not isinstance(strategy_class, type):
            self.strategy = strategy_class(**kwargs)
        else:
            self.strategy = strategy_class(**kwargs)

    def load(self, max_samples: int = -1, **kwargs) -> Tuple[List[str], List[str]]:
        """
        加载数据

        Args:
            max_samples: 最大样本数，-1表示全部
            **kwargs: 其他参数

        Returns:
            (人类文本列表, AI文本列表)
        """
        human_texts, ai_texts = self.strategy.load(max_samples=max_samples, **kwargs)
        return data_wrapper(human_texts, ai_texts)

    def get_available_domains(self) -> List[str]:
        """获取可用数据域"""
        return self.strategy.get_available_domains()

    def get_available_models(self) -> List[str]:
        """获取可用模型"""
        return self.strategy.get_available_models()

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """注册新的数据源策略"""
        cls._strategies[name] = strategy_class


# 快捷函数
def load_m4_data(max_samples: int = 1000, **kwargs):
    """M4数据加载便捷函数"""
    loader = DataLoader("m4")
    return loader.load(max_samples=max_samples, **kwargs)

def load_detectrl_data(dataset_type: str = "multidomain", max_samples: int = -1, **kwargs):
    """DetectRL数据加载便捷函数"""
    loader = DataLoader(f"detectrl_{dataset_type}")
    return loader.load(max_samples=max_samples, **kwargs)

def load_raid_data(max_samples: int = 1000, **kwargs):
    """RAID"""
    loader = DataLoader("raid")
    return loader.load(max_samples=max_samples, **kwargs)

def load_realdet_data(max_samples: int = -1, **kwargs):
    """RealDet数据加载便捷函数"""
    loader = DataLoader("realdet")
    return loader.load(max_samples=max_samples, **kwargs)

def load_main_data(dataset: str, source_model: str, max_samples: int = -1):
    """主要数据集加载便捷函数"""
    loader = DataLoader("main")
    return loader.load(dataset=dataset, source_model=source_model, max_samples=max_samples)

def load_text_attack_data(model: str = "GPT4", attack_type: str = "delete", max_samples: int = -1):
    """Text Attack数据加载便捷函数"""
    loader = DataLoader("text_attack")
    return loader.load(model=model, attack_type=attack_type, max_samples=max_samples)

def load_base_data(dataset: str, source_model: str, max_samples: int = -1):
    loader = DataLoader("base")
    return loader.load(dataset=dataset, source_model=source_model, max_samples=max_samples)

def load_test_data(max_samples: int = -1, **kwargs):
    """测试数据加载便捷函数"""
    loader = DataLoader("test")
    return loader.load(max_samples=max_samples, **kwargs)

def data_wrapper(human_texts: List, ai_texts: List):
    """Wrap human and AI texts into standard format"""
    data = {
        "original": human_texts,
        "sampled": ai_texts
    }
    n_samples = len(data["sampled"])
    return data, n_samples

def truncate_data_dict(data: dict, max_words: int) -> dict:
    """
    Truncate texts in data dictionary to max_words

    Args:
        data: Dictionary with 'original' and 'sampled' keys containing text lists
        max_words: Maximum number of words to keep in each text

    Returns:
        dict: Data dictionary with truncated texts
    """
    def truncate_text(text: str, max_words: int) -> str:
        """Truncate a single text to max_words"""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words])

    truncated_data = {
        "original": [truncate_text(text, max_words) for text in data.get("original", [])],
        "sampled": [truncate_text(text, max_words) for text in data.get("sampled", [])]
    }
    return truncated_data


def load_data(args):
    if args.data_source == 'main':
        data, n_samples = load_main_data(args.dataset, args.source_model, args.max_samples)
    elif args.data_source == 'm4':
        data, n_samples = load_m4_data(max_samples=args.max_samples)
    elif args.data_source in ['detectrl_multidomain', 'detectrl_multillm']:
        dataset_type = args.data_source.replace('detectrl_', '')
        data, n_samples = load_detectrl_data(dataset_type=dataset_type, max_samples=args.max_samples)
    elif args.data_source == 'raid':
        data, n_samples = load_raid_data(max_samples=args.max_samples)
    elif args.data_source == 'realdet':
        data, n_samples = load_realdet_data(max_samples=args.max_samples)
    elif args.data_source == 'text_attack':
        # Map source_model to Text_attack model names
        model_mapping = {
            'claude3.7': 'Claude',
            'gemini2.0': 'Gemini',
            'gpt4o': 'GPT4'
        }
        attack_model = model_mapping.get(args.source_model, 'GPT4')
        data, n_samples = load_text_attack_data(
            model=attack_model,
            attack_type=args.attack_type,
            max_samples=args.max_samples
        )
    elif args.data_source == 'base':
        data, n_samples = load_base_data(args.base_dataset, args.base_source_model, args.max_samples)
    elif args.data_source == 'test':
        data, n_samples = load_test_data(max_samples=args.max_samples)

    # Apply text truncation if max_words is specified
    if hasattr(args, 'max_words') and args.max_words is not None:
        data = truncate_data_dict(data, args.max_words)
        print(f"[INFO] Texts truncated to {args.max_words} words for ablation study")

    return data, n_samples