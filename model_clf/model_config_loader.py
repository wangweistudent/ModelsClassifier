import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import os

def get_model_param_grid(config_path='model_param_config.json', device='cuda',seed = 42):
    """
    加载模型参数配置，并返回模型对象 + 参数网格字典

    参数:
        config_path (str): JSON 配置路径（默认当前目录下）
        use_gpu (bool): 是否使用 GPU 加速的模型版本

    返回:
        dict: {model_name: (model_object, param_grid)}
    """
    use_gpu = device.lower() in ['cuda', 'gpu']
    
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),config_path)
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"❌ 参数文件不存在: {config_path}")

    with open(config_file, 'r') as f:
        param_config = json.load(f)

    model_param_grid = {}

    for model_name in param_config.keys():
        if model_name == "Random_Forest":
            model = RandomForestClassifier(random_state=seed)
        elif model_name == "Logistic_Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "LightGBM":
            model = LGBMClassifier(
                verbose=-1,
                random_state=seed,
                device="gpu" if use_gpu else "cpu"
            )
        elif model_name == "CatBoost":
            model = CatBoostClassifier(
                task_type="GPU" if use_gpu else "CPU",
                verbose=0,
                random_state=seed
            )
        elif model_name == "XGBoost":
            model = XGBClassifier(
                tree_method='hist',
                device="cuda" if use_gpu else "cpu",
                # predictor='gpu_predictor', 
                # use_label_encoder=False,
                eval_metric="logloss",
                random_state=seed
            )
        elif model_name == "tabpfn":
            model = TabPFNClassifier(
                device="cuda" if use_gpu else "cpu",
                ignore_pretraining_limits=True
            )
        else:
            raise ValueError(f"❌ 不支持的模型名: {model_name}")

        param_grid = param_config.get(model_name, {})
        model_param_grid[model_name] = (model, param_grid)

    return model_param_grid
