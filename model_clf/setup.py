from setuptools import setup, find_packages

setup(
    name="model_afs",
    version="0.1.0",
    author="wangw",
    description="A flexible classification training module with SHAP and ROC analysis",
    packages=find_packages(),  # 会自动找到包含 __init__.py 的目录
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "shap",
        "xgboost",
        "lightgbm",
        "catboost",
        "tabpfn @ git+https://github.com/automl/TabPFN.git",
        "tabpfn-extensions @ git+https://github.com/automl/TabPFN_extensions.git"
    ],
    python_requires='>=3.8',
)