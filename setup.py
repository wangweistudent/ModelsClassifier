from setuptools import setup, find_packages

setup(
    name="model_clf",
    version="0.1.1",
    author="wangwei",
    email="wangwei@siat.ac.cn",
    description="A flexible classification training module with SHAP and ROC analysis",
    packages=find_packages(), 
    include_package_data=True,
    install_requires=[
         "tqdm",
        "jupyter",
        "ipywidgets",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "shap",
        "xgboost",
        "lightgbm",
        "catboost"
    ],
    python_requires='>=3.8',
)