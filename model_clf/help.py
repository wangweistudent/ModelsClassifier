def get_help_text():
    return '''
ModelsClassifier 类帮助文档：

初始化方法：
    ModelsClassifier(model_name, X, y, data_name, test_size=0.2, random_seed=42, cv_n=5, device='cuda')
        - model_name: 模型名称(在 JSON 配置中定义)
        - X: 特征 DataFrame
        - y: 标签 Series
        - data_name: 用于命名输出的字符串
        - test_size: 测试集占比(默认 0.2)
        - random_seed: 随机种子(默认 42)
        - cv_n: 交叉验证折数(默认 5)
        - device: 设备选择('cuda' 或 'cpu')
        - X_test/y_test: 验证集/测试集,用于检验模型性能和绘制roc曲线

方法：
    train_model()
        - 执行模型训练、交叉验证、评估、保存结果至 self.results 并打印摘要

    shap_plot(plot_type='bar', shap_plot_file=None, force=False)
        - 生成 SHAP 可解释性图并保存 PDF/CSV/Pickle
        - 参数:plot_type 可选 'bar' 或 'dot'
        - shap_plot_file 可指定输出路径
        - force=True 时强制重新绘图

    search_best(save=False, model_file=None)
        - 执行超参数搜索，返回搜索器对象 self.searcher
        - 参数:save=True 时保存模型搜索器至文件

    compute_roc_auc_ci(y_true, y_scores, n_bootstraps=1000, alpha=0.95)
        - 计算 ROC AUC 的 bootstrap 置信区间
    
    roc_plot(roc_plot_file=None, force=False)
        - 绘制roc曲线
        - roc_plot_file 可指定输出路径
        - force=True 时强制重新绘图

    save_model(model_file=None)
        - 保存最优模型至 pickle 文件（默认命名）

    load_model(model_file)
        - 加载最优模型(覆盖 self.best_model)

    save_results(results_file=None)
        - 保存 self.results 至 pickle 文件

    load_results(results_file)
        - 加载指定 pickle 结果至 self.results
    
    print_result()
        - 输出上一轮训练的模型结果

说明：
    - 所有模型与参数通过 model_param_config.json 控制
    - 支持 GPU 加速的模型包括:LightGBM, XGBoost, CatBoost, TabPFN
    - 不支持加速的模型包括:Random_Forest, Logistic_Regression
    - SHAP 图支持树模型与线性模型
    - 若 feature_names 中含中文，请设置 matplotlib 字体支持
'''

