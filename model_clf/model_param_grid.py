import os
os.environ["SCIPY_ARRAY_API"] = "1"
os.environ['CATBOOST_GPU_MEMORY_PART'] = '0.8'
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve,confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split,RandomizedSearchCV
import pandas as pd
import shap
import matplotlib.pyplot as plt
from .model_config_loader import get_model_param_grid
from .help import get_help_text
from .utils import *
plt.rcParams['font.family'] = 'Arial'
class ModelsClassifier:
    def __init__(self, model_name, X, y,data_name,test_size=0.2, random_seed=42,cv_n=5,device = 'cuda'):
        self.help = get_help_text
        self.device = device
        self.random_seed = random_seed
        model_param_grid = get_model_param_grid(device=self.device,seed= self.random_seed)
        self.model_name = model_name
        self.model = model_param_grid[model_name][0]
        self.X = X
        self.y = y
        self.param_grid = model_param_grid[model_name][1]
        self.searcher = None
        self.best_model = None
        self.results = {}
        self.cv_n = cv_n
        self.test_size = test_size
        self.data_name = data_name
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,                # 特征矩阵和标签
            test_size=self.test_size,       # 测试集占比（也可用 train_size）
            random_state=self.random_seed,     # 随机种子，保证可复现
            stratify=self.y           # （可选）按类别分层抽样，分类任务常用
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def compute_roc_auc_ci(self,y_true, y_scores, n_bootstraps=1000, alpha=0.95):
        rng = np.random.RandomState(self.random_seed)
        bootstrapped_scores = []
        # y_true = y_true.to_numpy() if isinstance(y_true, pd.Series) else y_true
        y_true = np.array(y_true)
        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_scores), len(y_scores))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_scores[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
        upper = sorted_scores[int((1.0 + alpha) / 2 * len(sorted_scores))]
        return lower, upper

    def search_best(self, save = False, model_file = None ):
        model_name = self.model_name
        param_grid = self.param_grid
        cv_n = self.cv_n
        model = self.model
        random_seed = self.random_seed
        if param_grid:
            print('Start RandomizedSearchCV:',model_name)
            searcher = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=30, 
                n_jobs=1, # 可调
                scoring='roc_auc',
                cv=StratifiedKFold(n_splits=cv_n, shuffle=True, random_state=random_seed),
                verbose=1,
                random_state=42
                )
            searcher.fit(self.X_train,self.y_train)
            self.searcher = searcher
            if save or model_file:
                if not model_file:
                    model_file = f'{model_name}_best_model.pkl'
                save_pickle(self.searcher,model_name)
        else:
            print(model_name,'has not param can searcher ! 😨')
            self.model.fit(self.X_train,self.y_train)
            return self.model
        return self.searcher

    def train_model(self):
        result = {}
        self.searcher = self.search_best()
        if hasattr(self.searcher, "best_estimator_"):
            self.best_model = self.searcher.best_estimator_
            print("✅ 最优参数组合：", self.searcher.best_params_)
            print("📈 最优 AUC:", self.searcher.best_score_)
        else:
            self.best_model = self.searcher
            print(f"✅ 使用默认模型 {self.model_name}，无参数搜索。")
        # best_score = roc_auc_score(self.y_test, self.best_model.predict_proba(self.X_test)[:, 1])   
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        acc = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_proba)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        ci_lower, ci_upper = self.compute_roc_auc_ci(self.y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        conf_matrix = {
                    "TP": int(tp),
                    "TN": int(tn),
                    "FP": int(fp),
                    "FN": int(fn)
                    }
        result['X_train']=self.X_train
        result['X_test']=self.X_test
        result['y_train']=self.y_train
        result['y_test']=self.y_test
        # result['best_score']=best_score
        result['searcher']=self.searcher
        result['best_model']=self.best_model
        result['y_proba']=y_proba
        result['y_pred']=y_pred
        result['acc']=acc
        result['precision']=precision
        result['recall']=recall
        result['f1']=f1
        result['auc']=auc
        result['fpr']=fpr
        result['tpr']=tpr
        result['acc']=acc
        result['auc_ci']=(ci_lower, ci_upper)
        result['confusion_matrix']=conf_matrix
        result['X']=self.X
        result['y']=self.y
        self.results = result
        print(f"✅ {self.model_name} 最佳ROC AUC: {auc:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
        print(f"   🎯 ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"   📊 混淆矩阵 (Confusion Matrix):")
        print(f"   {'':<8}{'Pred 0':<10}{'Pred 1':<10}")
        print(f"   {'True 0':<8}{tn:<10}{fp:<10}")
        print(f"   {'True 1':<8}{fn:<10}{tp:<10}")
        return self.results
    
    def print_result(self):
        result=self.results 
        conf_matrix = result['confusion_matrix']
        tp = conf_matrix['TP']
        tn = conf_matrix['TN']
        fp = conf_matrix['FP']
        fn = conf_matrix['FN']
        acc =result['acc']
        auc=result['auc']
        (ci_lower, ci_upper)=result['auc_ci']
        precision =result['precision']
        recall=result['recall']
        f1=result['f1']
        print(f"✅ {self.model_name} 最佳ROC AUC: {auc:.4f} (95% CI: {ci_lower:.4f} - {ci_upper:.4f})")
        print(f"   🎯 ACC: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"   📊 混淆矩阵 (Confusion Matrix):")
        print(f"   {'':<8}{'Pred 0':<10}{'Pred 1':<10}")
        print(f"   {'True 0':<8}{tn:<10}{fp:<10}")
        print(f"   {'True 1':<8}{fn:<10}{tp:<10}")
        
    def shap_plot(self,plot_type='bar',shap_plot_file=None,force =False,outdir='./'):
        ensure_dir(outdir, exist_ok=True)
        print('X_test type is:',type(self.X_test),type(self.X_test) == pd.core.frame.DataFrame)
        if type(self.X_test) == pd.core.frame.DataFrame:
            feature_names = self.X_test.columns.to_list()
        else:
            feature_names = [f'feature_{i}' for i in range(self.X_test.shape[1])]
        self.feature_names = feature_names
        x = pd.DataFrame(self.X_test, columns=self.feature_names).apply(pd.to_numeric, errors='coerce').astype(float)
        
        try:
            if isinstance(self.best_model, (RandomForestClassifier, LGBMClassifier, XGBClassifier, CatBoostClassifier)):
                explainer = shap.TreeExplainer(self.best_model, feature_names=feature_names)
                shap_values = explainer(x)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]  # 默认选择 class 1
                else:
                    shap_vals = shap_values.values if hasattr(shap_values, 'values') else shap_values
            else :
                explainer = shap.LinearExplainer(self.best_model, x, feature_perturbation="interventional")
                shap_values = explainer(x)
                shap_vals = shap_values.values

                # 多分类情况
            if shap_vals.ndim == 3:
                shap_vals =shap_vals[:, :,1] # 选中 class 1
            shap_mean = np.abs(shap_vals).mean(axis=0)
            feature_num = [f'Feature_{i}' for i,j in enumerate(feature_names)]
            # feature_names = [ENSEMBL_DICT.get(eid, eid) for eid in ensembl_ids]
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Mean SHAP Value': shap_mean,
                'Feature_num':feature_num
            }).sort_values(by='Mean SHAP Value', ascending=False).reset_index(drop=True)
            save_pickle(feature_importance,f'{self.data_name}_{self.model_name}.feature_importance.pickle')
            pd.DataFrame(feature_importance).to_csv(f'{self.data_name}_{self.model_name}.feature_importance.csv')
            # 保存 SHAP summary plot
            if not shap_plot_file:
                shap_plot_file= f"{self.model_name}_{self.data_name}_shap_summary.pdf"
            shap_plot_file = os.path.join(outdir, f"{self.model_name}_{self.data_name}_shap_summary.pdf")
            save_pickle(shap_values,shap_plot_file.replace('.pdf','.shap_values.pickle'))
            if not os.path.exists(shap_plot_file) or force:
                plt.figure()
                if is_chinese(feature_names):
                    print('There are chinese characters in feature_names!')
                    shap.summary_plot(shap_vals, x, feature_names=feature_num, plot_type=plot_type, show=False)
                else:
                    shap.summary_plot(shap_vals, x, feature_names=feature_names, plot_type=plot_type, show=False)
                plt.title(f'SHAP Summary {self.data_name}  {self.data_name}')
                plt.tight_layout()
                plt.savefig(shap_plot_file)
                plt.close()
            print(f"🧠 已保存 SHAP summary plot 至: {shap_plot_file}")
            feature_importance.to_csv(shap_plot_file.replace('.pdf','.csv'))
        except Exception as e:
            print(f"⚠️ 无法为模型 {self.model_name} 生成 SHAP 图: {e}")
    
    def roc_plot(self, roc_plot_file=None,force=False,outdir='./'):
        """
        绘制 ROC 曲线并保存。
        """
        if self.best_model is None:
            raise ValueError("❌ best_model 为 None，请先训练模型或加载结果。")

        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc_score = roc_auc_score(self.y_test, y_proba)
        if not roc_plot_file:
            roc_plot_file = f"{self.model_name}_{self.data_name}_roc_curve.pdf"
        roc_plot_file = os.path.join(outdir,roc_plot_file)
        if not os.path.exists(roc_plot_file) or force:
            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {self.model_name} {self.data_name}")
            plt.legend(loc="lower right")
            plt.grid(True)
            
            plt.savefig(roc_plot_file)
            plt.close()
            print(f"📉 ROC 曲线已保存至: {roc_plot_file}")
        else:
            print(f"📉 ROC 曲线已存在于: {roc_plot_file}")
        


    def save_model(self, model_file = None):
        if not model_file:
            model_file = f'{self.model_name}_best_model.pkl'
        save_pickle(self.best_model, model_file)
        print(f"✅ 模型已保存至: {model_file}")

    def load_model(self, model_file):
        self.best_model = load_pickle(model_file)
        print(f"📦 模型已从 {model_file} 加载。")
        return self.best_model
    
    def save_results(self, results_file = None):
        if not results_file:
            results_file = f'{self.model_name}_results.pkl'
        save_pickle(self.results, results_file)

        print(f"✅ results已保存至: {results_file}")

    def load_results(self, results_file):
        self.results = load_pickle(results_file)
        self.X_train = self.results.get('X_train')
        self.X_test = self.results.get('X_test')
        self.y_train = self.results.get('y_train')
        self.y_test = self.results.get('y_test')
        self.best_model = self.results.get('best_model')
        self.model = self.best_model
        print(f"📦 results已从 {results_file} 加载。")
        return self.results
