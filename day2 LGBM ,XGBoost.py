import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        # 日期解析
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%y', errors='coerce')
        self.data = self.data.dropna(subset=['Date'])

        # 衍生时间特征
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Season'] = self.data['Month'].apply(
            lambda x: 'Spring' if 3 <= x <= 5 else 'Summer' if 6 <= x <= 8 else 'Autumn' if 9 <= x <= 11 else 'Winter'
        )
        self.data['Day'] = self.data['Date'].dt.day
        self.data['DayOfWeek'] = self.data['Date'].dt.dayofweek

        # 计算目标变量（平均价格）
        self.data['Average Price'] = (self.data['Low Price'] + self.data['High Price']) / 2

        # 缺失值填充
        for col in ['Type', 'Item Size', 'Color']:
            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])

        # 高基数类别处理
        for col in ['City Name', 'Origin']:
            top_10_cats = self.data[col].value_counts().head(10).index
            self.data[col] = self.data[col].apply(lambda x: x if x in top_10_cats else 'Other')

        # 删除冗余列
        drop_cols = [
            'Grade', 'Environment', 'Unit of Sale', 'Quality',
            'Condition', 'Appearance', 'Storage', 'Crop',
            'Trans Mode', 'Unnamed: 24', 'Unnamed: 25',
            'Low Price', 'High Price', 'Mostly Low', 'Mostly High',
            'Sub Variety', 'Origin District', 'Repack'
        ]
        self.data = self.data.drop(drop_cols, axis=1, errors='ignore')

        return self.data


class FeatureEngineer:
    def __init__(self, data):
        self.data = data

    def engineer(self):
        numerical_features = ['Month', 'Year', 'Day', 'DayOfWeek']
        categorical_features = [
            'City Name', 'Type', 'Package', 'Variety',
            'Origin', 'Item Size', 'Color', 'Season'
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        return preprocessor, numerical_features, categorical_features


class ModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def train_and_evaluate(self, model):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2

    def improved_training(self):
        results = {}

        # 线性回归
        print("开始训练线性回归模型...")
        lr_model = LinearRegression()
        lr_mse, lr_r2 = self.train_and_evaluate(lr_model)
        results['Linear Regression'] = {'MSE': lr_mse, 'R²': lr_r2}
        print(f"线性回归的均方误差（MSE）: {lr_mse}")
        print(f"线性回归的决定系数（R²）: {lr_r2}")

        # 岭回归
        print("开始训练岭回归模型...")
        ridge_model = Ridge()
        param_grid_ridge = {'alpha': [0.1, 1, 10]}
        grid_search_ridge = GridSearchCV(estimator=ridge_model, param_grid=param_grid_ridge, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_ridge.fit(self.X_train, self.y_train)
        best_ridge_model = grid_search_ridge.best_estimator_
        ridge_mse, ridge_r2 = self.train_and_evaluate(best_ridge_model)
        results['Ridge Regression'] = {'MSE': ridge_mse, 'R²': ridge_r2}
        print(f"优化后的岭回归的均方误差（MSE）: {ridge_mse}")
        print(f"优化后的岭回归的决定系数（R²）: {ridge_r2}")

        # Lasso回归
        print("开始训练Lasso回归模型...")
        lasso_model = Lasso()
        param_grid_lasso = {'alpha': [0.1, 1, 10]}
        grid_search_lasso = GridSearchCV(estimator=lasso_model, param_grid=param_grid_lasso, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_lasso.fit(self.X_train, self.y_train)
        best_lasso_model = grid_search_lasso.best_estimator_
        lasso_mse, lasso_r2 = self.train_and_evaluate(best_lasso_model)
        results['Lasso Regression'] = {'MSE': lasso_mse, 'R²': lasso_r2}
        print(f"优化后的Lasso回归的均方误差（MSE）: {lasso_mse}")
        print(f"优化后的Lasso回归的决定系数（R²）: {lasso_r2}")

        # 随机森林回归
        print("开始训练随机森林回归模型...")
        rf_model = RandomForestRegressor(random_state=42)
        param_grid_rf = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5,
                                      scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_rf.fit(self.X_train, self.y_train)
        best_rf_model = grid_search_rf.best_estimator_
        rf_mse, rf_r2 = self.train_and_evaluate(best_rf_model)
        results['Random Forest'] = {'MSE': rf_mse, 'R²': rf_r2}
        print(f"优化后的随机森林回归的均方误差（MSE）: {rf_mse}")
        print(f"优化后的随机森林回归的决定系数（R²）: {rf_r2}")

        # XGBoost回归
        print("开始训练XGBoost回归模型...")
        xgb_model = XGBRegressor(random_state=42)
        param_grid_xgb = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'lambda': [0.1, 1, 10]
        }
        grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5,
                                       scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_xgb.fit(self.X_train, self.y_train)
        best_xgb_model = grid_search_xgb.best_estimator_
        xgb_mse, xgb_r2 = self.train_and_evaluate(best_xgb_model)
        results['XGBoost'] = {'MSE': xgb_mse, 'R²': xgb_r2}
        print(f"优化后的XGBoost回归的均方误差（MSE）: {xgb_mse}")
        print(f"优化后的XGBoost回归的决定系数（R²）: {xgb_r2}")

        # LightGBM回归 - 优化参数设置以减少警告
        print("开始训练LightGBM回归模型...")
        lgb_model = lgb.LGBMRegressor(
            random_state=42,
            min_child_samples=5,  # 减少每个叶子节点所需的最小样本数
            min_split_gain=0.01,  # 降低分裂所需的最小增益阈值
            num_leaves=31,  # 限制树的复杂度
            learning_rate=0.1,  # 适中的学习率
            n_estimators=100,  # 减少树的数量
            subsample=0.8,  # 样本采样比例
            colsample_bytree=0.8,  # 特征采样比例
            reg_alpha=0.1,  # L1正则化
            reg_lambda=0.1,  # L2正则化
            verbose=-1  # 减少日志输出
        )

        param_grid_lgb = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'num_leaves': [15, 31, 63]  # 降低最大叶子节点数
        }

        grid_search_lgb = GridSearchCV(
            estimator=lgb_model,
            param_grid=param_grid_lgb,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0  # 减少网格搜索过程中的输出
        )

        # 修复：移除了fit()中的verbose=0参数
        grid_search_lgb.fit(self.X_train, self.y_train)
        best_lgb_model = grid_search_lgb.best_estimator_

        # 训练最佳模型时也减少日志
        best_lgb_model.set_params(verbose=-1)
        lgb_mse, lgb_r2 = self.train_and_evaluate(best_lgb_model)
        results['LightGBM'] = {'MSE': lgb_mse, 'R²': lgb_r2}
        print(f"优化后的LightGBM回归的均方误差（MSE）: {lgb_mse}")
        print(f"优化后的LightGBM回归的决定系数（R²）: {lgb_r2}")

        # 支持向量机回归
        print("开始训练支持向量机回归模型...")
        svr_model = SVR()
        param_grid_svr = {
            'C': [0.1, 1, 10],
            'epsilon': [0.1, 0.2, 0.3],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
        grid_search_svr = GridSearchCV(estimator=svr_model, param_grid=param_grid_svr, cv=5,
                                       scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_svr.fit(self.X_train, self.y_train)
        best_svr_model = grid_search_svr.best_estimator_
        svr_mse, svr_r2 = self.train_and_evaluate(best_svr_model)
        results['SVR'] = {'MSE': svr_mse, 'R²': svr_r2}
        print(f"优化后的支持向量机回归的均方误差（MSE）: {svr_mse}")
        print(f"优化后的支持向量机回归的决定系数（R²）: {svr_r2}")

        # 绘制柱状图比较模型性能
        models = list(results.keys())
        mse_values = [results[model]['MSE'] for model in models]
        r2_values = [results[model]['R²'] for model in models]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(models, mse_values, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title('Model Comparison - MSE')
        plt.xticks(rotation=45)

        plt.subplot(1, 2, 2)
        plt.bar(models, r2_values, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('R-squared (R²)')
        plt.title('Model Comparison - R²')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        return results


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize(self):
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data['Average Price'], bins=20, alpha=0.5, label='Average Price', kde=True)
        plt.title('Average Price Distribution')
        plt.xlabel('Price')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Variety', y='Average Price', data=self.data)
        plt.title('Average Price by Variety')
        plt.xticks(rotation=45)
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.barplot(x='Variety', y='Average Price', data=self.data)
        plt.title('Average Price by Variety')
        plt.xticks(rotation=45)
        plt.show()


class DataAnalyzer:
    def __init__(self, data, processed_data):
        self.data = data
        self.processed_data = processed_data

    def analyze(self):
        print("\n数据集的前几行：")
        print(self.data.head())

        print("\n数据集的基本信息：")
        print(self.data.info())

        print("\n数据集的统计信息：")
        print(self.data.describe())

        print("\n清洗后的数据：")
        print(self.processed_data.head())

        print("\n描述性统计信息：")
        print(self.processed_data.describe(include='all'))

        print("\n偏度和峰度：")
        print(self.processed_data['Average Price'].skew())
        print(self.processed_data['Average Price'].kurtosis())

        # 计算不同品种的平均价格
        average_prices = self.processed_data.groupby('Variety')[['Average Price']].mean()
        print("\n不同品种的平均价格：")
        print(average_prices)

        # 计算不同产地的平均价格
        average_prices_by_origin = self.processed_data.groupby('Origin')[['Average Price']].mean()
        print("\n不同产地的平均价格：")
        print(average_prices_by_origin)

        # 计算不同尺寸的平均价格
        average_prices_by_size = self.processed_data.groupby('Item Size')[['Average Price']].mean()
        print("\n不同尺寸的平均价格：")
        print(average_prices_by_size)

        print("\n总结：")
        print("通过可视化和分析，我们可以看到不同品种、产地和尺寸的南瓜价格分布情况。")
        print("例如，某些品种的南瓜价格波动较大，而某些产地的南瓜价格相对稳定。")
        print("这些信息可以帮助我们更好地理解南瓜市场的价格动态。")


def main():
    # 加载数据
    print("开始加载数据...")
    data = pd.read_csv('US-pumpkins.csv')
    print(data.head())

    # 数据预处理
    print("开始数据预处理...")
    preprocessor = DataPreprocessor(data)
    data_processed = preprocessor.preprocess()
    print(data_processed.head())

    # 特征工程
    print("开始特征工程...")
    engineer = FeatureEngineer(data_processed)
    preprocessor_obj, _, _ = engineer.engineer()

    # 数据划分
    print("开始数据划分...")
    X = data_processed.drop(columns=['Average Price'], errors='ignore')
    y = data_processed['Average Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 预处理数据
    print("开始预处理数据...")
    X_train_processed = preprocessor_obj.fit_transform(X_train)
    X_test_processed = preprocessor_obj.transform(X_test)
    print(X_train_processed.shape)
    print(X_test_processed.shape)

    # 改进后的模型训练和评估
    print("开始改进后的模型训练和评估...")
    trainer = ModelTrainer(X_train_processed, X_test_processed, y_train, y_test)
    results = trainer.improved_training()

    # 数据可视化
    visualizer = DataVisualizer(data_processed)
    visualizer.visualize()

    # 数据探索性分析
    analyzer = DataAnalyzer(data, data_processed)
    analyzer.analyze()

    return results


if __name__ == "__main__":
    results = main()