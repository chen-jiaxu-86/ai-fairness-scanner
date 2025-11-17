import pandas as pd
import numpy as np
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    count
)
from fairlearn.reductions import GridSearch, DemographicParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import streamlit as st
import warnings

warnings.filterwarnings("ignore")


def load_data(file_path, file_type='csv'):
    try:
        if file_type.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif file_type.lower() == 'excel':
            df = pd.read_excel(file_path)
        else:
            raise ValueError("文件类型必须是'excel'或‘csv'")
        print(f"✅ 数据加载成功！形状: {df.shape}")
        print(f"📊 数据列: {list(df.columns)}")
        print("\n🔍 数据前5行:")
        print(df.head())
        print("\n📋 数据基本信息:")
        print(df.info())

        return df
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None


def data_preprocessing(df, features, sensitive_feature, target_column):
    try:
        print("=== 预处理函数内部开始 ===")
        print(f"输入数据形状: {df.shape}")

        # 检查列是否存在
        required_columns = features + [sensitive_feature, target_column]
        print(f"需要的列: {required_columns}")

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"错误: 以下列不存在: {missing_columns}")
            print(f"数据框中实际存在的列: {list(df.columns)}")
            return None, None

        print("所有需要的列都存在")

        # 检查缺失值
        print("缺失值统计:")
        missing_stats = df[required_columns].isnull().sum()
        print(missing_stats)

        # 处理缺失值 - 删除有缺失值的行
        df_clean = df[required_columns].copy()
        initial_count = len(df_clean)
        df_clean = df_clean.dropna()
        final_count = len(df_clean)

        print(f"数据清理: {initial_count} -> {final_count} 行")

        # 确保还有数据
        if len(df_clean) == 0:
            print("警告: 清理后没有数据了")
            return None, None

        print("预处理完成!")
        print(f"返回数据形状: {df_clean.shape}")
        return df_clean, features

    except Exception as e:
        print(f"预处理函数内部错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None


print("🎯 检查点：data_preprocessing调用完成")


def fairlearn_analysis(df, sensitive_feature, target_column, features):
    X = df[features]
    y = df[target_column]
    A = df[sensitive_feature]

    # 将 X、y、A 按列组合，然后进行分割
    combined = pd.concat([X, y, A], axis=1)
    train, test = train_test_split(combined, test_size=0.3, random_state=42, stratify=y)
    X_train = train[features]
    X_test = test[features]
    y_train = train[target_column]
    y_test = test[target_column]
    A_train = train[sensitive_feature]
    A_test = test[sensitive_feature]

    print(f"\n📊 数据分割:")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"敏感特征分布:\n{A_test.value_counts()}")
    # 训练基础模型
    print("\n🤖 训练基础模型...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)

    print("\n" + "=" * 60)
    print("📊 FAIRLEARN 公平性分析报告")
    print("=" * 60)

    # 基础模型性能
    base_accuracy = accuracy_score(y_test, y_pred_base)
    base_precision = precision_score(y_test, y_pred_base, average='weighted')
    base_recall = recall_score(y_test, y_pred_base, average='weighted')

    print(f"\n🎯 基础模型性能:")
    print(f"准确率: {base_accuracy:.3f}")
    print(f"精确率: {base_precision:.3f}")
    print(f"召回率: {base_recall:.3f}")
    # 基础模型性能
    dp_diff = demographic_parity_difference(y_test, y_pred_base, sensitive_features=A_test)
    eo_diff = equalized_odds_difference(y_test, y_pred_base, sensitive_features=A_test)

    print(f"\n⚖️ 公平性指标:")
    print(f"统计均等差异: {dp_diff:.3f} (越接近0越公平)")
    print(f"均等几率差异: {eo_diff:.3f} (越接近0越公平)")

    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='binary'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='binary'),
        'selection_rate': selection_rate,
        'count': count,
    }

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_pred_base,
        sensitive_features=A_test,
    )

    print(f"\n📋 按 [{sensitive_feature}] 分组的详细指标:")
    print(metric_frame.by_group.round(3))

    # 偏差分析
    print(f"\n📈 偏差分析:")
    overall_selection_rate = metric_frame.overall['selection_rate']
    group_selection_rate = metric_frame.by_group['selection_rate']

    for group, rate in group_selection_rate.items():
        bias = rate - overall_selection_rate
        print(f"  {group}: 选择率 = {rate:.3f}, 偏差 = {bias:+.3f}")

    return {
        'model': base_model,
        'X_test': X_test,
        'y_test': y_test,
        'A_test': A_test,
        'y_pred_base': y_pred_base,
        'metrics': metric_frame,
        'fairness_metrics': {
            'demographic_parity_diff': dp_diff,
            'equalized_odds_diff': eo_diff,
        }
    }


if __name__ == '__main__':
    print("⚠️  注意：当前模式将训练一个新的随机森林模型用于演示")
    print("📊 实际业务中请使用 '模型评估' 模式")
    print("AI安全性分析工具")
    print("1.加载数据文件")

    file_path = "fairlearn_data.csv"  # 或 .xlsx
    file_type = "csv"  # 或 "excel"

    features = ['age', 'income', 'credit_score','employment_years', 'debt_to_income']
    df = load_data(file_path, file_type)


    # 在主程序中添加更详细的调试
    print("=== 详细调试信息 ===")

    print("=== 调用前的变量检查 ===")
    print(f"df 类型: {type(df)}")
    print(f"df 形状: {df.shape}")
    print(f"features 值: {features}")
    print(f"features 类型: {type(features)}")

    # 检查 features 是否正常
    if not isinstance(features, list) or features != ['age', 'income', 'credit_score', 'employment_years',
                                                      'debt_to_income']:
        print("⚠️ 警告: features 变量异常，重新定义!")
        features = ['age', 'income', 'credit_score', 'employment_years', 'debt_to_income']
        print(f"重新定义后的 features: {features}")

    print(f"sensitive_feature: {'gender'}")
    print(f"target_column: {'loan_approved'}")

    # 1. 检查预处理函数调用
    print("1. 调用预处理函数...")
    df_clean, features_clean = data_preprocessing(
        df,
        features=['age', 'income', 'credit_score', 'employment_years', 'debt_to_income'],
        sensitive_feature='gender',
        target_column='loan_approved'
    )

    print("2. 预处理函数返回结果:")
    print(f"df_clean 类型: {type(df_clean)}")
    print(f"df_clean 是否为 None: {df_clean is None}")
    print(f"features_clean: {features_clean}")

    if df_clean is not None:
        print(f"df_clean 形状: {df_clean.shape}")
        print(f"df_clean 列名: {df_clean.columns.tolist()}")
        print("预处理后的数据样本:")
        print(df_clean.head())
    else:
        print("预处理返回了 None，检查预处理函数内部")
        exit()

    # 3. 检查 fairlearn_analysis 函数调用
    print("\n3. 准备调用公平性分析...")
    print(f"将传递的参数:")
    print(f"- df_clean 类型: {type(df_clean)}")
    print(f"- features_clean: {features_clean}")
    print(f"- sensitive_feature: gender")
    print(f"- target_column: loan_approved")

    # 4. 在调用前再次验证列是否存在
    if df_clean is not None:
        print(f"'gender' 在 df_clean 中: {'gender' in df_clean.columns}")
        for feature in features_clean:
            print(f"'{feature}' 在 df_clean 中: {feature in df_clean.columns}")
        print(f"'loan_approved' 在 df_clean 中: {'loan_approved' in df_clean.columns}")

    # 5. 调用公平性分析
    print("\n4. 调用公平性分析函数...")
    try:
        results = fairlearn_analysis(
            df_clean,
            features=features_clean,
            sensitive_feature='gender',
            target_column='loan_approved'
        )
        print("公平性分析完成!")
    except Exception as e:
        print(f"公平性分析出错: {e}")
        print(f"错误类型: {type(e)}")
        import traceback

        traceback.print_exc()

    if df is not None:
        df_clean, features = data_preprocessing(
            df,
            features=features,
            sensitive_feature='gender',  # 替换为你的敏感特征列
            target_column='loan_approved'  # 替换为你的目标列
        )

        # 公平性分析
        results = fairlearn_analysis(
            df_clean,
            sensitive_feature='gender',
            target_column='loan_approved',
            features=features
        )

        print(f"\n🎉 分析完成！")
        print(f"📊 发现 {len(results['A_test'].unique())} 个敏感特征组")
        print(f"⚖️ 模型公平性评估完毕")

    else:
        print("\n💡 使用示例数据进行演示...")

        # 创建示例数据
        np.random.seed(42)
        n_samples = 1000
        example_data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.normal(50000, 20000, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.5, 0.45, 0.05]),
            'loan_approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        }
        df_example = pd.DataFrame(example_data)

        print("2. 示例数据公平性分析")
        df_clean, features = data_preprocessing(
            df_example,
            features=features,
            sensitive_feature='gender',
            target_column='loan_approved'

        )

        results = fairlearn_analysis(
            df_clean,
            sensitive_feature='gender',
            target_column='loan_approved',
            features=features
        )