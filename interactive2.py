import pandas as pd
from fairlearn.metrics import(
    demographic_parity_difference ,
    equalized_odds_difference,
    MetricFrame,
    selection_rate,
    count
)
from fairlearn.reductions import GridSearch,DemographicParity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


def load_data(file_path,file_type='csv'):
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
    """调试版预处理"""
    try:
        print("=== 调试预处理开始 ===")
        print(f"输入数据形状: {df.shape}")
        print(f"特征: {features}")
        print(f"敏感特征: {sensitive_feature}")
        print(f"目标变量: {target_column}")

        # 选择需要的列
        required_columns = features + [sensitive_feature, target_column]
        print(f"需要的列: {required_columns}")

        # 检查列是否存在
        for col in required_columns:
            if col not in df.columns:
                print(f"❌ 列不存在: {col}")
                return None, None

        df_clean = df[required_columns].copy()
        print(f"选择列后形状: {df_clean.shape}")

        # 删除缺失值
        df_clean = df_clean.dropna()
        print(f"删除缺失值后形状: {df_clean.shape}")

        if len(df_clean) == 0:
            print("警告: 清理后没有数据了")
            return None, None

        print("数据类型:")
        print(df_clean.dtypes)

        # 编码分类变量
        object_cols = df_clean.select_dtypes(include=['object']).columns
        print(f"需要编码的列: {list(object_cols)}")

        for col in object_cols:
            print(f"处理列: {col}")
            df_clean[col] = pd.factorize(df_clean[col])[0]

        print("✅ 预处理成功!")
        return df_clean, features

    except Exception as e:
        print(f"❌ 调试预处理错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
def fairlearn_analysis(df,sensitive_feature,target_column,features):
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
    A_train = train[sensitive_feature].squeeze(axis=1) if isinstance(train[sensitive_feature], pd.DataFrame) else train[
        sensitive_feature]
    A_test = test[sensitive_feature].squeeze(axis=1) if isinstance(test[sensitive_feature], pd.DataFrame) else test[
        sensitive_feature]

    # ---------------------- 新增调试修复代码开始 ----------------------
    print("🔍 调试：敏感特征数据结构检查")
    print(f"敏感特征列形状: {A_test.shape}")
    print(f"敏感特征列类型: {type(A_test)}")

    # 确保敏感特征列为一维（修复核心逻辑）
    if len(A_test.shape) > 1:
        print(f"⚠️  发现多维敏感特征，自动转为一维...")
        # 方式1：适用于多维数组（优先使用）
        A_test = A_test.iloc[:,0]
        print(f"敏感特征列形状: {A_test.shape}")
        print(f"敏感特征列类型: {type(A_test)}")
        # 若方式1失败，注释上面一行，启用方式2（适用于嵌套列表）
        # A_test = A_test.explode().reset_index(drop=True)
    # ---------------------- 新增调试修复代码结束 ----------------------
    print(f"\n📊 数据分割:")
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    print(f"敏感特征分布:")
    print(A_test.value_counts())
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
    dp_diff = demographic_parity_difference(y_test,y_pred_base,sensitive_features=A_test)
    eo_diff = equalized_odds_difference(y_test,y_pred_base,sensitive_features=A_test)

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

    for group,rate in group_selection_rate.items():
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
            'demographic_parity_diff':dp_diff,
            'equalized_odds_diff':eo_diff,
        },
        'base_accuracy': base_accuracy,
        'base_precision': base_precision,
        'base_recall': base_recall

    }

if __name__ == '__main__':
    print("⚠️  注意：当前模式将训练一个新的随机森林模型用于测试")
    print("AI安全性分析工具")
    print("1.加载数据文件")
    features = []
    use_sample = input("是否使用示例数据？（y/n）:")
    if use_sample == 'y':
        file_path = "fairlearn_data.csv"  # 或 .xlsx
        print(f"默认使用文件：{file_path}")
        file_type = "csv"  # 或 "excel"
        print(f"使用默认文件类型{file_type}")

        df = load_data(file_path, file_type)
        print("\n💡 使用示例数据进行演示...")

        features = ['age', 'income', 'credit_score']
        sensitive_feature = 'gender'
        target_column = 'loan_approved'
    else:
        file_path = input("请输入数据文件路径：").strip()
        file_type = input("请输入文件类型：").strip()

        df = load_data(file_path, file_type)

        print("\n" + "=" * 50)
        print("请配置分析参数")
        print("=" * 50 )

#       显示所有特征
        all_columns = df.columns.tolist()
        print(f"数据中所有列：{all_columns}")
#       选择特征列
        print("\n请选择特征列(用于训练模型列)：")
        for i,col in enumerate(all_columns,1):
            print(f"{i}. {col}")

        feature_choices=input("请输入特征列编用逗号隔开，如：1，2，3）：").strip(',')
        features=[all_columns[int(i.strip())-1] for i in feature_choices if i.strip().isdigit()]

#       选择敏感特征
        print(f"\n请输入敏感特征列 (用于公平性分析的列):")
        for i, col in enumerate(all_columns, 1):
            print(f"  {i}. {col}")
        sensitive_idx = input("请输入1个敏感特征列的编号: ").strip()
        sensitive_feature = all_columns[int(sensitive_idx) - 1] if sensitive_idx.isdigit() else None

        # 选择目标变量
        print(f"\n🎯 请选择目标变量列:")
        for i, col in enumerate(all_columns, 1):
            print(f"  {i}. {col}")
        target_idx = input("请输入1个目标变量列的编号: ").strip()
        target_column = all_columns[int(target_idx) - 1] if target_idx.isdigit() else None

        print(f"\n🔍 调试信息:")
        print(f"features: {features} (长度: {len(features)})")
        print(f"sensitive_feature: {sensitive_feature}")
        print(f"target_column: {target_column}")

        # 验证选择
        if not features or not sensitive_feature or not target_column:
            print("❌ 参数选择不完整，请重新运行！")
            print(f"  缺失的特征: {'features' if not features else ''}")
            print(f"  缺失的敏感特征: {'sensitive_feature' if not sensitive_feature else ''}")
            print(f"  缺失的目标变量: {'target_column' if not target_column else ''}")
            exit()

    print(f"\n✅ 分析配置确认:")
    print(f"特征列: {features}")
    print(f"敏感特征: {sensitive_feature}")
    print(f"目标变量: {target_column}")



    df_clean, features_clean = data_preprocessing(
        df,
        features=features,
        sensitive_feature=sensitive_feature,  # 替换为你的敏感特征列
        target_column=target_column  # 替换为你的目标列
     )


    # 公平性分析
    if df_clean is not None:
        results = fairlearn_analysis(
            df_clean,
            sensitive_feature=sensitive_feature,
            target_column=target_column,
            features=features_clean
        )
        if results is not None:
            print(f"\n🎉 分析完成！")
            print(f"📊 发现 {len(results['A_test'].unique())} 个敏感特征组")
            print(f"⚖️ 模型公平性评估完毕")
        else:
            print("公平性分析失败")
    else:
        print("数据处理失败")

