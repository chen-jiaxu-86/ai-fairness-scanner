from flask import Flask,request,render_template,jsonify
import pandas as pd
import os

from interactive1 import load_data,data_preprocessing,fairlearn_analysis

app = Flask(__name__)
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI公平性分析平台</title>
        <style>
            body { font-family: Arial;margin: 40px; }
            .container { max-width: 600px; margin: 0 auto;}
            input,button { padding-left: 10px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI公平性扫描</h1>
            <form action="/analyze" method="post" enctype="multipart/form-data">
                <h3>上传数据文件（csv）</h3>
                <input type="file" name="file" accept=".csv" required>

                <h3> 2.设置分析参数</h3>
                <input type="text" name="sensitive_feature" placeholder="敏感特征列名" required>
                <input type="text" name="target_column" placeholder="目标变量列名" required>
                
                <br><br>
                <button type="submit">开始分析</button>
            </form>
        </div>
    </body>
    </html>
    '''
@app.route('/analyze',methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        sensitive_feature = request.form['sensitive_feature']
        target_column = request.form['target_column']
        print(f"📁 收到文件: {file.filename}")
        print(f"🔍 敏感特征: {sensitive_feature}")
        print(f"🎯 目标变量: {target_column}")

        df = pd.read_csv(file)
        print(f"📊 数据读取成功，形状: {df.shape}")
        print(f"📋 数据列名: {list(df.columns)}")

        features = [col for col in df.columns if col not in [sensitive_feature, target_column]]
        print(f"🎯 特征列: {features}")

        print("🔄 开始数据预处理...")
        df_clean,features_clean = data_preprocessing(df,features,sensitive_feature,target_column)

        if df_clean is not None:
            print("✅ 数据预处理成功")
            print(f"🔄 开始公平性分析...")
            results = fairlearn_analysis(df_clean,sensitive_feature,target_column,features_clean)
            print("✅ 公平性分析完成")

            #生成网页报告
            return f'''
            <h1>公平性分析报告</h1>
            <div style="background:#f5f5f5;padding:20px;border-radius:10px;">
                <h2>模型性能</h2>
                <p>准确率: {results.get("accuracy", "N/A")}</p >
                
                <h2>公平性指标</h2>
                <p>统计均等差异: {results['fairness_metrics']['demographic_parity_diff']:.3f}</p >
                <p>均等几率差异: {results['fairness_metrics']['equalized_odds_diff']:.3f}</p >
                
                <h2>详细结果</h2>
                <pre>{results['metrics'].by_group if 'metrics' in results else '详细结果暂不可用'}</pre>
            </div>
            <br>
            <a href="/">返回首页</a >
            '''
        else:
            print("❌ 数据预处理失败")
            return "数据处理失败，请检查数据格式"
    except Exception as e:
        print(f"💥 分析过程中出现错误: {str(e)}")
        import traceback
        print(f"🔍 详细错误信息: {traceback.format_exc()}")
        return f"分析过程中出现错误：{str(e)}"

if __name__ == "__main__":
    app.run(debug=True)