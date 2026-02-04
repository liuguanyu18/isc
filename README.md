# Agentscope 负荷预测可视化智能体

基于设计文档实现的可视化智能体原型，支持从 MySQL 读取园区负荷数据，完成字段语义分析、数据清洗、特征分析、模型推荐、训练验证与结果可视化，并提供用户反馈闭环。

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动应用

```bash
streamlit run app.py
```

3. 在页面左侧填写 MySQL 连接信息并读取数据。

## 使用说明

- **时间列**：建议为可解析的时间戳字段。
- **目标列**：选择负荷/功率等数值字段。
- **运行分析**：执行语义解析、清洗与特征分析。
- **训练模型**：在推荐模型中选择并训练。
- **反馈**：在底部表单提交反馈，记录在 `data/feedback.json`。

## 目录结构

- `app.py`：Streamlit 可视化主程序
- `core/`：技能模块（数据接入、语义、清洗、特征、模型、训练、可视化、反馈）
- `data/`：运行时产物与反馈记录
- `requirements.txt`：依赖列表

## 说明

- 模型推荐默认包含 `Naive` 与 `LinearRegression`，若安装 `statsmodels` 可使用 ARIMA/SARIMA。
- 可按需安装 `prophet` 以启用 Prophet 模型。
