import pandas as pd
import streamlit as st

from core.data_ingestion import SourceConfig, build_mysql_engine, load_data, normalize_dataframe, schema_summary
from core.field_semantics import infer_semantics, pick_best_column
from core.data_cleaning import clean_data
from core.feature_analysis import analyze_features
from core.model_recommendation import recommend_models
from core.training_validation import train_model
from core.result_visualization import plot_forecast, plot_error_distribution, generate_suggestions
from core.flow import build_flow_dot, STEP_ORDER
from core.interaction_feedback import FeedbackEntry, save_feedback


st.set_page_config(page_title="Agentscope 负荷预测智能体", layout="wide")


def init_state():
    defaults = {
        "df": None,
        "schema": None,
        "semantics": None,
        "cleaned": None,
        "clean_report": None,
        "feature_result": None,
        "recommendations": None,
        "train_result": None,
        "status": {key: "pending" for _, key in STEP_ORDER},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

st.title("基于园区负荷预测的 Agentscope 可视化智能体")

with st.sidebar:
    st.header("MySQL 数据源")
    host = st.text_input("Host", value="localhost")
    port = st.number_input("Port", min_value=1, max_value=65535, value=3306)
    user = st.text_input("User", value="root")
    password = st.text_input("Password", type="password")
    database = st.text_input("Database")

    source_mode = st.radio("读取方式", ["表名", "SQL"], horizontal=True)
    table = None
    query = None
    if source_mode == "表名":
        table = st.text_input("Table")
    else:
        query = st.text_area("SQL 查询", height=120)

    limit = st.number_input("读取行数限制", min_value=0, value=10000, step=1000)

    if st.button("读取数据"):
        try:
            config = SourceConfig(
                host=host,
                port=int(port),
                user=user,
                password=password,
                database=database,
                table=table if table else None,
                query=query if query else None,
                limit=int(limit) if limit else None,
            )
            engine = build_mysql_engine(config)
            df = load_data(engine, config)
            df, time_col = normalize_dataframe(df)

            st.session_state.df = df
            st.session_state.schema = schema_summary(df)
            st.session_state.time_col = time_col
            st.session_state.status["data_ingestion"] = "done"
            st.success("数据读取成功")
        except Exception as exc:
            st.session_state.status["data_ingestion"] = "error"
            st.error(f"读取失败: {exc}")

st.divider()

if st.session_state.df is None:
    st.info("请在左侧填写 MySQL 连接信息并读取数据。")
    st.stop()

# Parameter form
with st.form("params"):
    st.subheader("参数设置")
    df = st.session_state.df
    semantics = infer_semantics(df.columns.tolist())
    st.session_state.semantics = semantics
    time_col_default = getattr(st.session_state, "time_col", None) or pick_best_column(semantics.mapping, "time")
    target_default = pick_best_column(semantics.mapping, "power") or pick_best_column(semantics.mapping, "load")

    time_options = df.columns.tolist()
    time_index = time_options.index(time_col_default) if time_col_default in time_options else 0
    time_col = st.selectbox("时间列", options=time_options, index=time_index)

    target_options = df.columns.tolist()
    target_index = target_options.index(target_default) if target_default in target_options else 0
    target_col = st.selectbox("目标列", options=target_options, index=target_index)

    fill_strategy = st.selectbox("缺失值处理", ["interpolate", "mean", "median", "zero"], index=0)
    outlier_strategy = st.selectbox("异常值处理", ["clip", "none"], index=0)
    scale_numeric = st.checkbox("标准化数值特征", value=False)

    test_ratio = st.slider("测试集比例", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    lag = st.number_input("滞后阶数", min_value=1, max_value=168, value=24)

    run_pipeline = st.form_submit_button("运行分析")

if run_pipeline:
    try:
        if target_col not in df.columns:
            raise ValueError("目标列不存在")
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError("目标列必须是数值类型")

        st.session_state.time_col = time_col
        st.session_state.target_col = target_col
        st.session_state.status["field_semantics"] = "done"

        cleaning_result = clean_data(
            df,
            time_col=time_col,
            fill_strategy=fill_strategy,
            outlier_strategy=outlier_strategy,
            scale_numeric=scale_numeric,
        )
        st.session_state.cleaned = cleaning_result.data
        st.session_state.clean_report = cleaning_result.report
        st.session_state.status["data_cleaning"] = "done"

        feature_result = analyze_features(cleaning_result.data, time_col, target_col)
        st.session_state.feature_result = feature_result
        st.session_state.status["feature_analysis"] = "done"

        recommendations = recommend_models(len(cleaning_result.data), feature_result.has_seasonality)
        st.session_state.recommendations = recommendations
        st.session_state.status["model_recommendation"] = "done"

    except Exception as exc:
        st.error(f"分析失败: {exc}")

left, center, right = st.columns([1.1, 1, 1.1])

with left:
    st.subheader("数据与语义面板")
    st.write("数据预览")
    st.dataframe(st.session_state.df.head(20), use_container_width=True)

    if st.session_state.schema is not None:
        st.write("字段结构")
        st.dataframe(st.session_state.schema, use_container_width=True)

    if st.session_state.semantics is not None:
        st.write("字段语义映射")
        st.dataframe(st.session_state.semantics.mapping, use_container_width=True)
        if st.session_state.semantics.ambiguous:
            st.warning(f"需要确认字段: {', '.join(st.session_state.semantics.ambiguous)}")

    if st.session_state.clean_report:
        st.write("清洗报告")
        st.json(st.session_state.clean_report)

with center:
    st.subheader("流程链路图")
    dot = build_flow_dot(st.session_state.status)
    try:
        st.graphviz_chart(dot)
    except Exception:
        st.code(dot)

    st.write("节点状态")
    for label, key in STEP_ORDER:
        state = st.session_state.status.get(key, "pending")
        st.write(f"{label}: {state}")

with right:
    st.subheader("模型与结果面板")
    if st.session_state.recommendations:
        st.write("推荐模型")
        rec_df = [{"模型": r.name, "理由": r.rationale, "依赖": r.requires} for r in st.session_state.recommendations]
        st.dataframe(rec_df, use_container_width=True)

        model_names = [r.name for r in st.session_state.recommendations]
        selected_model = st.selectbox("选择模型", model_names)

        if st.session_state.cleaned is None:
            st.warning("请先运行分析以生成清洗数据。")
        elif st.button("训练模型"):
            try:
                exogenous = []
                if st.session_state.feature_result is not None:
                    exogenous = st.session_state.feature_result.key_variables.index.tolist()
                exogenous = [col for col in exogenous if col in st.session_state.cleaned.columns and col != target_col]

                train_result = train_model(
                    selected_model,
                    st.session_state.cleaned,
                    time_col,
                    target_col,
                    exogenous_cols=exogenous,
                    lag=int(lag),
                    test_ratio=float(test_ratio),
                    seasonal=st.session_state.feature_result.has_seasonality if st.session_state.feature_result else False,
                )
                st.session_state.train_result = train_result
                st.session_state.status["training_validation"] = "done"
                st.session_state.status["result_visualization"] = "done"
            except Exception as exc:
                st.session_state.status["training_validation"] = "error"
                st.error(f"训练失败: {exc}")

    if st.session_state.train_result:
        st.write(f"模型: {st.session_state.train_result.model_name}")
        st.write(st.session_state.train_result.metrics)

        history = st.session_state.cleaned[target_col]
        predictions = st.session_state.train_result.predictions
        fig = plot_forecast(history, predictions)
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(plot_error_distribution(predictions), use_container_width=True)
        st.write("建议")
        for suggestion in generate_suggestions(predictions):
            st.write(f"- {suggestion}")

st.divider()

with st.container():
    st.subheader("用户交互与反馈")
    with st.form("feedback"):
        user_name = st.text_input("用户")
        rating = st.slider("预测满意度", min_value=1, max_value=5, value=4)
        comments = st.text_area("反馈意见", height=120)
        adjustments = {
            "time_col": time_col,
            "target_col": target_col,
            "fill_strategy": fill_strategy,
            "outlier_strategy": outlier_strategy,
            "scale_numeric": scale_numeric,
            "test_ratio": test_ratio,
            "lag": lag,
        }
        submitted = st.form_submit_button("提交反馈")

    if submitted:
        save_feedback(
            FeedbackEntry(
                user=user_name or "anonymous",
                rating=int(rating),
                comments=comments,
                adjustments=adjustments,
            )
        )
        st.session_state.status["interaction_feedback"] = "done"
        st.success("反馈已保存")
