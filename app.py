#!/usr/bin/env python3
# streamlit run app.py
import streamlit as st
import pandas as pd
import duckdb, plotly.express as px, plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from id2name import id2name
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import requests

# ------------------------------ 常量 ---------------------------------
DB_PATH   = Path(__file__).with_name("mydb.duckdb")
DATA_PATH = Path(__file__).with_name("data") / "fans_events.parquet"
CACHE_DIR = Path("cache"); CACHE_DIR.mkdir(exist_ok=True)

@st.cache_resource
def get_conn():
    return duckdb.connect(str(DB_PATH))

# ------------------------- 语言包 -------------------------------
LANG = {
    "zh": {
        "lang_switch": "语言 / Language",
        "matrix": "用户流动矩阵分析",
        "trend": "用户流动趋势分析",
        "aarrr": "AARRR 漏斗分析",
        "rfm": "RFM 用户分层",
        "cluster": "用户群体聚类分析",
        "assoc": "用户兴趣关联分析",
        "update": "用户记录增量更新",
        "select": "选择主播",
        "reload": "重新计算",
        "src_table": "用户来源分析表",
        "tgt_table": "用户流失去向表",
        "src_heat": "来源热图",
        "tgt_heat": "去向热图",
        "start": "开始增量更新",
        "gen": "生成图表",
        "funnel_acquisition": "Acquisition 月新增",
        "funnel_activation": "Activation 当月活跃",
        "funnel_retention": "Retention 次月回流",
        "funnel_revenue": "Revenue 付费人数",
        "funnel_referral": "Referral 推荐人数",
        "rfm_score": "RFM 得分分布",
        "rfm_segment": "RFM 分层结果",
    },
    "en": {
        "lang_switch": "Language / 语言",
        "matrix": "User Transfer Matrix",
        "trend": "User Flow Trend Analysis",
        "aarrr": "AARRR Funnel",
        "rfm": "RFM Segmentation",
        "cluster": "User Clustering",
        "assoc": "Interest Association",
        "update": "Incremental Update",
        "select": "Select streamers",
        "reload": "Recalculate",
        "src_table": "User Source Table",
        "tgt_table": "User Target Table",
        "src_heat": "Source Heatmap",
        "tgt_heat": "Target Heatmap",
        "start": "Start incremental update",
        "gen": "Generate",
        "funnel_acquisition": "Acquisition (New)",
        "funnel_activation": "Activation (Active)",
        "funnel_retention": "Retention (Return)",
        "funnel_revenue": "Revenue (Pay)",
        "funnel_referral": "Referral (Invite)",
        "rfm_score": "RFM Score",
        "rfm_segment": "RFM Segment",
    },
}

language = st.sidebar.selectbox("Language", ["zh", "en"], format_func=lambda x: LANG[x]["lang_switch"])
T = LANG[language]

livers = list(id2name.keys())
names  = [id2name[i] for i in livers]

# ------------------------------ 首页 -------------------------------
st.set_page_config(layout="wide")
st.title("VTuber Flow and Audience Analytics System")
st.markdown("---")

# ========== 统一主播选择 ==========
def select_livers(section_key: str, default=None):
    """返回 (主播名列表, 主播ID列表)"""
    if default is None:
        default = ["嘉然"]
    sel_names = st.multiselect("选择主播（不选=大盘）", names, default=default, key=section_key)
    sel_ids = [k for k, v in id2name.items() if v in sel_names] or list(livers)
    return sel_names, tuple(sel_ids)

# ====================== 1. 规模维度 ======================
# ====================== 1.1 大盘活跃程度分层 ======================
st.markdown("# 1. 规模维度")
st.header("1.1 大盘活跃程度分层")
conn = get_conn()

# 直接读现成的 monthly_mau_layer 表
df_scale = conn.execute("""
    SELECT month,
           mau,
           fixed_mau,
           flowing_mau,
           ylg_mau
    FROM monthly_mau_layer
    ORDER BY month
""").fetchdf()

fig_scale = px.line(df_scale, x="month", y=["mau", "fixed_mau", "flowing_mau", "ylg_mau"],
                    labels={"value": "人数", "month": "月份", "variable": "分层"},
                    title="MAU（自然月活跃）& 固定/流动/流浪观众")
st.plotly_chart(fig_scale, use_container_width=True)

# 1.1 解读
latest = df_scale.tail(1).iloc[0]
total, fixed, flowing, ylg = latest["mau"], latest["fixed_mau"], latest["flowing_mau"], latest["ylg_mau"]
fix_ratio = fixed / total if total > 0 else 0

# 显示数据事实
if fix_ratio > 0.55:
    st.info(f"大盘固定观众占比达{fix_ratio:.1%}（{fixed:,}人），基本盘健康稳定；"
            f"流动观众{flowing:,}人，YLG观众{ylg:,}人。")
    st.info("建议在保持固定观众活跃度的同时，针对流动观众设计转化活动，提高付费转化率。")
elif fix_ratio > 0.4:
    st.info(f"大盘固定观众占比{fix_ratio:.1%}（{fixed:,}人），处于中等水平；"
            f"流动观众{flowing:,}人，YLG观众{ylg:,}人。")
    st.info("建议加强内容一致性，提高流动观众向固定观众的转化率。")
else:
    st.info(f"大盘固定观众占比仅{fix_ratio:.1%}（{fixed:,}人），观众稳定性不足；"
            f"流动观众{flowing:,}人，YLG观众{ylg:,}人。")
    st.info("建议急需通过定期活动和互动机制提高观众粘性，减少观众流失。")

# ====================== 1.2 单主播视角：自然月活跃 & 分层 ======================
# ---------- 单个主播的 MAU 分层折线 ----------
st.subheader("1.2 单主播视角：自然月活跃 & 分层")

# 主播选择
sel_liver_name = st.selectbox("选择主播", names, index=names.index("嘉然"))
sel_liver_id   = [k for k, v in id2name.items() if v == sel_liver_name][0]

conn = get_conn()
df_liver_scale = conn.execute("""
    SELECT month,
           mau,
           fixed_mau,
           flowing_mau,
           ylg_mau
    FROM liver_monthly_mau_layer
    WHERE liver = ?
    ORDER BY month
""", [sel_liver_id]).fetchdf()

if df_liver_scale.empty:
    st.warning(f"{sel_liver_name} 暂无数据")
else:
    fig_liver = px.line(df_liver_scale, x="month",
                        y=["mau", "fixed_mau", "flowing_mau", "ylg_mau"],
                        labels={"value": "人数", "month": "月份", "variable": "分层"},
                        title=f"{sel_liver_name} 的 MAU 分层趋势")
    st.plotly_chart(fig_liver, use_container_width=True)

 # >>> 1.2 运营解读
if not df_liver_scale.empty:
        latest = df_liver_scale.tail(1).iloc[0]
        fix_ratio = latest["fixed_mau"] / latest["mau"] if latest["mau"] else 0
        prev_month = df_liver_scale.iloc[-2] if len(df_liver_scale) > 1 else latest
        prev_fix_ratio = prev_month["fixed_mau"] / prev_month["mau"] if prev_month["mau"] > 0 else 0
        trend = "上升" if fix_ratio > prev_fix_ratio else "下降"

if fix_ratio < 0.4:
    st.info(f"{sel_liver_name} 固定观众占比{fix_ratio:.1%}（{latest['fixed_mau']:,}人），"
            f"较上月{trend}{abs(fix_ratio-prev_fix_ratio):.1%}，处于不稳定状态。"
            f"流动观众{latest['flowing_mau']:,}人，YLG观众{latest['ylg_mau']:,}人。")
    st.info(
            "建议提高开播规律性和互动福利密度，建立观众观看习惯。")
elif fix_ratio > 0.7:
    st.info(f"{sel_liver_name} 固定观众占比高达{fix_ratio:.1%}（{latest['fixed_mau']:,}人），"
            f"较上月{trend}{abs(fix_ratio-prev_fix_ratio):.1%}，核心观众群体稳固。"
            f"流动观众{latest['flowing_mau']:,}人，YLG观众{latest['ylg_mau']:,}人。")
    st.info(
            "可尝试内容多元化测试，拓展新的观众群体，防止核心观众审美疲劳。")
else:
    st.info(f"{sel_liver_name} 固定观众占比{fix_ratio:.1%}（{latest['fixed_mau']:,}人），"
            f"较上月{trend}{abs(fix_ratio-prev_fix_ratio):.1%}，处于成长阶段。"
            f"流动观众{latest['flowing_mau']:,}人，YLG观众{latest['ylg_mau']:,}人。")
    st.info(
            "应重点关注高频互动但未固定的观众，设计专属活动提高转化率。")


# ====================== 1.3 行业渗透率堆叠面积图 ======================
st.header("1.3 行业渗透率堆叠面积图")
conn = get_conn()
pen_df = conn.execute("""
    SELECT month,
           liver,
           mau,
           industry_mau,
           penetration,
           pct_change
    FROM v_penetration_stacked
    ORDER BY month, liver
""").fetchdf()
pen_df["主播名"] = pen_df["liver"].map(id2name).fillna("YLG")
fig_pen = px.area(pen_df, x="month", y="penetration", color="主播名",
                  groupnorm="fraction",   # 自动堆叠 100%
                  title="主播 MAU 占行业比例（堆叠面积）")
st.plotly_chart(fig_pen, use_container_width=True)

# >>> 1.3 运营解读
last_month = pen_df["month"].max()
last_data = pen_df[pen_df["month"] == last_month]
top3 = last_data.nlargest(3, "penetration")
top3_penetration = top3["penetration"].sum()

if top3_penetration > 0.6:
    top_names = "、".join(top3["主播名"].tolist())
    st.info(f"{last_month.strftime('%Y年%m月')}，头部三位主播{top_names} "
            f"总渗透率达{top3_penetration:.1%}，市场集中度高。"
            f"其中{top3.iloc[0]['主播名']}渗透率最高({top3.iloc[0]['penetration']:.1%})。"
            "中腰部主播应寻找差异化定位，避免与头部主播直接竞争。")
elif top3_penetration > 0.4:
    st.info(f"{last_month.strftime('%Y年%m月')}，头部三位主播总渗透率{top3_penetration:.1%}，"
            "市场处于半集中状态。新主播仍有较大发展空间，应注重内容特色和社群运营。")
else:
    st.info(f"{last_month.strftime('%Y年%m月')}，头部三位主播总渗透率仅{top3_penetration:.1%}，"
            "市场分散，处于群雄混战期。所有主播都有机会通过优质内容和社群运营实现突破。")

# 解读
top1 = pen_df.sort_values("month").groupby("month").last().reset_index().tail(1)
if not top1.empty:
    name, pct, delta = top1.iloc[0]["主播名"], top1.iloc[0]["penetration"], top1.iloc[0]["pct_change"]
    st.info(f"{name} 最新月渗透率 {pct:.1%}，环比 {'+' if delta>=0 else ''}{delta:.1%} pct，基本盘仍在扩张。")

# ====================== 1.4 主播 S 曲线拟合 & 可视化 ======================
st.header("1.4 主播 S 曲线拟合（Logistic）")

from scipy.optimize import curve_fit

@st.cache_data(show_spinner=True)
def fit_logistic(df):
    """返回 DataFrame：liver, K, r, t0, R2"""
    def logistic(t, K, r, t0):
        return K / (1 + np.exp(-r * (t - t0)))

    res = []
    for liver, g in df.groupby("liver"):
        t, y = g["seq"].values, g["mau"].values
        if len(t) < 5:                      # 至少 5 个点才拟
            continue
        p0 = [y.max() * 1.2, 0.3, len(t) / 2]
        try:
            popt, _ = curve_fit(logistic, t, y, p0=p0, maxfev=5000)
            K, r, t0 = popt
            # 简单 R²
            y_pred = logistic(t, *popt)
            r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)
            res.append({"liver": liver, "K": K, "r": r, "t0": t0, "R2": r2})
        except RuntimeError:
            continue
    return pd.DataFrame(res)

# 1. 取数据（内存，不落盘）
scurve_df = conn.execute("""
    SELECT liver, month, mau, seq
    FROM v_scurve_data
    ORDER BY liver, seq
""").fetchdf()
scurve_df["主播名"] = scurve_df["liver"].map(id2name).fillna("YLG")

# 2. 侧边栏：选主播
sel = st.selectbox("选择主播", scurve_df["主播名"].unique())
sel_id = scurve_df.loc[scurve_df["主播名"] == sel, "liver"].iloc[0]
sub = scurve_df[scurve_df["liver"] == sel_id]

# 3. 拟合
if len(sub) < 5:
    st.warning("数据点不足 5 个，无法拟合")
else:
    fit_df = fit_logistic(scurve_df)
    params = fit_df[fit_df["liver"] == sel_id]
    if params.empty:
        st.warning("拟合失败，请换主播")
    else:
        K, r, t0, R2 = params.iloc[0][["K", "r", "t0", "R2"]]

        # 4. 画图
        t = sub["seq"].values
        y = sub["mau"].values
        t_fit = np.linspace(t.min(), t.max() + 2, 100)
        y_fit = K / (1 + np.exp(-r * (t_fit - t0)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='markers', name='实际 MAU', marker=dict(color='dodgerblue')))
        fig.add_trace(go.Scatter(x=t_fit, y=y_fit, mode='lines', name='Logistic 拟合', line=dict(color='crimson', width=3)))
        fig.update_layout(title=f"{sel} 的 S 曲线（R²={R2:.3f}）",
                          xaxis_title="Month Seq (t)",
                          yaxis_title="MAU")
        st.plotly_chart(fig, use_container_width=True)

        # =====  数值有效性检查  =====
        def valid_check(sub, params):
            """返回 (是否可信, 原因)"""
            n = len(sub)
            r2 = params.iloc[0]["R2"]
            k = params.iloc[0]["K"]
            y_max = sub["mau"].max()
            y_last = sub["mau"].iloc[-1]
            y_range = sub["mau"].max() - sub["mau"].min()

            if n < 8:
                return False, f"数据点不足 8 个（实际 {n}），拟合容易过拟合，建议忽略曲线"
            if r2 < 0.65:
                return False, f"R²={r2:.2f} < 0.65，拟合度差，参数不可信"
            if k / y_max < 1.05:
                return False, f"天花板 L={k:,.0f} 仅比历史峰值高 {k/y_max-1:.0%}，无增长空间提示"
            if y_range == 0:
                return False, "历史 MAU 完全没波动，无法估计增长参数"
            if y_last / y_max < 0.5:
                return False, "最近 MAU 已跌破峰值 50%，模型假设（最终趋于饱和）可能失效"
            return True, "拟合结果可信"

# 解读
        ok, reason = valid_check(sub, params)
        if not ok:
            st.warning(f"⚠️ {reason}；下方参数请仅作参考，不建议直接用于预算决策")
        else:
            st.success("✅ 拟合质量通过，可直接用于天花板与增长节奏预估")

        # 5. 运营解读
        st.info(f"天花板 L ≈ {K:,.0f}，增长率 r ≈ {r:.2f}，起飞点 t₀ ≈ 第 {t0:.0f} 月；"
                f"当前处于 {'成熟期' if t[-1] > t0 else '起飞期'}，资源倾斜可提前放量。")

        # >>> 1.4 运营解读
        if ok and t0 > 0:
            remain = max(0, K - y[-1])
            st.info(f"距离天花板还有 ≈{remain:,.0f} 空间；"
                    f"当前月增速 ≈{r*100:.1f}%/月，建议在「加速-峰值」阶段加大资源投放，"
                    "用 2-3 个月窗口把潜在渗透一次吃尽。")

        # 6. 批量下载参数（可选）
        csv_params = fit_df.to_csv(index=False)
        st.download_button("下载全主播拟合参数", csv_params, "scurve_params.csv", "text/csv")

# >>> 1.4 运营解读
if ok and t0 > 0:
    remain = max(0, K - y[-1])
    st.info(f"距离天花板还有 ≈{remain:,.0f} 空间；"
            f"当前月增速 ≈{r*100:.1f}%/月，建议在「加速-峰值」阶段加大资源投放，"
            "用 2-3 个月窗口把潜在渗透一次吃尽。")

# ====================== 1.5 主播生命周期五阶段  ======================
st.header("1.5 主播生命周期五阶段")

five_df = conn.execute("""
    SELECT liver, month, mau, seq, stage
    FROM v_five_stage_all
    ORDER BY liver, seq
""").fetchdf()
five_df["主播名"] = five_df["liver"].map(id2name).fillna("YLG")

sel = st.selectbox("选择主播（看全周期）", five_df["主播名"].unique())
sub = five_df[five_df["主播名"] == sel]

if sub.empty:
    st.warning("该主播暂无五阶段数据")
else:
    # 彩带面积图
    fig = px.area(sub, x="seq", y="mau", color="stage",
                  title=f"{sel} 生命周期五阶段",
                  color_discrete_map={
                      "起飞期": "#1f77b4",
                      "加速期": "#ff7f0e",
                      "峰值冲刺期": "#2ca02c",
                      "增速放缓期": "#d62728",
                      "回落预警期": "#9467bd",
                      "衰退期": "#8c564b"
                  })
    fig.update_layout(xaxis_title="Month Seq (t)", yaxis_title="MAU")
    st.plotly_chart(fig, use_container_width=True)

    # 解读
    curr = sub.iloc[-1]["stage"]
    st.info(f"当前处于 **{curr}**，"
            + ("建议提前布局回流活动" if "回落预警" in curr else "继续保持资源投放"))
            # >>> 1.5 运营解读（追加）
    if "衰退" in curr:
        st.info("进入回落预警/衰退期，需立即启动「第二增长曲线」："
                "内容翻新或跨界联动，否则 MAU 可能持续阴跌。")

# ====================== 2. 热度维度 ======================
# ====================== 2.1 兴趣关联 ======================
st.markdown("# 2. 热度维度")
st.header("2.1 兴趣关联")

DEFAULT_TARGETS, DEFAULT_TOP_N, DEFAULT_EXCLUDE = ["嘉然"], 8, False

@st.cache_data(show_spinner=True)
def _compute_assoc(target_ids: tuple, top_n: int, exclude: bool):
    conn = get_conn()
    src_df = conn.execute(f"""
        SELECT source_liver AS liver, SUM(cnt) AS cnt
        FROM monthly_matrix_in
        WHERE target_liver IN {target_ids}
        GROUP BY source_liver
    """).fetchdf()
    tgt_df = conn.execute(f"""
        SELECT target_liver AS liver, SUM(cnt) AS cnt
        FROM monthly_matrix_out
        WHERE source_liver IN {target_ids}
        GROUP BY target_liver
    """).fetchdf()
    total_df = pd.concat([src_df, tgt_df]).groupby("liver", as_index=False)["cnt"].sum()
    total_df = total_df[~total_df["liver"].isin(target_ids)]
    if exclude:
        total_df = total_df[total_df["liver"] != -3]
    if total_df.empty:
        return None
    top = total_df.sort_values("cnt", ascending=False).head(top_n)
    rest_cnt = total_df.iloc[top_n:]["cnt"].sum()
    if rest_cnt > 0:
        top = pd.concat([top, pd.DataFrame({"liver": [-999], "cnt": [rest_cnt]})])
        id2name[-999] = "Others"
    top["主播名"] = top["liver"].map(id2name).fillna("YLG") # .fillna("YLG") 是为了处理 -3 这个特殊值
    return top

def _show_assoc(top_df):
    fig = px.pie(top_df, names="主播名", values="cnt",
                 title=f"Users who like {'/'.join(top_df.loc[top_df['liver']!=-999, '主播名'])} also like",
                 color_discrete_sequence=px.colors.sequential.YlGnBu_r)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with st.form("assoc_form"):
    target_names = st.multiselect("Target streamers", names, default=DEFAULT_TARGETS, key="a1")
    target_ids   = tuple([k for k, v in id2name.items() if v in target_names])
    exclude      = st.checkbox("Exclude YLG (-3)", value=DEFAULT_EXCLUDE)
    top_n        = st.slider("Top N in pie", 3, 15, DEFAULT_TOP_N)
    run          = st.form_submit_button(T["gen"], use_container_width=True)

if st.button("🗑 清兴趣关联缓存"):
    _compute_assoc.clear()
    st.success("缓存已清空，请重新生成！")

if "assoc_auto" not in st.session_state:
    top_df = _compute_assoc(tuple([k for k, v in id2name.items() if v in DEFAULT_TARGETS]),
                            DEFAULT_TOP_N, DEFAULT_EXCLUDE)
    if top_df is not None:
        _show_assoc(top_df)
    st.session_state.assoc_auto = True

# >>> 2.1 运营解读
if top_df is not None and len(top_df) > 1:
    cross_rate = top_df[top_df["liver"] != -999]["cnt"].sum() / top_df["cnt"].sum()
    top_related = top_df[top_df["liver"] != -999].nlargest(3, "cnt")
    top_names = "、".join([f"{row['主播名']}({row['cnt']}人)" for _, row in top_related.iterrows()])
    
    if cross_rate > 0.6:
        st.info(f"观众跨主播流动性高({cross_rate:.1%})，最相关的三位主播是：{top_names}。"
                "可尝试与这些主播进行「连麦/联播」活动，将共同兴趣转化为双向增粉机会。")
    else:
        st.info(f"观众粘性较高，跨主播流动性仅{cross_rate:.1%}，最相关的三位主播是：{top_names}。"
                "适合打造「深度私域」生态，通过会员体系和专属互动玩法提升用户忠诚度和ARPPU。")

if run:
    top_df = _compute_assoc(target_ids, top_n, exclude)
    if top_df is None:
        st.warning("No data")
    else:
        _show_assoc(top_df)



# ====================== 2.2 事件类型拆分 ======================
# ====================== 2.2 事件类型拆分 ======================
st.header("2.2 事件类型拆分")
sel_hot_names, sel_hot_ids = select_livers("hot")

df_hot = conn.execute(f"""
    SELECT day,
           SUM(CASE WHEN liver IN {sel_hot_ids} THEN weak ELSE 0 END) AS weak,
           SUM(CASE WHEN liver IN {sel_hot_ids} THEN strong ELSE 0 END) AS strong,
           SUM(CASE WHEN liver IN {sel_hot_ids} THEN total ELSE 0 END) AS total
    FROM daily_events_by_liver
    GROUP BY day
    ORDER BY day
""").fetchdf()

df_hot = df_hot.melt(id_vars="day", value_vars=["weak","strong"], var_name="type", value_name="cnt")
fig_hot = px.bar(df_hot, x="day", y="cnt", color="type", barmode="stack",
                 title=f"每日事件量（{'全体' if not sel_hot_names else '/'.join(sel_hot_names)}）")
st.plotly_chart(fig_hot, use_container_width=True)

# >>> 2.2 运营解读
weak_share = df_hot[df_hot["type"] == "weak"]["cnt"].sum() / df_hot["cnt"].sum()
strong_share = 1 - weak_share

# 获取最近7天数据对比
recent_7d = df_hot[df_hot["day"] > df_hot["day"].max() - pd.Timedelta(days=7)]
weak_recent = recent_7d[recent_7d["type"] == "weak"]["cnt"].sum() / recent_7d["cnt"].sum() if not recent_7d.empty else 0

trend = "上升" if weak_recent > weak_share else "下降"

if weak_share > 0.6:
    st.info(f"弱互动事件占比{weak_share:.1%}，近期{trend}，观众以围观为主。"
            f"强互动事件仅占{strong_share:.1%}。建议通过打卡任务、弹幕互动等活动，"
            "将弱互动用户转化为强互动用户，提高社区活跃度。")
elif weak_share > 0.4:
    st.info(f"互动事件分布相对均衡，弱互动占{weak_share:.1%}，强互动占{strong_share:.1%}，"
            f"近期趋势{trend}。可针对不同互动层级用户设计差异化活动，"
            "提高整体参与度。")
else:
    st.info(f"强互动事件占比{strong_share:.1%}，占主导地位，近期趋势{trend}。"
            "社区活跃度高，可引导强互动用户参与付费内容或二创生产，"
            "进一步提升用户价值和社区生态。")

# ====================== 3. 健康维度 ======================
# ====================== 3.1 月度趋势 ======================
st.markdown("# 3. 健康维度")
st.header("3.1 月度趋势")

sel_trend = st.multiselect(T["select"], names, default=["嘉然"], key="trend_sel")
sel_ids_t = [k for k, v in id2name.items() if v in sel_trend]
if sel_ids_t:
    conn = get_conn()
    df_trend = conn.execute(f"""
        SELECT DATE_TRUNC('month', month) AS month,
               SUM(CASE WHEN target_liver IN {tuple(sel_ids_t)} THEN cnt ELSE 0 END) AS new_users,
               SUM(CASE WHEN source_liver IN {tuple(sel_ids_t)} THEN cnt ELSE 0 END) AS lost_users
        FROM monthly_matrix_in
        GROUP BY month ORDER BY month
    """).fetchdf()
    st.line_chart(df_trend.set_index("month")[["new_users", "lost_users"]])

# >>> 3.1 运营解读
if not df_trend.empty:
    net = df_trend["new_users"].sum() - df_trend["lost_users"].sum()
    if net > 0:
        st.info("净流入为正，说明主播矩阵整体吸粉；可把增量资源投向吸粉效率最高的月份/主播，放大正循环。")
    else:
        st.info("净流入为负，需先止血：定位流失高峰月，重点召回当月高价值互动用户，再谈增长。")


# ====================== 3.2 用户流动矩阵 ======================
st.header("3.2 用户流动矩阵")

sel_names = st.multiselect(T["select"], names, default=["嘉然"], key="matrix_sel")
sel_ids   = [k for k, v in id2name.items() if v in sel_names]
if sel_ids:
    cache_key = f"{'-'.join(map(str, sel_ids))}_{datetime.now():%Y-%m}"
    src_cache = CACHE_DIR / f"src_{cache_key}.parquet"
    tgt_cache = CACHE_DIR / f"tgt_{cache_key}.parquet"

    if st.button(T["reload"], key="matrix_reload"):
        src_cache.unlink(missing_ok=True); tgt_cache.unlink(missing_ok=True)

    @st.cache_data(show_spinner=True)
    def compute_matrix(_ids):
        conn = get_conn()
        src = conn.execute(f"""
            SELECT month, source_liver, SUM(cnt) cnt
            FROM monthly_matrix_in
            WHERE target_liver IN {tuple(_ids)}
            GROUP BY month, source_liver
        """).fetchdf()
        tgt = conn.execute(f"""
            SELECT month, target_liver, SUM(cnt) cnt
            FROM monthly_matrix_out
            WHERE source_liver IN {tuple(_ids)}
            GROUP BY month, target_liver
        """).fetchdf()
        src["month"] = src["month"].dt.strftime("%Y-%m")
        tgt["month"] = tgt["month"].dt.strftime("%Y-%m")
        src["主播"] = src["source_liver"].map(id2name)
        tgt["主播"] = tgt["target_liver"].map(id2name)
        src.to_parquet(src_cache, index=False)
        tgt.to_parquet(tgt_cache, index=False)
        return src, tgt

    if src_cache.exists() and tgt_cache.exists():
        src, tgt = pd.read_parquet(src_cache), pd.read_parquet(tgt_cache)
    else:
        src, tgt = compute_matrix(sel_ids)

    src_tbl = src.pivot_table(index="month", columns="主播", values="cnt", fill_value=0).astype(int)
    tgt_tbl = tgt.pivot_table(index="month", columns="主播", values="cnt", fill_value=0).astype(int)

    st.subheader(T["src_table"])
    st.dataframe(src_tbl.style.background_gradient(cmap="YlGnBu"))
    st.subheader(T["src_heat"])
    st.plotly_chart(px.imshow(src_tbl, labels=dict(x="主播", y="月份", color="人数"),
                                color_continuous_scale="YlGnBu", aspect="auto"), use_container_width=True)

    st.subheader(T["tgt_table"])
    st.dataframe(tgt_tbl.style.background_gradient(cmap="YlGnBu"))
    st.subheader(T["tgt_heat"])
    st.plotly_chart(px.imshow(tgt_tbl, labels=dict(x="主播", y="月份", color="人数"),
                                color_continuous_scale="YlGnBu", aspect="auto"), use_container_width=True)


# >>> 3.2 运营解读
if src_tbl.shape[1] > 1:
    max_src = src_tbl.iloc[-1].idxmax()
    max_tgt = tgt_tbl.iloc[-1].idxmax()
    st.info(f"最近月份最大来源={max_src}，最大去向={max_tgt}；"
            "可针对「来源」做联合直播，针对「去向」做流失预警召回。")

# ====================== 3.3 AARRR 漏斗 ======================
# ====================== 3.3 AARRR 漏斗 ======================
st.header("3.3 AARRR 漏斗")
sel_aarr_names, sel_aarr_ids = select_livers("aarr")

avail_raw = conn.execute(f"""
    SELECT DISTINCT month
    FROM aarr_metrics
    WHERE liver IN {sel_aarr_ids}
""").fetchdf()["month"]
avail = [d.strftime("%Y-%m") for d in sorted(avail_raw)]

if not avail:
    st.warning("所选主播无 AARRR 数据")
    st.stop()

funnel_month = st.selectbox("Select month", avail)

raw = conn.execute("""
    SELECT SUM(acq)       AS acq,
           SUM(activ)     AS activ,
           SUM(reten)     AS reten,
           SUM(refer)     AS refer,
           SUM(revenue)   AS revenue
    FROM aarr_metrics
    WHERE month = ? AND liver IN ?
""", [pd.to_datetime(funnel_month), sel_aarr_ids]).fetchdf().fillna(0).iloc[0]

funnel = {
    "acq":   int(raw.acq),
    "activ": int(raw.activ),
    "reten": int(raw.reten),
    "refer": int(raw.refer),
    "revenue": int(raw.revenue),
}

fig_f = go.Figure(go.Funnel(
        y=[T["funnel_acquisition"],
           T["funnel_activation"],
           T["funnel_retention"],
           T["funnel_revenue"],
           T["funnel_referral"]],
        x=[funnel["acq"],
           funnel["activ"],
           funnel["reten"],
           funnel["revenue"],   # 统一用 revenue
           funnel["refer"]],
        textinfo="value+percent initial"))
st.plotly_chart(fig_f, use_container_width=True)


# >>> 3.3 运营解读
rates = {"激活": funnel["activ"] / max(funnel["acq"], 1),
         "留存": funnel["reten"] / max(funnel["acq"], 1),
         "推荐": funnel["refer"] / max(funnel["acq"], 1)}
min_stage, min_rate = min(rates.items(), key=lambda x: x[1])
if min_rate < 0.2:
    st.info(f"{min_stage} 转化率仅 {min_rate:.1%}，为当前最短漏斗板；"
            "优先补强该环节，整体 ROI 提升最明显。")

# ====================== 3.4 MAU 分层转化漏斗 ======================
# ====================== 3.4 MAU 分层转化漏斗 ======================
st.header("3.4 MAU 分层转化漏斗")
sel_lf_names, sel_lf_ids = select_livers("lf")


avail_raw = conn.execute(f"""
    SELECT DISTINCT cohort_month
    FROM v_layer_funnel
    WHERE liver IN {sel_lf_ids}
""").fetchdf()["cohort_month"]
avail = [d.strftime("%Y-%m") for d in sorted(avail_raw)]

if not avail:
    st.warning("所选主播无分层转化数据")
    st.stop()

funnel_month = st.selectbox("选择 cohort 月份", avail, index=len(avail)-1)

funnel_df = conn.execute("""
    SELECT stage, SUM(users) users, SUM(users)*1.0/FIRST_VALUE(SUM(users)) OVER (PARTITION BY 1) pct
    FROM v_layer_funnel
    WHERE cohort_month = ? AND liver IN ?
    GROUP BY stage
    ORDER BY stage
""", [pd.to_datetime(funnel_month), sel_lf_ids]).fetchdf()

if len(funnel_df) >= 3:
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_df["stage"],
        x=funnel_df["users"],
        textinfo="value+percent previous"
    ))
    st.plotly_chart(fig_funnel, use_container_width=True)
    new_, fix2 = funnel_df.iloc[0]["users"], funnel_df.iloc[2]["users"]
    st.info(f"{funnel_month} 新增 {new_:,.0f}，两个月后沉淀固定 {fix2:,.0f}，转化率 {fix2/new_:.1%}。")
else:
    st.warning(f"{funnel_month} 数据不足，无法展示完整三阶段漏斗。")


# ====================== 3.5 流动层净流失率折线 ======================
st.header("3.5 流动层净流失率折线")
sel_churn_names, sel_churn_ids = select_livers("churn")

churn_df = conn.execute(f"""
    SELECT month, SUM(net_flow) net_flow, AVG(churn_rate) churn_rate
    FROM flowing_net_churn_liver
    WHERE liver IN {sel_churn_ids}
    GROUP BY month
    HAVING churn_rate IS NOT NULL
    ORDER BY month
""").fetchdf()

if churn_df.empty:
    st.warning("所选主播无流动层数据")
else:
    fig_churn = px.line(churn_df, x="month", y="churn_rate",
                        title=f"流动层净流失率（{'全体' if not sel_churn_names else '/'.join(sel_churn_names)}）")
    st.plotly_chart(fig_churn, use_container_width=True)
    latest = churn_df.tail(1).iloc[0]
    m, r = latest["month"], latest["churn_rate"]
    st.info(f"{m:%Y-%m} 净流失率 {r:.1%}，连续收窄中，掉血趋缓。" if r < 0 else f"{m:%Y-%m} 净流失率 {r:.1%}，需回流运营。")


# >>> 3.5 运营解读
if abs(r) < 0.03:
    st.info("净流失率趋近于 0，流动层基本平衡；此时可尝试「付费转化」或「二创激励」，把平衡态推向增量态。")
else:
    st.info("净流失率绝对值仍高，优先做「流失预警」+「召回触达」，避免失血过快消耗基本盘。")

# ====================== 4. 忠诚维度 ======================
# ====================== 4.1 固定观众留存 ======================
st.markdown("# 4. 忠诚维度")
st.header("4.1 固定观众留存")

df_cohort = conn.execute("""
    SELECT cohort_month, month_age, retained,
           FIRST_VALUE(retained) OVER (PARTITION BY cohort_month ORDER BY month_age) AS acquired
    FROM cohort_retention_30d
    WHERE month_age <= 3   -- 只看 0-1-2-3 个月
""").fetchdf()
df_cohort["留存率"] = df_cohort["retained"] / df_cohort["acquired"]
fig_ret = px.line(df_cohort, x="month_age", y="留存率", color="cohort_month",
                  markers=True, title="固定观众 Cohort 留存（30-60-90 日）")
st.plotly_chart(fig_ret, use_container_width=True)

# >>> 4.1 运营解读
avg_ret = df_cohort[df_cohort["month_age"] == 1]["留存率"].mean()
if avg_ret < 0.35:
    st.info("30 日留存低于 35%，需强化「首播 7 日体验」：打卡、弹幕彩蛋、粉丝牌任务，降低早期流失。")
else:
    st.info("30 日留存健康，可把重点放在 60-90 日「深度留存」：会员日、专属直播、二创征集。")


# ====================== 5. 贡献维度 ======================
# ====================== 5.1 RFM 分层 ======================
st.header("5.1 RFM 分层")
sel_rfm_names, sel_rfm_ids = select_livers("rfm", default=[])

rfm_df = conn.execute(f"""
    SELECT rfm_code, rfm_tag, COUNT(*) cnt
    FROM rfm_user
    WHERE liver IN {sel_rfm_ids}
    GROUP BY rfm_code, rfm_tag
    ORDER BY rfm_code
""").fetchdf()

col1, col2 = st.columns(2)
with col1:
    st.subheader(T["rfm_score"])
    st.bar_chart(rfm_df["rfm_code"].value_counts().sort_index())
with col2:
    st.subheader(T["rfm_segment"])
    st.bar_chart(rfm_df["rfm_tag"].value_counts())

# >>> 5.1 运营解读
high_val = rfm_df[rfm_df["rfm_tag"] == "高价值忠诚"]["cnt"].sum()
if high_val / rfm_df["cnt"].sum() < 0.1:
    st.info("高价值忠诚人群不足 10%，说明付费或深度互动转化弱；"
            "可先做「小金额打赏激励」+「荣誉榜单」测试，提高 ARPPU。")
else:
    st.info("高价值群体占比已高，下一步用「等级会员」「生日直播」进一步延长 LTV。")

# ====================== 5.2 聚类散点 ======================
st.header("5.2 聚类散点")

DEFAULT_SEL, DEFAULT_MAX_U, DEFAULT_K, DEFAULT_EXCLUDE = [], 3000, 4, True

@st.cache_data(show_spinner=True)
def _compute_cluster(_ids: tuple, max_u: int, k: int, exclude: bool):
    conn = get_conn()
    cond_liver = f"AND liver IN {_ids}" if _ids else ""
    cond_ylg   = "AND liver != -3" if exclude else ""
    df = conn.execute(f"""
        SELECT uid, liver
        FROM events
        WHERE 1=1 {cond_liver} {cond_ylg}
    """).fetchdf()
    if df.empty:
        return None, None, None
    top_u = df["uid"].value_counts().head(max_u).index
    df    = df[df["uid"].isin(top_u)]
    matrix = df.assign(flag=1).pivot_table(index="uid", columns="liver", values="flag", fill_value=0)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None, None, None
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(matrix)
    pca    = PCA(2, random_state=42).fit_transform(matrix)
    plot_df = pd.DataFrame(pca, columns=["x", "y"])
    plot_df["cluster"] = labels.astype(str)
    return plot_df, labels, matrix

def _show_cluster(plot_df, labels, matrix):
    fig = px.scatter(plot_df, x="x", y="y", color="cluster",
                     title=f"{len(matrix)} users × {len(plot_df['cluster'].unique())} clusters")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top5 streamers per cluster")
    for c in sorted(plot_df["cluster"].unique()):
        idx  = labels == int(c)
        top5 = matrix[idx].mean().sort_values(ascending=False).head(5)
        st.write(f"**Cluster {c}**: " + ", ".join([id2name[i] for i in top5.index]))

with st.form("cluster_form"):
    sel_names = st.multiselect(T["select"], names, default=DEFAULT_SEL, key="c1")
    sel_ids   = tuple([k for k, v in id2name.items() if v in sel_names])
    total_u   = get_conn().execute("SELECT COUNT(DISTINCT uid) FROM events").fetchone()[0]
    max_u     = st.slider("Max users", 100, int(total_u), min(DEFAULT_MAX_U, int(total_u)), 100)
    k         = st.slider("Cluster count", 2, 10, DEFAULT_K)
    exclude   = st.checkbox("Exclude YLG", value=DEFAULT_EXCLUDE)
    run       = st.form_submit_button(T["gen"], use_container_width=True)

if st.button("🗑 清聚类缓存"):
    _compute_cluster.clear()
    st.success("缓存已清空，请重新生成！")

if "cluster_auto" not in st.session_state:
    plot_df, labels, matrix = _compute_cluster(tuple(DEFAULT_SEL), DEFAULT_MAX_U, DEFAULT_K, DEFAULT_EXCLUDE)
    if plot_df is not None:
        _show_cluster(plot_df, labels, matrix)
    st.session_state.cluster_auto = True

if run:
    plot_df, labels, matrix = _compute_cluster(sel_ids, max_u, k, exclude)
    if plot_df is None:
        st.warning("数据过少或无数据")
    else:
        _show_cluster(plot_df, labels, matrix)

# >>> 5.2 运营解读
if labels is not None:
    cluster_cnt = len(set(labels))
    if cluster_cnt <= 3:
        st.info("聚类≤3 组，观众兴趣分化不明显；可大胆尝试跨品类内容，扩大触达面。")
    else:
        st.info("聚类>3 组，观众兴趣多元；建议为主力 Cluster 定制专属栏目，为长尾 Cluster 做轻量级彩蛋，实现分层运营。")

# ====================== 5.3 RFM 各层事件占比 ======================
st.header("5.3 RFM 各层事件占比")
sel_rc_names, sel_rc_ids = select_livers("rc", default=[])

df_contrib = conn.execute(f"""
    SELECT day, rfm_tag, SUM(evt_ratio) AS evt_ratio
    FROM rfm_daily_contrib
    WHERE liver IN {sel_rc_ids}
    GROUP BY day, rfm_tag
    ORDER BY day, rfm_tag
""").fetchdf()

if df_contrib.empty:
    st.warning("所选主播无 RFM 占比数据")
else:
    fig_contrib = px.area(df_contrib, x="day", y="evt_ratio", color="rfm_tag",
                          title=f"每日互动量中各 RFM 层占比（{'全体' if not sel_rc_names else '/'.join(sel_rc_names)}）")
    st.plotly_chart(fig_contrib, use_container_width=True)


# >>> 5.3 运营解读
latest_share = df_contrib.dropna().sort_values("day").groupby("rfm_tag").tail(1)["evt_ratio"]
loss_tag = latest_share.idxmin()
if latest_share[loss_tag] < 0.15:
    st.info(f"「{loss_tag}」层事件占比过低，存在流失风险；"
            "可用专属弹幕色、粉丝牌升级任务等方式，把该层用户重新拉回高互动区间。")
else:
    st.info("各层事件占比相对均衡，继续保持现有分层运营节奏即可。")


# ------------------------------ 增量更新 -------------------------------
st.header(T["update"])
if st.button(T["start"], key="update_btn"):
    with st.spinner("Pulling..."):
        old = pd.read_parquet(DATA_PATH)
        uids = old["uid"].unique()
        rows = []
        for uid in uids:
            r = requests.get("https://danmakus.com/api/v2/user/watchedChannels", params={"uid": uid}, timeout=10)
            if r.status_code != 200: continue
            for item in r.json().get("data", []):
                rows.append({"uid": uid, "ts": pd.to_datetime(item["lastLiveDate"], unit="ms"), "liver": int(item["uId"])})
        new = pd.DataFrame(rows)
        new["key"] = new["uid"].astype(str) + "_" + new["ts"].astype(str) + "_" + new["liver"].astype(str)
        old["key"] = old["uid"].astype(str) + "_" + old["ts"].astype(str) + "_" + old["liver"].astype(str)
        new = new[~new["key"].isin(old["key"])].drop(columns="key")
        if not new.empty:
            pd.concat([old, new]).to_parquet(DATA_PATH, index=False)
            st.success(f"✅ Added {len(new)} rows")
        else:
            st.info("No new rows")


