
# VTuber-Flow-Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io/)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.9+-orange.svg)](https://duckdb.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README_EN.md) | 中文

> [!NOTE]
> **基于 [哔哩哔哩【V圈大数据】](https://b23.tv/cQoNsjj)数据参考，仅供学习与交流。**

> [!IMPORTANT]
> **本项目仅作为数据分析的学习实践结果，所有数据均为网络公开数据脱敏后示例，不含任何个人隐私，亦不构成商业建议。**

---

## 项目简介

一站式 VTuber 运营流动数据分析可视化仪表盘，支持：
- 主播间用户流入 & 流出矩阵
- 用户聚类 & 兴趣关联
- AARRR 漏斗 & RFM 分层
- 增量数据自动更新

## 技术栈

Python、DuckDB、Streamlit、Plotly、Scikit-learn

## 安装指南

### 前置要求
- Python 3.10+
- 1 GB 以上内存

### 安装步骤

1. **克隆仓库**：
```bash
git clone https://github.com/zengplus/VTuber-Flow-Analysis.git
cd VTuber-Flow-Analysis
```

2. **安装依赖**：

**方式 A：使用 uv **
```bash
uv sync
```

**方式 B：使用 pip **
```bash
pip install -r requirements.txt
```

3. **数据库构建**：
```bash
duckdb mydb.duckdb < analysis.sql
```

## 使用说明

### 启动应用
```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`

### 界面操作指南

#### 1. 侧边栏参数设置
- 设置语言（中文/英文）

#### 2. 结果展示
- 月度用户流动趋势分析
    - 月度用户流动趋势折线图
- 用户流动矩阵分析
    - 用户来源分析表
    - 用户来源热图
    - 用户流失去向表
    - 用户去向热图
- AARRR 漏斗分析
    - AARRR转化漏斗图
- RFM 用户分层
    - RFM得分分布条形图
    - RFM分层结果条形图
- 用户群体聚类分析
    - 用户群体聚类散点图
    - 聚类群体兴趣分布
- 用户兴趣关联分析
    - 用户兴趣关联饼图

### 界面预览

#### Demo演示
![Demo](images/demo.gif)

## 项目结构

```
VTuber-Flow-Analysis/
├── __pycache__/                # 运行时生成：Python缓存目录
│   └── ...
├── cache/                      # 运行时生成：数据缓存目录
│   └── ...
├── data/                       # 数据存储目录
│   └── fans_events.parquet     # 原始数据文件
├── images/                     # 图片资源目录
│   └── demo.gif                # 主页演示动画
├── analysis.sql                # 数据分析SQL脚本
├── app.py                      # 主应用程序
├── id2name.py                  # ID与名称映射
├── pyproject.toml              # 项目元数据
├── README.md                   # 项目说明文档
├── mydb.duckdb                 # 运行时生成：DuckDB数据库
├── requirements.txt            # 顶层依赖
├── uv.lock                     # uv 锁定文件
└── LICENSE                     # 项目许可证文件
```

---

## ⚠️ 重要提示

**重要提醒**：本项目仅为学习演示，所有预测结果不代表真实情况！
