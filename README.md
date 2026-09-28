# Cookie Cats A/B 测试与实验分析报告

## 📖 项目简介
基于 Kaggle 公开数据集（90,189 条真实用户数据），验证将游戏新手关卡从 30 级调整至 40 级对用户留存的影响。本项目完成了从数据获取、统计检验到业务决策的端到端闭环分析。

## 📊 核心结论
- **主指标（7日留存）**：gate_30 组 19.02% vs gate_40 组 18.20%（Δ-0.82pp，p=0.0016，95% CI [-1.33pp, -0.31pp]）。
- **统计方法**：两比例 Z 检验、Bootstrap 重抽（20,000 次）、贝叶斯后验分析。
- **进阶分析**：SRM 校验、CUPED 方差缩减、异质性分析（森林图）、ROI 敏感性测算。
- **业务决策**：不推荐上线 40 级门槛，维持 30 级（预计年化留存收入损失约 ¥59 万）。

## 🛠️ 技术栈
Python (pandas, scipy, statsmodels), ECharts, HTML/CSS

## 🔗 在线报告
👉 [点击查看完整交互式 HTML 报告](https://Nanak0-0.github.io/cookie-cats-ab-test/)

## 📂 文件说明
- `index.html`: 单文件离线交互报告（内联 ECharts，断网可看）
- `cookie_cats_analysis.py`: 完整分析代码（固定随机种子，可复现）
- `cookie_cats.csv`: 原始数据集
- `analysis_results.txt`: 统计结果数据
- `verify_offline.py`: 离线质检脚本（断网+多视口自动验证）

## 👤 作者
程杰伟 | 数据分析求职方向
[2927503740@qq.com]