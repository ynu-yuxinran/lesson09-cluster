"""
Step2：肘部法则（Elbow Method）—— 用 SSE 曲线验证 k=4 是否合理
运行环境：在本文件夹（bank_customers.csv 所在目录）下执行 python step2_elbow.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========== 解决 matplotlib 中文显示成方块的问题 ==========
# 指定使用黑体（SimHei）渲染中文；同时让负号能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取数据并取四个特征 ==========
df = pd.read_csv("bank_customers.csv")
feature_cols = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X = df[feature_cols].fillna(0)  # 缺失值填 0，防止标准化时报错

# ========== 2. 标准化 ==========
# 和 Step1 一样：四个特征量纲差别很大，必须先标准化成“均值 0、标准差 1”
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 3. 循环尝试 k=1~9，记录每个 k 的 SSE ==========
k_values = range(1, 10)   # k 从 1 到 9
sse_list = []             # 用来保存每个 k 对应的 SSE（误差平方和）

for k in k_values:
    # 固定 random_state=42，保证结果可复现
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    # inertia_ 就是 SSE：所有点到各自簇中心的距离平方和，越小说明簇越紧凑
    sse_list.append(kmeans.inertia_)
    print(f"k={k:>2d}  SSE={kmeans.inertia_:>10.4f}")

print("\nk=1 到 9 的 SSE 数值列表：")
print([round(sse, 2) for sse in sse_list])

# ========== 4. 画折线图 ==========
plt.figure(figsize=(9, 5))
plt.plot(list(k_values), sse_list, marker="o", color="#1f77b4", linewidth=2, markersize=8)

# 把每个点的 SSE 数值标注在折线上方
for k, sse in zip(k_values, sse_list):
    plt.annotate(f"{sse:.0f}",                    # 文本内容（保留整数，避免太挤）
                 xy=(k, sse),                    # 标注的位置（点本身）
                 xytext=(0, 10),                 # 文字向上偏移 10 个像素
                 textcoords="offset points",
                 ha="center")                    # 水平居中

# 标题与坐标轴标签
plt.title("KMeans 肘部法则：不同 k 值对应的 SSE")
plt.xlabel("簇数 k")
plt.ylabel("SSE（inertia，误差平方和）")
plt.xticks(list(k_values))                       # 横轴只显示 1~9 的整数刻度
plt.grid(alpha=0.3)                              # 浅色网格，方便看趋势

# 保存到当前文件夹，再弹窗显示
plt.savefig("elbow_plot.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n已保存图片：elbow_plot.png")
print("Step2 运行成功")
