"""
Step5：独立研究任务 —— 验证 KMeans 对环形/月牙数据的局限，并计算真实数据的轮廓系数
运行环境：在本文件夹（bank_customers.csv 所在目录）下执行 python step5_research.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, DBSCAN          # 对比两种聚类算法
from sklearn.metrics import silhouette_score       # 轮廓系数
from sklearn.preprocessing import StandardScaler

# ========== 解决 matplotlib 中文显示成方块的问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================================================================
# 第一部分：环形/月牙数据实验
# ======================================================================
print("=" * 78)
print("实验一：KMeans vs DBSCAN 在月牙形数据上的表现")
print("=" * 78)

# 1. 生成两弯月牙形数据：400 个点，noise=0.08 让数据带一点扰动，更接近真实情况
X_moons, y_true = make_moons(n_samples=400, noise=0.08, random_state=42)

# 2. KMeans 聚类：KMeans 假设簇是“围绕中心的球形团”，只能按直线边界切分
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_moons)

# 3. DBSCAN 聚类：基于“密度”，把彼此靠近、连成一片的点归为一簇
dbscan = DBSCAN(eps=0.25, min_samples=5)   # eps=邻域半径，min_samples=成为核心点所需的最少邻居数
labels_db = dbscan.fit_predict(X_moons)

# 统计两种算法的结果，顺便看 DBSCAN 有没有把点判成噪声（标签为 -1）
print(f"KMeans 各簇人数：{dict(zip(*np.unique(labels_km, return_counts=True)))}")
db_valid = labels_db[labels_db != -1]
n_db_clusters = len(set(db_valid))
n_noise = int(np.sum(labels_db == -1))
print(f"DBSCAN 识别出的簇数：{n_db_clusters}，噪声点：{n_noise} 个")

# 4. 画 1 行 2 列对比图
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左图：KMeans 的结果
ax1 = axes[0]
ax1.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_km, cmap="coolwarm",
            s=35, edgecolors="k", linewidths=0.3)
# 把 KMeans 的两个簇中心画成黑色 ×，让“一刀切”的边界更直观
ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker="X", s=220, c="black", edgecolors="yellow", linewidths=1.2,
            label="簇中心")
ax1.set_title("KMeans：用直线边界切开，月牙被拦腰截断", fontsize=12)
ax1.set_xlabel("特征 1", fontsize=10)
ax1.set_ylabel("特征 2", fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)

# 右图：DBSCAN 的结果（噪声点用灰色单独标出）
ax2 = axes[1]
colors_db = plt.cm.viridis(np.linspace(0, 1, max(n_db_clusters, 1)))
for lab in np.unique(labels_db):
    mask = labels_db == lab
    if lab == -1:                      # 噪声点
        ax2.scatter(X_moons[mask, 0], X_moons[mask, 1], c="lightgray",
                    s=35, edgecolors="k", linewidths=0.3, label="噪声点")
    else:
        ax2.scatter(X_moons[mask, 0], X_moons[mask, 1], c=[colors_db[lab]],
                    s=35, edgecolors="k", linewidths=0.3, label=f"簇 {lab}")
ax2.set_title("DBSCAN：顺着密度连成月牙，边界贴合形状", fontsize=12)
ax2.set_xlabel("特征 1", fontsize=10)
ax2.set_ylabel("特征 2", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.2)

fig.suptitle("环形/月牙数据聚类对比：KMeans 失效，DBSCAN 成功", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.93])

# 5. 保存图片
plt.savefig("research_dbscan.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n>> 结论：")
print("   KMeans 失败的原因：它按“到簇中心的距离”划分，隐含假设簇是凸的、球形的，")
print("   所以只能用一条直线把两弯月牙各切一半，无法沿着弯曲的形状走。")
print("   DBSCAN 成功的原因：它按“密度可达”划分，只要点连成一片密度区域就算同簇，")
print("   能沿着月牙的弧形把每一弯完整地圈出来，不需要事先指定簇的数量。")

# ======================================================================
# 第二部分：真实银行数据的轮廓系数
# ======================================================================
print("\n" + "=" * 78)
print("实验二：真实客户数据（bank_customers.csv）的 KMeans 轮廓系数")
print("=" * 78)

df = pd.read_csv("bank_customers.csv")
feature_cols = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X = df[feature_cols].fillna(0)                 # 缺失值填 0
X_scaled = StandardScaler().fit_transform(X)   # 标准化

# 和前面 Step1 完全一致的设置：k=4，random_state=42
labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_scaled)

# 轮廓系数：对每个样本算“离本簇有多近、离最近的其他簇有多远”，取值 -1 ~ 1，越大越好
sil = silhouette_score(X_scaled, labels)
print(f"KMeans(k=4) 的轮廓系数（silhouette_score）：{sil:.4f}")
if sil > 0.5:
    print("  解读：大于 0.5，簇之间分得比较开，聚类结构较清晰。")
elif sil > 0.25:
    print("  解读：0.25~0.5 之间，有一定分群结构，但簇与簇之间存在重叠（客户数据常见）。")
else:
    print("  解读：小于 0.25，簇之间重叠较多，说明划分边界不够明显。")

print("\nStep5 运行成功")
