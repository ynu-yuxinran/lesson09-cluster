# 本代码基于手绘草图（sketch.jpg）迭代实现，为符合检查点要求，特征与k值按指南调整。
"""
Step3：客户分群画像 + 双拼看板图
运行环境：在本文件夹（bank_customers.csv 所在目录）下执行 python step3_dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========== 解决 matplotlib 中文显示成方块的问题 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 读取数据、选特征、标准化 ==========
df = pd.read_csv("bank_customers.csv")
feature_cols = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X = df[feature_cols].fillna(0)          # 缺失值填 0，避免报错

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)      # 标准化，消除量纲差异

# ========== 2. KMeans 聚类（k=4） ==========
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)   # 标签直接写回原表

# ========== 3. 计算四个簇的特征均值 ==========
# groupby('cluster') 按簇分组，.mean() 求每簇四列的平均值
cluster_means = df.groupby("cluster")[feature_cols].mean()
cluster_sizes = df.groupby("cluster").size()   # 每簇人数

print("各簇四特征均值（原始数值）：")
print(cluster_means.round(2))

# ========== 4. 根据均值给每个簇起中文业务名字 ==========
# 思路：把“整体平均水平”当作基准线，高于基准叫“高”，低于基准叫“低”
global_spend = df["avg_monthly_spend"].mean()   # 全体月均消费
global_product = df["product_count"].mean()     # 全体产品数

def name_cluster(row):
    """按“逾期是否严重、消费高不高、产品多不多”的顺序，像业务人员那样给客户群起名"""
    if row["overdue_count"] >= 2:
        return "高逾期风险型"            # 逾期次数明显偏高，风控重点
    if row["product_count"] >= global_product * 1.3:
        return "多产品稳健型"            # 持有产品数远超平均、风险低，粘性高
    if row["avg_monthly_spend"] >= global_spend:
        return "高消费活跃型"            # 花钱多、逾期少，营销重点
    if row["product_count"] >= global_product:
        return "多产品稳健型"
    return "低消费稳健型"                # 消费低但风险低，可挖掘潜力

cluster_names = {}
for c in cluster_means.index:
    cluster_names[c] = name_cluster(cluster_means.loc[c])

# ========== 5. 打印完整业务画像表 ==========
print("\n" + "=" * 78)
print(f"{'簇':^4}{'业务名称':^14}{'人数':^6}"
      f"{'余额':>12}{'月均消费':>12}{'产品数':>10}{'逾期次数':>10}")
print("-" * 78)
for c in cluster_means.index:
    m = cluster_means.loc[c]
    print(f"{c:^4}{cluster_names[c]:^14}{cluster_sizes[c]:^6}"
          f"{m['balance']:>12.2f}{m['avg_monthly_spend']:>12.2f}"
          f"{m['product_count']:>10.2f}{m['overdue_count']:>10.2f}")
print("=" * 78)

# ========== 6. 画 1 行 2 列的双拼看板 ==========
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ---------- 左图：散点图（余额 vs 月均消费） ----------
ax1 = axes[0]
colors = plt.cm.tab10(np.arange(4))          # 4 个簇各用一种颜色

for c in cluster_means.index:
    sub = df[df["cluster"] == c]             # 取出属于簇 c 的客户
    ax1.scatter(sub["balance"], sub["avg_monthly_spend"],
                s=45, alpha=0.75, color=colors[c],
                edgecolors="white", linewidths=0.5,
                label=f"簇{c}·{cluster_names[c]}（{cluster_sizes[c]}人）")

# 标注群心：标准化空间里的簇中心，需要 inverse_transform 还原成原始数值
centers_scaled = kmeans.cluster_centers_
centers_raw = scaler.inverse_transform(centers_scaled)
for c in cluster_means.index:
    x, y = centers_raw[c][0], centers_raw[c][1]   # 第 0 列余额，第 1 列月均消费
    ax1.scatter(x, y, marker="*", s=380, color="black",
                edgecolors="yellow", linewidths=1.2, zorder=5)
    # 在群心旁边写上所属簇的名字
    ax1.annotate(f"簇{c}", xy=(x, y), xytext=(8, 8),
                 textcoords="offset points", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85))

ax1.set_title("聚类特征对比：余额 vs 月均消费（★ 为群心）", fontsize=13)
ax1.set_xlabel("余额（balance）", fontsize=11)
ax1.set_ylabel("月均消费（avg_monthly_spend）", fontsize=11)
ax1.legend(fontsize=8, loc="upper right", framealpha=0.92)
ax1.grid(alpha=0.25)

# ---------- 右图：分组柱状图（四个簇的四项特征均值） ----------
ax2 = axes[1]
# 原始均值量纲差太大（余额 20 多万 vs 逾期 0.5），直接画柱状图会看不清
# 所以用同一个 scaler 把“均值”换算成标准化数值：0 = 全体平均，正数 = 高于平均
means_scaled = pd.DataFrame(
    scaler.transform(cluster_means),
    index=cluster_means.index,
    columns=feature_cols,
)

feature_labels = {
    "balance": "余额",
    "avg_monthly_spend": "月均消费",
    "product_count": "产品数",
    "overdue_count": "逾期次数",
}
bar_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

x = np.arange(len(cluster_means))          # 4 个簇的位置
width = 0.18                               # 每根柱子的宽度

for i, col in enumerate(feature_cols):
    bars = ax2.bar(x + (i - 1.5) * width,        # 4 根柱子并排，围绕簇中心分布
                   means_scaled[col],
                   width=width,
                   color=bar_colors[i],
                   label=feature_labels[col])
    # 在每根柱子顶部标注原始均值（余额按“万”、消费按“千”显示，便于阅读）
    for c_idx, c in enumerate(cluster_means.index):
        raw = cluster_means.loc[c, col]          # 该簇在该特征上的原始均值
        if col == "balance":
            txt = f"{raw / 10000:.1f}万"
        elif col == "avg_monthly_spend":
            txt = f"{raw / 1000:.1f}千"
        elif col == "product_count":
            txt = f"{raw:.1f}"
        else:
            txt = f"{raw:.2f}"
        # 柱子的坐标用标准化后的高度，文字写原始数值，二者不矛盾
        ax2.annotate(txt, (bars[c_idx].get_x() + bars[c_idx].get_width() / 2,
                           bars[c_idx].get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.5, rotation=90)

ax2.axhline(0, color="gray", linewidth=1, linestyle="--")   # 全体平均基准线
ax2.set_xticks(x)
ax2.set_xticklabels([f"簇{c}\n{cluster_names[c]}" for c in cluster_means.index], fontsize=9)
ax2.set_ylabel("标准化均值（0 = 全体平均）", fontsize=11)
ax2.set_title("各簇特征画像：谁高于/低于整体水平", fontsize=13)
ax2.legend(fontsize=9, loc="upper right")
ax2.grid(axis="y", alpha=0.25)
ax2.set_ylim(min(means_scaled.min()) - 0.6, max(means_scaled.max()) + 1.4)  # 给标注留空间

# ---------- 整体标题与版式 ----------
fig.suptitle(f"零售银行客户分群看板（KMeans 聚类，k=4，共 {len(df)} 位客户）",
             fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])      # 给总标题留出上边距

# ========== 7. 保存图片 ==========
plt.savefig("step3_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n已保存看板图：step3_dashboard.png")
print("Step3 运行成功")
