"""
Step1：客户分群 —— 读取数据、特征标准化、KMeans 聚类、统计每簇人数
运行环境：在本文件夹（bank_customers.csv 所在目录）下执行 python step1.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========== 1. 读取数据 ==========
# 用 pandas 读取当前文件夹下的 CSV 文件，得到一张二维表格 df（DataFrame）
df = pd.read_csv("bank_customers.csv")
print("原始数据形状（行数, 列数）：", df.shape)

# ========== 2. 选取四个特征列 ==========
# 只挑我们关心的四列：余额、月均消费、产品数、逾期次数
# 注意：先做 fillna(0) 把缺失值填成 0，否则后面的标准化/聚类会因为 NaN 报错
feature_cols = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]
X = df[feature_cols].fillna(0)  # X 是一个 200 行 4 列的特征矩阵

# ========== 3. 标准化处理 ==========
# KMeans 使用的是“距离”，而四个特征的量纲差别很大（余额是十万级，逾期次数只有 0~3）
# 所以必须先用 StandardScaler 把每列变成“均值 0、标准差 1”的分布，否则余额会主导聚类结果
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 返回的是 numpy 数组，形状仍是 200×4

# ========== 4. KMeans 聚类 ==========
# n_clusters=4 表示分成 4 个客户群；random_state=42 固定随机种子，保证每次运行结果一致
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)  # 得到每个客户所属的簇编号（0/1/2/3）

# ========== 5. 把簇标签写回原数据 ==========
# 新增一列 cluster，之后就可以按簇分析客户特征了
df["cluster"] = labels

# ========== 6. 统计每个簇的人数 ==========
# value_counts() 统计各簇出现次数；sort_index() 让输出按簇编号 0,1,2,3 排列更整齐
counts = df["cluster"].value_counts().sort_index()
print("\n每个簇的人数：")
for cluster_id, num in counts.items():
    print(f"簇 {cluster_id}: {num} 人")

# ========== 验证逻辑 ==========
# 四个簇人数相加应当等于总客户数 200
total = int(counts.sum())
print(f"\n各簇人数合计：{total}，总客户数：{len(df)}")
assert total == len(df), "各簇人数之和与总数据量不一致！"
assert len(df) == 200, "数据行数不是 200，请检查原始文件！"

# 查看前 5 行，确认 cluster 列已正确添加
print("\n前 5 行预览：")
print(df[["customer_id", "balance", "avg_monthly_spend", "product_count", "overdue_count", "cluster"]].head())

print("\nStep1 运行成功")
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取当前文件夹中的客户数据文件。
data = pd.read_csv("bank_customers.csv")

# 指定参与聚类的四个数值特征。
feature_columns = [
    "balance",
    "avg_monthly_spend",
    "product_count",
    "overdue_count",
]

# 先检查所需列是否都存在，避免列名写错时产生难以理解的错误。
missing_columns = [column for column in feature_columns if column not in data.columns]
if missing_columns:
    raise ValueError(f"CSV 文件缺少聚类所需的列：{missing_columns}")

# 将数据中的缺省值（NaN）填为 0，避免标准化和聚类时因缺失值报错。
data = data.fillna(0)

# 从原数据中取出聚类特征。
features = data[feature_columns]

# 标准化特征：把不同量纲的数据缩放到相近范围，
# 避免金额等数值较大的列对聚类结果产生不成比例的影响。
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 创建并训练 KMeans 模型，将客户分成 4 个簇。
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_features)

# 将每位客户对应的簇标签添加为原数据的新列。
data["cluster"] = cluster_labels

# 显示簇 0、1、2、3 各自包含的客户人数；reindex 确保即使某簇人数为 0 也会显示。
cluster_counts = data["cluster"].value_counts().reindex(range(4), fill_value=0)
print("每个簇的客户人数：")
for cluster_id, customer_count in cluster_counts.items():
    print(f"簇 {cluster_id}：{customer_count} 人")

# 检查所有簇的人数加总与数据总人数一致；本作业数据应为 200 人。
assert int(cluster_counts.sum()) == len(data), "各簇人数之和与数据总人数不一致。"
assert len(data) == 200, f"预期有 200 位客户，实际读取到 {len(data)} 位。"

print("Step1 运行成功")
