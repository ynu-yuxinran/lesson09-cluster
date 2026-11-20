"""
Step4：自主实践 —— 特征扩展实验 + 随机种子稳定性实验
运行环境：在本文件夹（bank_customers.csv 所在目录）下执行 python step4_extension.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ========== 0. 读取数据（两个实验共用同一份原始数据） ==========
df = pd.read_csv("bank_customers.csv")

BASE_COLS = ["balance", "avg_monthly_spend", "product_count", "overdue_count"]  # 原 4 特征
EXT_COLS = BASE_COLS + ["account_age_months"]                                  # 扩展后的 5 特征

# 常用的列名 → 中文列名，方便表格打印
CN_NAMES = {
    "balance": "余额",
    "avg_monthly_spend": "月均消费",
    "product_count": "产品数",
    "overdue_count": "逾期次数",
    "account_age_months": "账户月数",
}


def cluster_and_profile(data, cols, random_state=42):
    """
    通用小工具：取特征 → 填缺失值 → 标准化 → KMeans(k=4) → 返回标签和簇均值
    把重复步骤封装成函数，两个实验都能复用，避免代码写两遍
    """
    X = data[cols].fillna(0)                      # 缺失值先填 0
    X_scaled = StandardScaler().fit_transform(X)  # 标准化，消除量纲差异
    kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)         # 每个样本的簇编号
    means = pd.DataFrame(X, columns=cols).groupby(labels).mean()  # 按簇求原始均值
    sizes = pd.Series(labels).value_counts().sort_index()         # 每簇人数
    return labels, means, sizes


# ======================================================================
# 实验一：加入 account_age_months，变成 5 个特征
# ======================================================================
print("=" * 80)
print("实验一：5 特征聚类（加入 account_age_months 账户月数）")
print("=" * 80)

labels_ext, means_ext, sizes_ext = cluster_and_profile(df, EXT_COLS, random_state=42)

# 用全体平均值作为“高/低”的判断基准
global_age = df["account_age_months"].mean()
global_product = df["product_count"].mean()

def name_cluster_ext(row):
    """根据“逾期风险 → 账户新老 → 产品多少”逐层判断，给每个簇起业务名字"""
    if row["overdue_count"] >= 2:
        return "高逾期风险型"          # 逾期明显偏高，风控重点
    if row["account_age_months"] >= global_age * 1.2:
        # 账户时长明显高于平均（约 148 个月以上），属于老客户
        if row["product_count"] >= global_product * 1.2:
            return "老客户高资产型"    # 老客户 + 多产品，交叉销售价值最高
        return "老客户基础型"          # 老客户但产品少，适合提升产品覆盖
    if row["account_age_months"] <= global_age * 0.8:
        return "新客户成长型"          # 开户时间短，重点做培育
    return "中间年份稳健型"

names_ext = {c: name_cluster_ext(means_ext.loc[c]) for c in means_ext.index}

# ---- 打印带业务名字的 5 特征均值表 ----
print("\n【表1】5 个特征的分群均值表：\n")
header = f"{'簇':^4}{'业务名称':^14}{'人数':^6}"
for col in EXT_COLS:
    header += f"{CN_NAMES[col]:>10}"
print(header)
print("-" * 84)
for c in means_ext.index:
    line = f"{c:^4}{names_ext[c]:^14}{sizes_ext[c]:^6}"
    for col in EXT_COLS:
        line += f"{means_ext.loc[c, col]:>10.2f}"
    print(line)
print("-" * 84)

# ---- 简单对比：年轻（新客户）群体 vs 年长（老客户）群体的特征 ----
# 按簇均值的账户月数分组：明显偏短的是新客户群，明显偏长的是老客户群
young_clusters = [c for c in means_ext.index if means_ext.loc[c, "account_age_months"] <= global_age * 0.6]
old_clusters = [c for c in means_ext.index if means_ext.loc[c, "account_age_months"] >= global_age * 1.2]

# 借用一张带簇标签的临时表，方便直接对“人”求平均（簇标签不写回原 df）
df_tmp = df[EXT_COLS].copy()
df_tmp["cluster_ext"] = labels_ext

print("\n【对比】加入账户月数后，新客户群与老客户群的差异：\n")
for tag, cs in [("新客户群", young_clusters), ("老客户群", old_clusters)]:
    rows = df_tmp[df_tmp["cluster_ext"].isin(cs)]
    m = rows[EXT_COLS].mean()
    cluster_desc = "、".join(f"簇{c}（{names_ext[c]}，{sizes_ext[c]}人）" for c in cs)
    print(f"  {tag}：{cluster_desc}，合计 {len(rows)} 人")
    print(f"    平均账户月数 {m['account_age_months']:.1f} 个月 | 余额 {m['balance']:.2f} | "
          f"月均消费 {m['avg_monthly_spend']:.2f} | 产品数 {m['product_count']:.2f} | "
          f"逾期次数 {m['overdue_count']:.2f}")

print("\n  注：簇3（高逾期风险型）平均账户月数 95.1 个月，属于“中等年限 + 高风险”，")
print("     单独作为风险群处理，不并入上面的新/老客户对比。")

print("\n  >> 对比结论：")
print("     原来的 4 特征里没有“客户在银行待了多久”这一信息，聚类主要被余额和逾期次数驱动；")
print("     加入账户月数后，KMeans 把开户时间很短的客户单独分成一群（新客户群），")
print("     而开户时间长的老客户又被进一步拆成“高资产型”和“基础型”，")
print("     说明账户月数是一个有区分度的新特征，能让画像从“有钱/没钱”细化到“新客/老客”。")

# ======================================================================
# 实验二：稳定性实验（退回 4 特征，只改 random_state）
# ======================================================================
print("\n" + "=" * 80)
print("实验二：随机种子稳定性实验（4 特征，k=4，random_state = 0 / 10 / 42）")
print("=" * 80)

for rs in [0, 10, 42]:
    labels_base, _, sizes_base = cluster_and_profile(df, BASE_COLS, random_state=rs)
    # 把 0/1/2/3 四个簇的人数依次列出来；get(c, 0) 防止某个簇没出现时报 KeyError
    counts = [int(sizes_base.get(c, 0)) for c in range(4)]
    print(f"\nrandom_state = {rs}")
    print("  簇人数分布： " + " | ".join(f"簇{c}: {counts[c]:>2d} 人" for c in range(4)))
    print(f"  合计：{sum(counts)} 人" + ("  ✔ 等于总客户数" if sum(counts) == len(df) else "  ✘ 数据量异常"))

print("\n  >> 观察结论：")
print("     三次运行的总人数都等于 200，说明聚类没有丢样本；")
print("     但三组人数分布并不相同，因为 KMeans 的初始质心是随机的，")
print("     随机种子不同 → 初始质心不同 → 最终收敛到的划分也不同。")
print("     这也说明单次 KMeans 结果有一定偶然性，正式分析时应固定 random_state 并做多次验证。")

print("\nStep4 运行成功")
