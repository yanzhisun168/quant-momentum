import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

etf_pool = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "512100": "中证1000ETF",
    "518880": "黄金ETF",
    "513100": "纳指ETF",
    "513500": "标普500ETF",
    "511010": "国债ETF",
}

def get_etf_data(symbol, start_date="20200101", end_date="20251231"):
    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    
    df = df.rename(columns={
        "日期": "date",
        "收盘": "close"
    })
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.sort_index()
    
    return df[["close"]]

price_data = pd.DataFrame()

for code, name in etf_pool.items():
    print(f"正在下载：{name}")
    df = get_etf_data(code)
    price_data[name] = df["close"]

price_data = price_data.ffill().dropna()

print(price_data.head())
print(price_data.tail())
print("所有ETF数据下载完成")
# 计算每日收益率
returns = price_data.pct_change().dropna()

print("每日收益率：")
print(returns.head())
# 设置动量周期
lookback = 60

# 计算过去60个交易日收益率，作为动量因子
momentum = price_data.pct_change(lookback)

# 向后移动一天，避免未来函数
momentum = momentum.shift(1)

print("动量因子：")
print(momentum.tail())
# 获取每个月最后一个交易日作为调仓日
monthly_rebalance_dates = price_data.resample("ME").last().index
monthly_rebalance_dates = price_data.index[price_data.index.isin(monthly_rebalance_dates)]

print("调仓日期：")
print(monthly_rebalance_dates[:10])

top_n = 1
# 创建持仓表
positions = pd.DataFrame(index=price_data.index, columns=price_data.columns, dtype=float)

for date in monthly_rebalance_dates:
    if date not in momentum.index:
        continue

    signal = momentum.loc[date].dropna()

    if len(signal) == 0:
        continue

    selected = signal.sort_values(ascending=False).head(top_n).index

    # 先把当天所有ETF仓位设为0
    positions.loc[date, :] = 0

    # 再给选中的ETF分配仓位
    positions.loc[date, selected] = 1 / top_n

# 非调仓日沿用上一次持仓
positions = positions.ffill().fillna(0)

print("持仓表：")
print(positions.tail())
print("每日仓位合计：")
print(positions.sum(axis=1).tail())

# 用昨天的持仓赚今天的钱，避免未来函数
strategy_returns = (positions.shift(1) * returns).sum(axis=1)
strategy_returns = strategy_returns.dropna()

print("策略每日收益：")
print(strategy_returns.head())
# 策略净值
strategy_nav = (1 + strategy_returns).cumprod()
strategy_nav.name = "动量轮动策略"

# 基准：沪深300ETF
benchmark_returns = returns["沪深300ETF"]
benchmark_nav = (1 + benchmark_returns).cumprod()
benchmark_nav.name = "沪深300ETF"

# 等权组合
equal_weight_returns = returns.mean(axis=1)
equal_weight_nav = (1 + equal_weight_returns).cumprod()
equal_weight_nav.name = "ETF等权组合"

# 合并净值
nav_df = pd.concat([strategy_nav, benchmark_nav, equal_weight_nav], axis=1).dropna()

print("净值数据：")
print(nav_df.tail())

plt.figure(figsize=(12, 6))

x = nav_df.index.to_numpy()

plt.plot(x, nav_df["动量轮动策略"].to_numpy(), label="Momentum Strategy")
plt.plot(x, nav_df["沪深300ETF"].to_numpy(), label="CSI 300 ETF")
plt.plot(x, nav_df["ETF等权组合"].to_numpy(), label="Equal Weight")

plt.title("Strategy NAV Comparison")
plt.xlabel("Date")
plt.ylabel("Net Asset Value")
plt.legend()
plt.grid(True)

plt.show()
# 计算最大回撤
def calculate_drawdown(nav):
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return drawdown

drawdown_df = nav_df.apply(calculate_drawdown)

print("最大回撤数据：")
print(drawdown_df.tail())

plt.figure(figsize=(12, 6))

x = drawdown_df.index.to_numpy()

plt.plot(x, drawdown_df["动量轮动策略"].to_numpy(), label="Momentum Strategy")
plt.plot(x, drawdown_df["沪深300ETF"].to_numpy(), label="CSI 300 ETF")
plt.plot(x, drawdown_df["ETF等权组合"].to_numpy(), label="Equal Weight")

plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)

plt.show()
# 计算绩效指标
def performance_metrics(returns, nav):
    annual_return = nav.iloc[-1] ** (252 / len(nav)) - 1
    annual_volatility = returns.std() * np.sqrt(252)

    if annual_volatility == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = annual_return / annual_volatility

    drawdown = calculate_drawdown(nav)
    max_drawdown = drawdown.min()

    if max_drawdown == 0:
        calmar_ratio = np.nan
    else:
        calmar_ratio = annual_return / abs(max_drawdown)

    win_rate = (returns > 0).mean()
    cumulative_return = nav.iloc[-1] - 1

    return {
        "年化收益率": annual_return,
        "年化波动率": annual_volatility,
        "夏普比率": sharpe_ratio,
        "最大回撤": max_drawdown,
        "Calmar比率": calmar_ratio,
        "日胜率": win_rate,
        "累计收益率": cumulative_return
    }
# 对齐收益率和净值日期
strategy_ret_aligned = strategy_returns.loc[nav_df.index]
benchmark_ret_aligned = benchmark_returns.loc[nav_df.index]
equal_weight_ret_aligned = equal_weight_returns.loc[nav_df.index]

strategy_metrics = performance_metrics(
    strategy_ret_aligned,
    nav_df["动量轮动策略"]
)

benchmark_metrics = performance_metrics(
    benchmark_ret_aligned,
    nav_df["沪深300ETF"]
)

equal_weight_metrics = performance_metrics(
    equal_weight_ret_aligned,
    nav_df["ETF等权组合"]
)

metrics_df = pd.DataFrame({
    "动量轮动策略": strategy_metrics,
    "沪深300ETF": benchmark_metrics,
    "ETF等权组合": equal_weight_metrics
})

print("绩效指标表：")
print(metrics_df)

formatted_metrics = metrics_df.copy()

percent_rows = ["年化收益率", "年化波动率", "最大回撤", "日胜率", "累计收益率"]

for row in percent_rows:
    formatted_metrics.loc[row] = formatted_metrics.loc[row].apply(lambda x: f"{x:.2%}")

for row in ["夏普比率", "Calmar比率"]:
    formatted_metrics.loc[row] = formatted_metrics.loc[row].apply(lambda x: f"{x:.2f}")

print("格式化后的绩效指标表：")
print(formatted_metrics)
# 第十步：参数敏感性分析

def run_momentum_backtest(price_data, lookback=60, top_n=1):
    returns = price_data.pct_change().dropna()

    momentum = price_data.pct_change(lookback).shift(1)

    rebalance_dates = price_data.resample("ME").last().index
    rebalance_dates = price_data.index[price_data.index.isin(rebalance_dates)]

    positions = pd.DataFrame(index=price_data.index, columns=price_data.columns, dtype=float)

    for date in rebalance_dates:
        if date not in momentum.index:
            continue

        signal = momentum.loc[date].dropna()

        if len(signal) == 0:
            continue

        selected = signal.sort_values(ascending=False).head(top_n).index

        positions.loc[date, :] = 0
        positions.loc[date, selected] = 1 / top_n

    positions = positions.ffill().fillna(0)

    strategy_returns = (positions.shift(1) * returns).sum(axis=1).dropna()
    strategy_nav = (1 + strategy_returns).cumprod()

    return strategy_returns, strategy_nav, positions
# 测试不同参数组合
param_results = []

for lookback in [20, 60, 120]:
    for top_n in [1, 2, 3]:
        ret, nav, pos = run_momentum_backtest(
            price_data,
            lookback=lookback,
            top_n=top_n
        )

        metrics = performance_metrics(ret, nav)
        metrics["动量周期"] = lookback
        metrics["持仓数量"] = top_n

        param_results.append(metrics)

param_df = pd.DataFrame(param_results)

param_df = param_df[
    ["动量周期", "持仓数量", "年化收益率", "年化波动率", "夏普比率", "最大回撤", "Calmar比率", "日胜率", "累计收益率"]
]

print("参数敏感性分析结果：")
print(param_df.sort_values(by="夏普比率", ascending=False))
#第十二步：保存结果

import os

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
plt.figure(figsize=(12, 6))

x = nav_df.index.to_numpy()

plt.plot(x, nav_df["动量轮动策略"].to_numpy(), label="Momentum Strategy")
plt.plot(x, nav_df["沪深300ETF"].to_numpy(), label="CSI 300 ETF")
plt.plot(x, nav_df["ETF等权组合"].to_numpy(), label="Equal Weight")

plt.title("Strategy NAV Comparison")
plt.xlabel("Date")
plt.ylabel("Net Asset Value")
plt.legend()
plt.grid(True)

plt.savefig("results/figures/nav_curve.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 6))

x = drawdown_df.index.to_numpy()

plt.plot(x, drawdown_df["动量轮动策略"].to_numpy(), label="Momentum Strategy")
plt.plot(x, drawdown_df["沪深300ETF"].to_numpy(), label="CSI 300 ETF")
plt.plot(x, drawdown_df["ETF等权组合"].to_numpy(), label="Equal Weight")

plt.title("Drawdown Comparison")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.legend()
plt.grid(True)

plt.savefig("results/figures/drawdown_curve.png", dpi=300, bbox_inches="tight")
plt.show()

formatted_metrics.to_csv("results/tables/performance_metrics.csv", encoding="utf-8-sig")
param_df.to_csv("results/tables/parameter_analysis.csv", index=False, encoding="utf-8-sig")