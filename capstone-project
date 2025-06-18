# 필요한 라이브러리 설치
!pip install numpy pandas yfinance ta bayesian-optimization xgboost shap joblib tqdm matplotlib koreanize-matplotlib -q

import numpy as np
import pandas as pd
import yfinance as yf
from ta.trend import MACD
from ta.momentum import StochasticOscillator, ROCIndicator
from ta.volatility import AverageTrueRange
from bayes_opt import BayesianOptimization
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings
import joblib
import koreanize_matplotlib

warnings.filterwarnings('ignore')

# 1. 데이터 수집
def fetch_data(tickers, start_date, end_date):
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    df = df.interpolate(method='linear').ffill().bfill()
    df.columns = [f"{c[1]}_{c[0]}" for c in df.columns]
    return df

# 2. 피처 생성 (결측 보정 포함)
def make_features(df, params):
    f = pd.DataFrame(index=df.index)
    f['macd'] = MACD(df['SPY_Close'], params['macd_fast'], params['macd_slow'], params['macd_sign']).macd_diff()
    f['stoch'] = StochasticOscillator(df['SPY_High'], df['SPY_Low'], df['SPY_Close'], params['stoch_window']).stoch()
    f['roc'] = ROCIndicator(df['SPY_Close'], params['roc_window']).roc()
    f['vix_roc'] = ROCIndicator(df['^VIX_Close'], params['vix_roc_window']).roc()
    f['btc_roc'] = ROCIndicator(df['BTC-USD_Close'], params['btc_roc_window']).roc()
    f['tnx_roc'] = ROCIndicator(df['^TNX_Close'], params['tnx_roc_window']).roc()
    f['spy_sma200'] = df['SPY_Close'].rolling(window=200).mean()
    f['spy_atr'] = AverageTrueRange(df['SPY_High'], df['SPY_Low'], df['SPY_Close'], 14).average_true_range()
    f['vix_std'] = df['^VIX_Close'].rolling(window=10).std()
    return f.bfill().ffill().fillna(0)

# 3. 백테스트
def backtest(df, params_df, feature_df, freq='D'):
    annual_factor = 52 if freq == 'W' else 252
    rets = df[[c for c in ['SPY_Close','SSO_Close','SH_Close'] if c in df.columns]].pct_change().fillna(0)
    equity = [1.0]; prev_weights = None
    for i in range(1, len(df)):
        dt = df.index[i]
        p = params_df.loc[params_df.index <= dt].iloc[-1]
        x = feature_df.loc[dt, feature_names].values.reshape(1,-1)
        if 'model' in p and p['model'] is not None:
            signal = int(p['model'].predict(x)[0]) - 1
        else:
            votes = np.sign([feature_df.at[dt, 'macd'], feature_df.at[dt, 'roc'], (1 if feature_df.at[dt, 'stoch'] < 20 else -1 if feature_df.at[dt, 'stoch'] > 80 else 0)])
            signal = int(np.sign(votes.sum()))

        is_high_fear = df.at[dt, '^VIX_Close'] > 35
        if is_high_fear: signal = -1
        is_downtrend = df.at[dt, 'SPY_Close'] < feature_df.at[dt, 'spy_sma200']
        if signal == 1 and is_downtrend: signal = 0

        weights = {'SSO': 0.0, 'SH': 0.0, 'SPY': 0.0, 'CASH': 0.0}
        if signal == 1: weights['SSO'] = p['w_sso']; weights['CASH'] = 1 - p['w_sso']
        elif signal == -1: weights['SH'] = p['w_sh']; weights['CASH'] = 1 - p['w_sh']
        else: weights['SPY'] = p['w_spy']; weights['CASH'] = 1 - p['w_spy']

        daily_ret = sum(weights[t] * rets.iloc[i][f'{t}_Close'] for t in ['SSO','SH','SPY'])
        if prev_weights is not None:
            daily_ret -= 0.001 * sum(abs(weights[t] - prev_weights[t]) for t in ['SSO','SH','SPY'])
        equity.append(equity[-1] * (1 + daily_ret))
        prev_weights = weights.copy()

    eq_series = pd.Series(equity, index=df.index)
    returns = eq_series.pct_change().fillna(0)
    sharpe = returns.mean()/returns.std() * np.sqrt(annual_factor) if returns.std() != 0 else 0
    mdd = (1 - eq_series/eq_series.cummax()).max()
    wins = returns[returns > 0]; losses = returns[returns < 0]
    pf = wins.sum()/abs(losses.sum()) if losses.sum() != 0 else 100
    wr = len(wins)/len(returns) if len(returns) > 0 else 0
    return {'sharpe': sharpe, 'mdd': mdd, 'profit_factor': pf, 'win_rate': wr, 'returns': returns}

# 4. 목적함수
def create_objective(df, feature_df, model):
    def objective(w_spy, w_sso, w_sh):
        total = w_spy + w_sso + w_sh
        if total == 0: return -1e9
        w_spy /= total; w_sso /= total; w_sh /= total
        params = pd.DataFrame([{'w_spy': w_spy, 'w_sso': w_sso, 'w_sh': w_sh, 'model': model}], index=[df.index[0]])
        res = backtest(df, params, feature_df, freq='W')
        return res['sharpe'] / (1 + res['mdd']) if res['mdd'] >= 0.01 else res['sharpe']
    return objective

# 5. Walk-Forward 실행
ml_usage_record = {}
def run_single_period(args):
    dt, df_all, feature_df, output_dir, use_xgb, feature_names, feature_configs = args
    train_start = dt - pd.DateOffset(years=3)
    train_end = dt - pd.DateOffset(days=1)
    df_train = df_all.loc[train_start:train_end]
    test_start = dt
    test_end = dt + pd.DateOffset(months=1) - pd.DateOffset(days=1)
    df_test = df_all.loc[test_start:test_end]

    if df_train.empty or df_test.empty:
        print(f"🚫 데이터 없음 → {dt.strftime('%Y-%m')}")
        return None, None

    df_train_weekly = df_train.resample('W-FRI').last()
    feat_train_weekly = make_features(df_train_weekly, feature_configs)

    model = None
    if use_xgb and len(df_train) > 50:
        X = df_train.join(feature_df, how='inner')[feature_names]
        y = np.sign(df_train['SPY_Close'].pct_change().shift(-1)).map({-1:0, 0:1, 1:2}).fillna(1)
        model = XGBClassifier(n_estimators=50, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        model.fit(X, y)
        ml_usage_record[dt.strftime('%Y-%m')] = 1
    else:
        ml_usage_record[dt.strftime('%Y-%m')] = 0

    final_feat_test = feature_df.loc[df_test.index]
    if final_feat_test.isnull().all().any():
        print(f"⚠️ 피처 결측 → {dt.strftime('%Y-%m')} 스킵")
        return None, None

    optimizer = BayesianOptimization(
        f=create_objective(df_train_weekly, feat_train_weekly, model),
        pbounds={'w_spy': (0.01, 1), 'w_sso': (0.01, 1), 'w_sh': (0.01, 1)},
        random_state=42, verbose=0)
    optimizer.maximize(init_points=10, n_iter=25)
    best_params = optimizer.max['params']

    final_params = pd.DataFrame([{'model': model, **best_params}], index=[dt])
    res = backtest(df_test, final_params, final_feat_test, freq='D')
    res['returns'].name = dt.strftime('%Y-%m')
    params_to_save = {'date': dt, **best_params}
    return res['returns'], params_to_save

def run_walk_forward(df_all, feature_df, test_dates, output_dir, use_xgb, feature_names, feature_configs):
    tasks = [(dt, df_all, feature_df, output_dir, use_xgb, feature_names, feature_configs) for dt in test_dates]
    all_returns, all_ports = [], []
    for task in tqdm(tasks, desc="Walk-Forward Optimization"):
        res_returns, res_port = run_single_period(task)
        if res_returns is not None:
            all_returns.append(res_returns)
            all_ports.append(res_port)
    return all_returns, all_ports

# 6. 시각화
def plot_ml_usage(record):
    df = pd.Series(record)
    plt.figure(figsize=(15, 3))
    df.plot(kind='bar', color=['green' if v == 1 else 'gray' for v in df])
    plt.title('월별 머신러닝 사용 여부 (초록: 사용함, 회색: 사용 안함)')
    plt.xticks(rotation=45); plt.tight_layout(); plt.show()

def plot_portfolio_ratios(portfolio_params):
    if not portfolio_params: return
    df_rat = pd.DataFrame(portfolio_params).set_index('date')
    total = df_rat[['w_spy', 'w_sso', 'w_sh']].sum(axis=1)
    df_rat_normalized = df_rat[['w_spy', 'w_sso', 'w_sh']].div(total, axis=0)
    plt.figure(figsize=(15, 6))
    plt.stackplot(df_rat_normalized.index, df_rat_normalized.T, labels=df_rat_normalized.columns, alpha=0.8)
    plt.title('월별 포트폴리오 비율 변화')
    plt.legend(); plt.tight_layout(); plt.show()

def plot_monthly_return_color(equity_curve):
    monthly_eq = equity_curve.resample('M').last()
    monthly_ret = monthly_eq.pct_change().fillna(0)
    colors = ['green' if x >= 0 else 'red' for x in monthly_ret]
    plt.figure(figsize=(15, 4))
    monthly_ret.plot(kind='bar', color=colors)
    plt.title('월별 수익률')
    plt.tight_layout(); plt.show()

# 7. 메인 실행
if __name__ == '__main__':
    tickers = ['SPY', 'SSO', 'SH', 'BTC-USD', '^VIX', '^TNX']
    start_date = '2014-01-01'  # ✅ 1년 이상 데이터 확보
    end_date = pd.to_datetime('today').strftime('%Y-%m-%d')
    test_start_date = '2020-01-01'

    feature_names = ['macd', 'stoch', 'roc', 'vix_roc', 'btc_roc', 'tnx_roc', 'spy_atr', 'vix_std']
    feature_params_config = {
        'macd_fast': 12, 'macd_slow': 26, 'macd_sign': 9, 'stoch_window': 14,
        'roc_window': 12, 'vix_roc_window': 5, 'btc_roc_window': 5, 'tnx_roc_window': 5,
    }

    output_dir = 'walk_forward_results'
    os.makedirs(output_dir, exist_ok=True)

    df = fetch_data(tickers, start_date, end_date)
    feature_df = make_features(df, feature_params_config)
    test_dates = pd.date_range(test_start_date, end_date, freq='MS')

    returns, ports = run_walk_forward(df, feature_df, test_dates, output_dir, True, feature_names, feature_params_config)

    if returns:
        strategy_returns = pd.concat(returns).sort_index()
        strategy_equity = (1 + strategy_returns).cumprod()
        spy_returns = df['SPY_Close'].pct_change().fillna(0)
        spy_equity = (1 + spy_returns.loc[strategy_returns.index]).cumprod()

        cagr = (strategy_equity.iloc[-1]**(252/len(strategy_equity)) - 1) * 100
        sharpe = strategy_returns.mean()/strategy_returns.std() * np.sqrt(252)
        mdd = (1 - strategy_equity / strategy_equity.cummax()).max() * 100
        print(f"누적 수익률: {(strategy_equity.iloc[-1]-1)*100:.2f}%")
        print(f"CAGR: {cagr:.2f}%  | Sharpe: {sharpe:.2f} | MDD: {mdd:.2f}%")

        plot_monthly_return_color(strategy_equity)
        plot_portfolio_ratios(ports)
        plot_ml_usage(ml_usage_record)

        plt.figure(figsize=(15, 6))
        plt.plot(strategy_equity, label='전략', linewidth=2)
        plt.plot(spy_equity, label='SPY Buy & Hold', linestyle='--')
        plt.legend(); plt.title('전략 vs. SPY 누적수익'); plt.grid(True); plt.tight_layout(); plt.show()
    else:
        print("백테스트 결과 없음")
