# best so far
# -*- coding: utf-8 -*-
!pip -q install lightgbm optuna koreanize-matplotlib tqdm -U

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from datetime import datetime
import koreanize_matplotlib

# 데이터 로딩
print("데이터 로딩 시작...")
df = pd.read_csv('/content/kpopradar_master_followers_wide.csv')
album_release_df = pd.read_csv('/content/240105_artist_album_release.csv')
youtube_daily_df = pd.read_csv('/content/KAIST_kpopradar_youtube_artists_daily_20230930.csv')
print("데이터 로딩 완료")

# 날짜 형식 변환
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
album_release_df['publish_date'] = pd.to_datetime(album_release_df['publish_date'], format='%Y%m%d')
youtube_daily_df['date'] = pd.to_datetime(youtube_daily_df['date'], format='%Y%m%d')

# 데이터 전처리
print("데이터 전처리 시작...")
yt_missing_ratio = df.groupby('artist_id')['youtube_followers'].apply(lambda x: x.isnull().mean())
valid_artist_ids = yt_missing_ratio[yt_missing_ratio <= 0.2].index
df_filtered = df[df['artist_id'].isin(valid_artist_ids)]
obs_counts = df_filtered.groupby('artist_id')['date'].nunique()
valid_artist_ids_long = obs_counts[obs_counts >= 365].index
df_filtered_long = df_filtered[df_filtered['artist_id'].isin(valid_artist_ids_long)].copy()
df_model = df_filtered_long.copy()
df_model.sort_values(by=['artist_id', 'date'], inplace=True)

print("데이터 전처리 완료")

# YouTube 팔로워 특징 생성
print("YouTube 특징 생성 시작...")
df_model['yt_diff'] = df_model.groupby('artist_id')['youtube_followers'].diff()
df_model['yt_pct_change'] = df_model.groupby('artist_id')['youtube_followers'].pct_change(fill_method=None)
df_model['yt_rolling_mean_7'] = df_model.groupby('artist_id')['youtube_followers'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
df_model['yt_rolling_mean_30'] = df_model.groupby('artist_id')['youtube_followers'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
df_model['yt_daily_change'] = df_model['yt_diff']

# Z-score 계산
epsilon = 1e-8
df_model['yt_daily_change_rolling_mean_365'] = df_model.groupby('artist_id')['yt_daily_change'].transform(lambda x: x.rolling(window=365, min_periods=1).mean())
df_model['yt_daily_change_rolling_std_365'] = df_model.groupby('artist_id')['yt_daily_change'].transform(lambda x: x.rolling(window=365, min_periods=1).std())
df_model['yt_daily_change_zscore'] = (df_model['yt_daily_change'] - df_model['yt_daily_change_rolling_mean_365']) / (df_model['yt_daily_change_rolling_std_365'] + epsilon)

def calculate_rsi(series, window=14):
    diff = series.diff()
    gain = diff.mask(diff < 0, 0)
    loss = diff.mask(diff > 0, 0).abs()
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / (avg_loss + epsilon)
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_model['yt_rsi_14'] = df_model.groupby('artist_id')['youtube_followers'].transform(lambda x: calculate_rsi(x))
df_model['day_of_week'] = df_model['date'].dt.dayofweek
df_model['month'] = df_model['date'].dt.month
df_model['yt_ema_12'] = df_model.groupby('artist_id')['youtube_followers'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df_model['yt_ema_26'] = df_model.groupby('artist_id')['youtube_followers'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df_model['yt_macd'] = df_model['yt_ema_12'] - df_model['yt_ema_26']
df_model['yt_signal_line'] = df_model.groupby('artist_id')['yt_macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())
print("YouTube 특징 생성 완료")

# 다른 플랫폼 특징 생성
print("다른 플랫폼 특징 생성 시작...")
social_media_cols = ['instagram_followers', 'twitter_followers', 'tiktok_followers', 'spotify_followers', 'fancafe_followers']
for col in social_media_cols:
    prefix = col.split("_")[0]
    df_model[f'{prefix}_diff'] = df_model.groupby('artist_id')[col].diff()
    df_model[f'{prefix}_pct_change'] = df_model.groupby('artist_id')[col].pct_change(fill_method=None)
    df_model[f'{prefix}_rolling_mean_7'] = df_model.groupby('artist_id')[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df_model[f'{prefix}_rolling_mean_30'] = df_model.groupby('artist_id')[col].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df_model[f'{prefix}_rolling_std_7'] = df_model.groupby('artist_id')[col].transform(lambda x: x.rolling(window=7, min_periods=1).std())
    df_model[f'{prefix}_rolling_std_30'] = df_model.groupby('artist_id')[col].transform(lambda x: x.rolling(window=30, min_periods=1).std())
    for lag in [1, 7, 30]:
        df_model[f'{prefix}_lag_{lag}d'] = df_model.groupby('artist_id')[col].shift(lag)

platforms = ['youtube', 'instagram', 'twitter', 'tiktok', 'spotify', 'fancafe']
for platform in platforms:
    diff_col = 'yt_diff' if platform == 'youtube' else f'{platform}_diff'
    df_model[f'{platform}_growing'] = (df_model[diff_col] > 0).astype(int)
df_model['num_platforms_growing'] = df_model[[f'{platform}_growing' for platform in platforms]].sum(axis=1)
print("다른 플랫폼 특징 생성 완료")

# 앨범 발매 데이터 병합
print("앨범 발매 데이터 병합 시작...")
album_releases_for_merge = album_release_df[['artist_id', 'publish_date']].copy()
album_releases_for_merge['artist_id'] = pd.to_numeric(album_releases_for_merge['artist_id'], errors='coerce').astype('Int64')
album_releases_for_merge = album_releases_for_merge.dropna(subset=['artist_id'])
album_releases_for_merge.rename(columns={'publish_date': 'date'}, inplace=True)
album_releases_for_merge['album_released_on_date'] = 1
df_model_merged = pd.merge(df_model, album_releases_for_merge, on=['artist_id', 'date'], how='left')
df_model_merged['album_released_on_date'].fillna(0, inplace=True)

df_model_merged.sort_values(by=['artist_id', 'date'], inplace=True)
df_model_merged['last_album_date'] = df_model_merged.groupby('artist_id')['date'].transform(lambda x: x.where(df_model_merged['album_released_on_date'] == 1).ffill())
df_model_merged['days_since_last_album'] = (df_model_merged['date'] - df_model_merged['last_album_date']).dt.days
df_model_merged['days_since_last_album'].fillna(df_model_merged['days_since_last_album'].max() + 1, inplace=True)
df_model_merged.drop('last_album_date', axis=1, inplace=True)
df_model_merged['rolling_album_freq_365'] = df_model_merged.groupby('artist_id')['album_released_on_date'].transform(lambda x: x.rolling(window=365, min_periods=1).sum())
print("앨범 발매 데이터 병합 완료")

# YouTube 조회수 데이터 병합
print("YouTube 조회수 데이터 병합 시작...")
youtube_daily_df = youtube_daily_df.rename(columns={'id': 'artist_id'})
youtube_daily_df.sort_values(by=['artist_id', 'date'], inplace=True)
youtube_daily_df['youtube_viewcount_rolling_std_7'] = youtube_daily_df.groupby('artist_id')['viewcount'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
youtube_daily_df['youtube_viewcount_rolling_std_30'] = youtube_daily_df.groupby('artist_id')['viewcount'].transform(lambda x: x.rolling(window=30, min_periods=1).std())
youtube_volatility_features = youtube_daily_df[['artist_id', 'date', 'youtube_viewcount_rolling_std_7', 'youtube_viewcount_rolling_std_30']].copy()
df_model_merged = pd.merge(df_model_merged, youtube_volatility_features, on=['artist_id', 'date'], how='left')
df_model_merged['youtube_viewcount_rolling_std_7'].fillna(0, inplace=True)
df_model_merged['youtube_viewcount_rolling_std_30'].fillna(0, inplace=True)
print("YouTube 조회수 데이터 병합 완료")

# 상호작용 및 새로운 피쳐 생성
print("상호작용 및 새로운 피쳐 생성 시작...")
df_model_merged['yt_inst_change_ratio'] = df_model_merged['yt_daily_change'] / (df_model_merged['instagram_diff'] + epsilon)
df_model_merged['yt_inst_follower_ratio'] = df_model_merged['youtube_followers'] / (df_model_merged['instagram_followers'] + epsilon)
df_model_merged['yt_daily_change_sq'] = df_model_merged['yt_daily_change']**2

# 새로운 피쳐: 다른 플랫폼 비율
for platform in ['twitter', 'tiktok', 'spotify', 'fancafe']:
    df_model_merged[f'yt_{platform}_follower_ratio'] = df_model_merged['youtube_followers'] / (df_model_merged[f'{platform}_followers'] + epsilon)

# 새로운 피쳐: 앨범 발매 관련
df_model_merged['days_since_last_album_sq'] = df_model_merged['days_since_last_album']**2
df_model_merged['album_freq_rolling_180'] = df_model_merged.groupby('artist_id')['album_released_on_date'].transform(lambda x: x.rolling(window=180, min_periods=1).sum())
print("상호작용 및 새로운 피쳐 생성 완료")

# 결측치 처리
print("결측치 처리 시작...")
cols_to_fill_zero = [col for col in df_model_merged.columns if 'rolling_' in col or 'lag_' in col or 'ratio' in col or 'sq' in col]
for col in cols_to_fill_zero:
    df_model_merged[col].fillna(0, inplace=True)
print("결측치 처리 완료")

# 특징 리스트 정의
features = [col for col in df_model_merged.columns if col not in ['artist_id', 'artist_name', 'date', 'youtube_followers_future', 'youtube_growth_pct', 'is_rapid_growth']]

# 모델 하이퍼파라미터 (Optuna 생략, 기본값 사용)
print("기본 하이퍼파라미터 사용")
best_params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01,
}

# 모델 평가 함수
def evaluate_model_with_params(df, growth_period, percentile_threshold, features, best_params):
    print(f"성장 기간 {growth_period}일, 백분위 임계값 {percentile_threshold} 평가 시작...")
    df_eval = df.copy()

    df_eval['youtube_followers_future'] = df_eval.groupby('artist_id')['youtube_followers'].shift(-growth_period)
    df_eval['youtube_growth_pct'] = ((df_eval['youtube_followers_future'] - df_eval['youtube_followers']) / (df_eval['youtube_followers'] + 1)) * 100
    df_eval = df_eval.dropna(subset=['youtube_followers_future', 'youtube_growth_pct']).copy()

    latest_date = df_eval['date'].max()
    split_date = latest_date - pd.DateOffset(years=1)
    validation_split_date = latest_date - pd.DateOffset(years=1, months=6)
    df_train = df_eval[df_eval['date'] < validation_split_date].copy()
    df_val = df_eval[(df_eval['date'] >= validation_split_date) & (df_eval['date'] < split_date)].copy()
    df_test = df_eval[df_eval['date'] >= split_date].copy()

    growth_threshold = df_train['youtube_growth_pct'].quantile(percentile_threshold)
    print(f"성장 임계값: {growth_threshold:.2f}%")
    df_train['is_rapid_growth'] = (df_train['youtube_growth_pct'] >= growth_threshold).astype(int)
    df_val['is_rapid_growth'] = (df_val['youtube_growth_pct'] >= growth_threshold).astype(int)
    df_test['is_rapid_growth'] = (df_test['youtube_growth_pct'] >= growth_threshold).astype(int)

    X_train = df_train[features]
    y_train = df_train['is_rapid_growth']
    X_val = df_val[features]
    y_val = df_val['is_rapid_growth']
    X_test = df_test[features]
    y_test = df_test['is_rapid_growth']

    train_class_counts = y_train.value_counts()
    scale_pos_weight = train_class_counts[0] / train_class_counts[1] if 1 in train_class_counts else 1.0
    lgbm_selector = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, **best_params)
    lgbm_selector.fit(X_train, y_train)
    feature_importance = pd.Series(lgbm_selector.feature_importances_, index=X_train.columns)
    selected_features = feature_importance[feature_importance > feature_importance.median()].index
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    lgbm = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, **best_params)
    lgbm.fit(X_train_selected, y_train)

    y_proba_val = lgbm.predict_proba(X_val_selected)[:, 1]
    precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, y_proba_val)
    beta = 0.5
    fbeta_scores_val = ((1 + beta**2) * precision_val * recall_val) / ((beta**2 * precision_val) + recall_val + 1e-8)
    optimal_threshold = thresholds_val[np.argmax(fbeta_scores_val[:len(thresholds_val)])] if len(thresholds_val) > 0 else 0.5

    y_proba_test = lgbm.predict_proba(X_test_selected)[:, 1]
    y_pred_optimal = (y_proba_test >= optimal_threshold).astype(int)
    f1_optimal = f1_score(y_test, y_pred_optimal)

    print(f"F1-Score: {f1_optimal:.4f}")
    return f1_optimal, optimal_threshold, X_test_selected, y_test, df_test

# 그리드 서치
print("그리드 서치 시작...")
growth_periods = [60, 120, 180]
percentile_thresholds = [0.80, 0.90, 0.95]
grid_results = []

for gp in growth_periods:
    for pt in percentile_thresholds:
        f1_score_val, opt_threshold, X_test_selected, y_test, df_test = evaluate_model_with_params(df_model_merged, gp, pt, features, best_params)
        grid_results.append({'growth_period': gp, 'percentile_threshold': pt, 'f1_score': f1_score_val, 'optimal_threshold': opt_threshold})

best_result = max(grid_results, key=lambda x: x['f1_score'])
best_growth_period = best_result['growth_period']
best_percentile_threshold = best_result['percentile_threshold']
best_f1 = best_result['f1_score']
best_optimal_threshold = best_result['optimal_threshold']

print(f"최적 F1-Score: {best_f1:.4f}, 성장 기간: {best_growth_period}일, 백분위 임계값: {best_percentile_threshold}")

# 그리드 서치 결과 시각화
print("그리드 서치 결과 시각화...")
results_df = pd.DataFrame(grid_results)
pivot_table = results_df.pivot(index='percentile_threshold', columns='growth_period', values='f1_score')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title('성장 기간 및 백분위 임계값에 따른 F1-Score 히트맵')
plt.xlabel('성장 기간 (일)')
plt.ylabel('백분위 임계값 (긍정 성장 기준)')
plt.yticks(rotation=0)
plt.show()

# 최종 모델 학습
print("최종 모델 학습 시작...")
df_final = df_model_merged.copy()
df_final['youtube_followers_future'] = df_final.groupby('artist_id')['youtube_followers'].shift(-best_growth_period)
df_final['youtube_growth_pct'] = ((df_final['youtube_followers_future'] - df_final['youtube_followers']) / (df_final['youtube_followers'] + 1)) * 100
df_final = df_final.dropna(subset=['youtube_followers_future', 'youtube_growth_pct']).copy()

latest_date = df_final['date'].max()
split_date = latest_date - pd.DateOffset(years=1)
validation_split_date = latest_date - pd.DateOffset(years=1, months=6)
df_train = df_final[df_final['date'] < validation_split_date].copy()
df_val = df_final[(df_final['date'] >= validation_split_date) & (df_final['date'] < split_date)].copy()
df_test = df_final[df_final['date'] >= split_date].copy()

growth_threshold = df_train['youtube_growth_pct'].quantile(best_percentile_threshold)
df_train['is_rapid_growth'] = (df_train['youtube_growth_pct'] >= growth_threshold).astype(int)
df_val['is_rapid_growth'] = (df_val['youtube_growth_pct'] >= growth_threshold).astype(int)
df_test['is_rapid_growth'] = (df_test['youtube_growth_pct'] >= growth_threshold).astype(int)

X_train_full = pd.concat([df_train[features], df_val[features]])
y_train_full = pd.concat([df_train['is_rapid_growth'], df_val['is_rapid_growth']])
X_test = df_test[features]
y_test = df_test['is_rapid_growth']

train_class_counts = y_train_full.value_counts()
scale_pos_weight = train_class_counts[0] / train_class_counts[1] if 1 in train_class_counts else 1.0
lgbm_selector = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, **best_params)
lgbm_selector.fit(X_train_full, y_train_full)
feature_importance = pd.Series(lgbm_selector.feature_importances_, index=X_train_full.columns)
selected_features = feature_importance[feature_importance > feature_importance.median()].index
X_train_full_selected = X_train_full[selected_features]
X_test_selected = X_test[selected_features]

lgbm_final = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, **best_params)
lgbm_final.fit(X_train_full_selected, y_train_full)

# 특징 중요도 출력
feature_importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': lgbm_final.feature_importances_
}).sort_values(by='importance', ascending=False).head(20)
print("\n최종 모델의 특징 중요도 상위 20개:")
print(feature_importance_df)
print("최종 모델 학습 완료")

# 성능 평가
print("최종 모델 성능 평가 시작...")
y_proba_test = lgbm_final.predict_proba(X_test_selected)[:, 1]
y_pred_default = (y_proba_test >= 0.5).astype(int)
y_pred_optimal = (y_proba_test >= best_optimal_threshold).astype(int)

accuracy_default = accuracy_score(y_test, y_pred_default)
precision_default = precision_score(y_test, y_pred_default)
recall_default = recall_score(y_test, y_pred_default)
f1_default = f1_score(y_test, y_pred_default)

print("\n기본 임계값 (0.5)에서의 성능:")
print(f"정확도: {accuracy_default:.4f}")
print(f"정밀도: {precision_default:.4f}")
print(f"재현율: {recall_default:.4f}")
print(f"F1-Score: {f1_default:.4f}")

accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
precision_optimal = precision_score(y_test, y_pred_optimal)
recall_optimal = recall_score(y_test, y_pred_optimal)
f1_optimal = f1_score(y_test, y_pred_optimal)

print(f"\n최적 임계값 ({best_optimal_threshold:.4f})에서의 성능:")
print(f"정확도: {accuracy_optimal:.4f}")
print(f"정밀도: {precision_optimal:.4f}")
print(f"재현율: {recall_optimal:.4f}")
print(f"F1-Score: {f1_optimal:.4f}")

df_test['prediction'] = y_pred_optimal
df_test['actual_label'] = y_test
df_test['artist_name'] = df_test['artist_name'].apply(lambda x: x[0] if isinstance(x, list) else str(x))

# 급성장 예측 데이터 추출 및 저장
predicted_rapid_growth = df_test[df_test['prediction'] == 1][['artist_id', 'artist_name', 'date', 'youtube_growth_pct', 'actual_label']]
print("\n급성장으로 예측된 경우 (날짜별):")
print(predicted_rapid_growth)
predicted_rapid_growth.to_csv('predicted_rapid_growth.csv', index=False)
print("급성장 예측 데이터가 'predicted_rapid_growth.csv'로 저장되었습니다.")

# PR-AUC 계산 및 시각화
precision, recall, _ = precision_recall_curve(y_test, y_proba_test)
pr_auc = auc(recall, precision)
print(f"\nPR-AUC: {pr_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('재현율')
plt.ylabel('정밀도')
plt.title('정밀도-재현율 곡선')
plt.grid(True)
plt.show()

# Baseline 모델: 최근 성장 기반 예측
print("\nBaseline Model: Recent Growth Prediction")

# ───────────────────────────────
# Baseline Model: Recent Growth Prediction
# ───────────────────────────────
past_growth_period = 30

# 1) 과거 성장률 계산 (split 전에 실행)
df_final['youtube_followers_past'] = (
    df_final.groupby('artist_id')['youtube_followers']
            .shift(past_growth_period)
)
df_final['past_growth_pct'] = (
    (df_final['youtube_followers'] - df_final['youtube_followers_past']) /
    (df_final['youtube_followers_past'] + 1)
) * 100

# 2) 최신 컬럼 포함한 상태로 재분할
df_train = df_final[df_final['date'] < validation_split_date].copy()
df_val   = df_final[(df_final['date'] >= validation_split_date) &
                    (df_final['date'] <  split_date)].copy()
df_test  = df_final[df_final['date'] >= split_date].copy()

# 3) Baseline 임계값 및 예측
past_growth_threshold = df_train['past_growth_pct'].quantile(0.90)
df_test['baseline_prediction'] = (
    df_test['past_growth_pct'] >= past_growth_threshold
).astype(int)

print(f"Baseline Model: 과거 성장 임계값 (90th percentile): {past_growth_threshold:.2f}%")
df_test['baseline_prediction'] = (df_test['past_growth_pct'] >= past_growth_threshold).astype(int)

baseline_accuracy = accuracy_score(y_test, df_test['baseline_prediction'])
baseline_precision = precision_score(y_test, df_test['baseline_prediction'])
baseline_recall = recall_score(y_test, df_test['baseline_prediction'])
baseline_f1 = f1_score(y_test, df_test['baseline_prediction'])

print("\nBaseline Model 성능:")
print(f"정확도: {baseline_accuracy:.4f}")
print(f"정밀도: {baseline_precision:.4f}")
print(f"재현율: {baseline_recall:.4f}")
print(f"F1-Score: {baseline_f1:.4f}")

print("\nMain Model vs Baseline Model:")
print(f"Main Model F1-Score: {f1_optimal:.4f}")
print(f"Baseline Model F1-Score: {baseline_f1:.4f}")

# Train-Validation-Test Split 비율 평가
total_samples = len(df_final)
train_samples = len(df_train)
val_samples = len(df_val)
test_samples = len(df_test)

print("\nTrain-Validation-Test Split 비율:")
print(f"Train: {train_samples / total_samples:.2%} ({train_samples} samples)")
print(f"Validation: {val_samples / total_samples:.2%} ({val_samples} samples)")
print(f"Test: {test_samples / total_samples:.2%} ({test_samples} samples)")
