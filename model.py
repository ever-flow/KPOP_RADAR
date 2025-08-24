# =====================================================================
# [BEST SO FAR - Instagram 버전] 저베이스 계정의 급성장 예측 (누수 차단)
#  - 종속변수: Instagram 팔로워
#  - 저베이스 필터: Train 하위 분위 수치 고정 → Val/Test 동일 적용 (누수 차단)
#  - Val/Test 라벨 생성 시 min_pos=1로 설정하여 None 방지
# =====================================================================

# 0) 환경 세팅 ─ 필수 패키지 설치
# =====================================================================
!pip -q install lightgbm optuna koreanize-matplotlib tqdm seaborn -U

# =====================================================================
# 1) 라이브러리 불러오기 & 기본 설정
# =====================================================================
import warnings, logging, random, gc, os
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, precision_recall_curve, auc
)
import koreanize_matplotlib  # 한글 폰트 자동 설정

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

EPS = 1e-8
TARGET_COL = 'instagram_followers'   # 종속변수(핵심)

# =====================================================================
# 2) 데이터 로드 (경로 수정 가능)
# =====================================================================
DATA_DIR = '/content'  # Colab 기준, 필요시 변경
FILES = {
    'followers'     : 'kpopradar_master_followers_wide.csv',
    'album_release' : '240105_artist_album_release.csv',
    'yt_daily'      : 'KAIST_kpopradar_youtube_artists_daily_20230930.csv'
}

followers_fp  = os.path.join(DATA_DIR, FILES['followers'])
album_fp      = os.path.join(DATA_DIR, FILES['album_release'])
yt_daily_fp   = os.path.join(DATA_DIR, FILES['yt_daily'])

df                = pd.read_csv(followers_fp)
album_release_df  = pd.read_csv(album_fp)
youtube_daily_df  = pd.read_csv(yt_daily_fp)

# 날짜 변환
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
album_release_df['publish_date'] = pd.to_datetime(album_release_df['publish_date'], format='%Y%m%d', errors='coerce')
youtube_daily_df['date'] = pd.to_datetime(youtube_daily_df['date'], format='%Y%m%d', errors='coerce')

df.sort_values(['artist_id','date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# =====================================================================
# 3) 1차 필터링 ─ Instagram 결측률 & 관측일수
# =====================================================================
print("▶ 1차 필터링 (Instagram 결측률 ≤20%, 관측≥365일)")

ig_missing_ratio = df.groupby('artist_id')[TARGET_COL].apply(lambda x: x.isna().mean())
valid_ids = ig_missing_ratio[ig_missing_ratio <= 0.20].index
df = df[df['artist_id'].isin(valid_ids)].copy()

obs_counts = df.groupby('artist_id')['date'].nunique()
valid_ids  = obs_counts[obs_counts >= 365].index
df = df[df['artist_id'].isin(valid_ids)].copy()

df.sort_values(['artist_id','date'], inplace=True)
df.reset_index(drop=True, inplace=True)

# =====================================================================
# 4) 피처 엔지니어링 (IG 중심 + 멀티플랫폼 보조)
# =====================================================================
def calculate_rsi(series, window: int = 14):
    diff = series.diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100 - (100 / (1 + rs))

def add_platform_features(df_: pd.DataFrame, col: str, prefix: str):
    grp = df_.groupby('artist_id')[col]
    df_[f'{prefix}_diff']            = grp.diff()
    df_[f'{prefix}_pct_change']      = grp.pct_change(fill_method=None)
    df_[f'{prefix}_rolling_mean_7']  = grp.transform(lambda x: x.rolling(7, 1).mean())
    df_[f'{prefix}_rolling_mean_30'] = grp.transform(lambda x: x.rolling(30, 1).mean())
    df_[f'{prefix}_rolling_std_7']   = grp.transform(lambda x: x.rolling(7, 1).std())
    df_[f'{prefix}_rolling_std_30']  = grp.transform(lambda x: x.rolling(30, 1).std())
    for lag in (1, 7, 30):
        df_[f'{prefix}_lag_{lag}d'] = grp.shift(lag)

print("▶ 피처 생성 (Instagram 메인 + 크로스플랫폼)")

# Instagram 중심 피처
df['ig_diff']            = df.groupby('artist_id')[TARGET_COL].diff()
df['ig_pct_change']      = df.groupby('artist_id')[TARGET_COL].pct_change()
df['ig_rolling_mean_7']  = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.rolling(7, 1).mean())
df['ig_rolling_mean_30'] = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.rolling(30, 1).mean())
df['ig_rsi_14']          = df.groupby('artist_id')[TARGET_COL].transform(calculate_rsi)
df['ig_ema_12']          = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.ewm(span=12, adjust=False).mean())
df['ig_ema_26']          = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['ig_macd']            = df['ig_ema_12'] - df['ig_ema_26']
df['ig_signal']          = df.groupby('artist_id')['ig_macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

# 요일/월
df['dayofweek'] = df['date'].dt.dayofweek
df['month']     = df['date'].dt.month

# 기타 플랫폼 보조 피처
platform_cols = {
    'youtube'  : 'youtube_followers',
    'twitter'  : 'twitter_followers',
    'tiktok'   : 'tiktok_followers',
    'spotify'  : 'spotify_followers',
    'fancafe'  : 'fancafe_followers'
}
for pfx, col in platform_cols.items():
    add_platform_features(df, col, pfx)

# 성장 플래그 & 멀티플랫폼 성장 카운트
growing_flags = ['ig_growing']
df['ig_growing'] = (df['ig_diff'] > 0).astype(int)
for pfx in platform_cols.keys():
    base = f'{pfx}_diff'
    flag = f'{pfx}_growing'
    df[flag] = (df[base] > 0).astype(int)
    growing_flags.append(flag)
df['num_platforms_growing'] = df[growing_flags].sum(axis=1)

# cross-platform ratio (동일 시점)
platform_raw_cols = [TARGET_COL] + list(platform_cols.values())
for i in range(len(platform_raw_cols)):
    for j in range(i + 1, len(platform_raw_cols)):
        col_i, col_j = platform_raw_cols[i], platform_raw_cols[j]
        ratio_name = f'ratio_{col_i.split("_")[0]}_to_{col_j.split("_")[0]}'
        df[ratio_name] = df[col_i] / (df[col_j] + EPS)

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# =====================================================================
# 5) 외부 테이블 병합 (앨범 발매, YouTube 뷰 변동성)
# =====================================================================
print("▶ 외부 테이블 병합")

album_df = album_release_df[['artist_id', 'publish_date']].copy()
album_df['artist_id'] = pd.to_numeric(album_df['artist_id'], errors='coerce')
album_df.dropna(subset=['artist_id'], inplace=True)
album_df.rename(columns={'publish_date': 'date'}, inplace=True)
album_df['album_release_flag'] = 1

df = df.merge(album_df, on=['artist_id', 'date'], how='left')
df['album_release_flag'] = df['album_release_flag'].fillna(0)

# 과거 발매일 ffill → 미래 정보 사용 없음
df['last_album_date'] = df.groupby('artist_id').apply(
    lambda g: g['date'].where(g['album_release_flag'] == 1).ffill()
).reset_index(level=0, drop=True)
df['days_since_last_album'] = (df['date'] - df['last_album_date']).dt.days
max_gap = df['days_since_last_album'].max(skipna=True)
df['days_since_last_album'].fillna(max_gap + 1, inplace=True)
df.drop(columns='last_album_date', inplace=True)

# 365일 내 발매 빈도(과거 누적)
df['album_freq_365'] = df.groupby('artist_id')['album_release_flag'].transform(lambda x: x.rolling(365, 1).sum())

# YouTube viewcount 변동성(보조)
youtube_daily_df.rename(columns={'id': 'artist_id'}, inplace=True)
youtube_daily_df.sort_values(['artist_id', 'date'], inplace=True)
youtube_daily_df['yt_view_std_7']  = youtube_daily_df.groupby('artist_id')['viewcount'].transform(lambda x: x.rolling(7, 1).std())
youtube_daily_df['yt_view_std_30'] = youtube_daily_df.groupby('artist_id')['viewcount'].transform(lambda x: x.rolling(30, 1).std())
vol_cols = ['artist_id', 'date', 'yt_view_std_7', 'yt_view_std_30']
df = df.merge(youtube_daily_df[vol_cols], on=['artist_id', 'date'], how='left')

# =====================================================================
# 6) 날짜 분할 (Train / Val / Test)
# =====================================================================
LATEST_DATE = df['date'].max()
TEST_DAYS   = 365
VAL_DAYS    = 180

TEST_START  = LATEST_DATE - pd.DateOffset(days=TEST_DAYS - 1)
VAL_START   = TEST_START - pd.DateOffset(days=VAL_DAYS)

df_train = df[df['date'] < VAL_START].copy()
df_val   = df[(df['date'] >= VAL_START) & (df['date'] < TEST_START)].copy()
df_test  = df[df['date'] >= TEST_START].copy()

print(f"Train: {df_train['date'].min().date()} ~ {df_train['date'].max().date()}  ({len(df_train):,} rows)")
print(f"Val  : {df_val['date'].min().date()} ~ {df_val['date'].max().date()}  ({len(df_val):,} rows)")
print(f"Test : {df_test['date'].min().date()} ~ {df_test['date'].max().date()}  ({len(df_test):,} rows)")

# =====================================================================
# 7) 결측치 정책 (누수 방지) ─ Train 기준으로만 결정
# =====================================================================
print("▶ 결측치 처리 정책 결정 (Train 기준)")

train_missing_ratio = df_train.isna().mean()
high_missing_cols = train_missing_ratio[train_missing_ratio > 0.9].index.tolist()
mid_missing_cols  = train_missing_ratio[(train_missing_ratio > 0.4) & (train_missing_ratio <= 0.9)].index.tolist()

def apply_missing_policy(df_part: pd.DataFrame,
                         drop_cols: list,
                         mid_cols: list) -> pd.DataFrame:
    df_part = df_part.copy()
    for col in mid_cols:
        if col in df_part.columns:
            df_part[f'{col}_missing'] = df_part[col].isna().astype(int)
    drop_actual = [c for c in drop_cols if c in df_part.columns]
    df_part.drop(columns=drop_actual, inplace=True, errors='ignore')
    return df_part

df_train = apply_missing_policy(df_train, high_missing_cols, mid_missing_cols)
df_val   = apply_missing_policy(df_val,   high_missing_cols, mid_missing_cols)
df_test  = apply_missing_policy(df_test,  high_missing_cols, mid_missing_cols)

# 분할 간 공통 피처 교집합만 사용
common_cols = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
df_train = df_train[[c for c in df_train.columns if c in common_cols]]
df_val   = df_val[[c for c in df_val.columns   if c in common_cols]]
df_test  = df_test[[c for c in df_test.columns if c in common_cols]]

# =====================================================================
# 8) 라벨 생성 (Instagram 성장률 기준)
#   - Train: perc_threshold 사용 + min_pos_train 적용(양성 부족 방지)
#   - Val/Test: fixed_threshold 사용 + min_pos=1 (None 방지)
#   - 저베이스 필터: Train 하위 분위 수치를 수치로 고정 후 전 구간 동일 적용
# =====================================================================
def label_growth(df_subset: pd.DataFrame,
                 growth_days: int,
                 perc_threshold: float = None,   # Train에서만 사용
                 fixed_threshold: float = None,  # Val/Test/TrainVal 적용
                 min_pos: int = 50,              # Train에서는 50, Val/Test는 1
                 target_col: str = TARGET_COL):
    d = df_subset.copy()
    d.sort_values(['artist_id','date'], inplace=True)

    d['target_future'] = d.groupby('artist_id')[target_col].shift(-growth_days)
    d = d.dropna(subset=['target_future'])
    d = d[d['target_future'] > 0]

    d['growth_pct'] = ((d['target_future'] - d[target_col]) / (d[target_col] + 1)) * 100
    d = d[(~d['growth_pct'].isna()) & (~np.isinf(d['growth_pct']))]

    if fixed_threshold is not None:
        current_thr = float(fixed_threshold)
    elif perc_threshold is not None:
        current_thr = float(d['growth_pct'].quantile(perc_threshold))
    else:
        raise ValueError("perc_threshold 또는 fixed_threshold 중 하나는 반드시 제공")

    d['is_rapid_growth'] = (d['growth_pct'] >= current_thr).astype(int)

    if d['is_rapid_growth'].sum() < min_pos:
        return None, None
    return d, float(current_thr)

# 저베이스 컷(Train 하위 분위 수치 → 고정)
LOW_BASE_CUT_PT = 0.80
base_cut_value = df_train[TARGET_COL].quantile(LOW_BASE_CUT_PT)

def apply_low_base_filter(df_part: pd.DataFrame, cut_value: float) -> pd.DataFrame:
    return df_part[df_part[TARGET_COL] <= cut_value].copy()

df_train_low = apply_low_base_filter(df_train, base_cut_value)
df_val_low   = apply_low_base_filter(df_val,   base_cut_value)
df_test_low  = apply_low_base_filter(df_test,  base_cut_value)

# =====================================================================
# 9) 그리드 서치 (growth_period × percentile) ─ 저베이스만
# =====================================================================
print("▶ 그리드 서치 (Instagram 성장률, 저베이스 전용)")

GROWTH_PERIODS        = [50, 100, 150]
PERCENTILE_THRESHOLDS = [0.80, 0.90, 0.95]
BEST_CONF, BEST_F1 = None, -1
GRID_LOG = []

BASE_PARAMS = dict(
    n_estimators       = 2000,
    learning_rate      = 0.05,
    num_leaves         = 63,
    max_depth          = -1,
    min_child_samples  = 20,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    reg_alpha          = 0.01,
    reg_lambda         = 0.01,
    random_state       = RANDOM_SEED,
    objective          = 'binary'
)

EXCLUDE = {'artist_id','artist_name','date','target_future','growth_pct','is_rapid_growth'}
FEATURES_ALL = [c for c in df_train_low.columns if c not in EXCLUDE]

for gp in GROWTH_PERIODS:
    for pt in PERCENTILE_THRESHOLDS:
        # Train 라벨: perc_threshold + min_pos_train
        tr_lab, thr_train = label_growth(df_train_low, gp, perc_threshold=pt, min_pos=50, target_col=TARGET_COL)
        if tr_lab is None:
            print(f"   gp={gp}, pt={pt:.2f} → Train positive 부족, skip")
            continue

        # Val 라벨: fixed_threshold + min_pos=1
        va_lab, _ = label_growth(df_val_low, gp, fixed_threshold=thr_train, min_pos=1, target_col=TARGET_COL)
        if va_lab is None:
            print(f"   gp={gp}, pt={pt:.2f} → Val positive 부족, skip")
            continue

        X_tr, y_tr = tr_lab[FEATURES_ALL], tr_lab['is_rapid_growth']
        X_va, y_va = va_lab[FEATURES_ALL], va_lab['is_rapid_growth']

        counts = y_tr.value_counts()
        scale_pos_weight = (counts.get(0,0) / counts.get(1,1)) if 1 in counts else 1.0

        clf = lgb.LGBMClassifier(**BASE_PARAMS, scale_pos_weight=scale_pos_weight)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=0)]
        )

        # Val에서 확률 임계값 탐색
        y_va_prob = clf.predict_proba(X_va)[:, 1]
        prec, rec, thr_arr = precision_recall_curve(y_va, y_va_prob)
        if len(thr_arr) == 0:
            # 안전 가드: 극단 케이스
            thr_prob = 0.5
            f1 = 0.0
        else:
            beta = 1.0
            fbeta = (1+beta**2) * prec * rec / np.clip((beta**2) * prec + rec, EPS, None)
            thr_prob = float(thr_arr[np.argmax(fbeta[:len(thr_arr)])])
            y_va_pred = (y_va_prob >= thr_prob).astype(int)
            f1 = f1_score(y_va, y_va_pred)

        GRID_LOG.append({
            'gp': gp, 'pt': pt, 'f1': f1,
            'thr_prob': thr_prob,
            'thr_train': float(thr_train),
            'best_iter': int(clf.best_iteration_),
            'spw': float(scale_pos_weight)
        })
        print(f"   gp={gp:3d}, pt={pt:.2f},  F1={f1:.4f}")

        if f1 > BEST_F1:
            BEST_F1 = f1
            BEST_CONF = GRID_LOG[-1]

print(f"\n▷ Best Config = {BEST_CONF},  Val-F1={BEST_F1:.4f}")

# 시각화 (선택)
grid_df = pd.DataFrame(GRID_LOG)
if not grid_df.empty:
    pivot = grid_df.pivot(index='gp', columns='pt', values='f1')
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap='viridis')
    plt.title('Grid Search F1 Scores by Growth Period & Percentile (Instagram, Low-Base)')
    plt.xlabel('Percentile Threshold')
    plt.ylabel('Growth Period (days)')
    plt.tight_layout()
    plt.show()

# =====================================================================
# 10) 최종 모델 학습 + 테스트 평가 (저베이스 전용)
# =====================================================================
print("\n▶ 최종 모델 학습 및 평가 (누수 차단·저베이스 전용)")

gp          = int(BEST_CONF['gp'])
pt          = float(BEST_CONF['pt'])
thr_train   = float(BEST_CONF['thr_train'])
best_iter   = int(BEST_CONF['best_iter'])
thr_prob    = float(BEST_CONF['thr_prob'])

# Train+Val 라벨: fixed_threshold + min_pos=1 (None 방지)
trainval_low = pd.concat([df_train_low, df_val_low], ignore_index=True)
trainval_lab, _ = label_growth(trainval_low, gp, fixed_threshold=thr_train, min_pos=1, target_col=TARGET_COL)
if trainval_lab is None or len(trainval_lab) == 0:
    raise RuntimeError("Train/Val 라벨 생성 실패: 양성 표본이 전무합니다. 저베이스 컷/퍼센타일/기간을 조정하세요.")

# Test 라벨: fixed_threshold + min_pos=1 (None 방지)
test_lab, _ = label_growth(df_test_low, gp, fixed_threshold=thr_train, min_pos=1, target_col=TARGET_COL)
if (test_lab is None) or (len(test_lab) == 0):
    # 극단 케이스에서도 평가가 가능하도록 안전 처리
    print("경고: Test 구간에서 라벨 생성 표본이 매우 적습니다(또는 0). 평가 지표가 왜곡될 수 있습니다.")
    # 빈 데이터프레임이면 예측/저장 단계만 스킵
    test_lab = pd.DataFrame(columns=list(trainval_lab.columns))

# 피처 목록 (라벨 관련 열 제외)
EXCLUDE_FINAL = {'artist_id','artist_name','date','target_future','growth_pct','is_rapid_growth'}
FEATURES_ALL_TV = [c for c in trainval_lab.columns if c not in EXCLUDE_FINAL]

# 학습
counts_tv = trainval_lab['is_rapid_growth'].value_counts()
scale_pos_weight_tv = (counts_tv.get(0,0) / counts_tv.get(1,1)) if 1 in counts_tv else 1.0

final_params = dict(
    n_estimators      = best_iter,
    learning_rate     = BASE_PARAMS['learning_rate'],
    num_leaves        = BASE_PARAMS['num_leaves'],
    max_depth         = BASE_PARAMS['max_depth'],
    min_child_samples = BASE_PARAMS['min_child_samples'],
    subsample         = BASE_PARAMS['subsample'],
    colsample_bytree  = BASE_PARAMS['colsample_bytree'],
    reg_alpha         = BASE_PARAMS['reg_alpha'],
    reg_lambda        = BASE_PARAMS['reg_lambda'],
    random_state      = RANDOM_SEED,
    objective         = 'binary'
)
final_clf = lgb.LGBMClassifier(**final_params, scale_pos_weight=scale_pos_weight_tv)
X_tv, y_tv = trainval_lab[FEATURES_ALL_TV], trainval_lab['is_rapid_growth']
final_clf.fit(X_tv, y_tv)

# 평가 함수 (양성/음성 전무 시도 안전 가드)
def report_at_threshold(y_true, y_prob, thr, title_suffix=""):
    if len(y_true) == 0:
        print(f"[스킵] Test 표본이 없어 {title_suffix} 평가를 생략합니다.")
        return
    y_pred = (y_prob >= thr).astype(int)
    try:
        pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(pr_rec, pr_prec)
    except Exception:
        pr_auc = float('nan')
    # 지표 계산 가드
    def safe_metric(fn, default=np.nan):
        try:
            return fn()
        except Exception:
            return default
    acc = safe_metric(lambda: accuracy_score(y_true, y_pred))
    prec = safe_metric(lambda: precision_score(y_true, y_pred, zero_division=0))
    rec  = safe_metric(lambda: recall_score(y_true, y_pred, zero_division=0))
    f1   = safe_metric(lambda: f1_score(y_true, y_pred, zero_division=0))

    print(f"Test Metrics {title_suffix} @prob_thr={thr:.3f}  (gp={gp}, IG_label_thr_from_train={thr_train:.3f}, low_base_cut={LOW_BASE_CUT_PT:.2f})")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")

# Test 예측
if len(test_lab) > 0:
    X_te, y_te = test_lab[FEATURES_ALL_TV], test_lab['is_rapid_growth']
    y_te_prob  = final_clf.predict_proba(X_te)[:, 1]
    report_at_threshold(y_te, y_te_prob, thr_prob, title_suffix="(Val기준 임계값)")
    print()
    report_at_threshold(y_te, y_te_prob, 0.50, title_suffix="(0.50 고정)")

    # PR-커브
    try:
        pr_prec, pr_rec, _ = precision_recall_curve(y_te, y_te_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(pr_rec, pr_prec)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Test Set, Instagram, Low-Base)')
        plt.tight_layout(); plt.show()
    except Exception:
        print("PR-커브 생성 불가(표본 부족).")

    # 결과 저장
    out_df = test_lab.copy()
    out_df['prob'] = y_te_prob
    out_df['pred'] = (y_te_prob >= thr_prob).astype(int)
    rapid = out_df[out_df['pred'] == 1][['artist_id', 'artist_name', 'date',
                                         'growth_pct', 'prob', TARGET_COL]]
    rapid = rapid.sort_values('prob', ascending=False)
    save_name = 'predicted_rapid_growth_final_instagram_lowbase.csv'
    rapid.to_csv(save_name, index=False)
    print(f"\n급성장이 예상된 샘플 CSV가 저장되었습니다 → {save_name}")
else:
    print("주의: Test 구간 라벨 표본이 없어 예측/저장 단계를 생략했습니다.")

print(f"저베이스 컷 값(Train {LOW_BASE_CUT_PT:.0%} 분위): {base_cut_value:,.0f} IG 팔로워 이하")
