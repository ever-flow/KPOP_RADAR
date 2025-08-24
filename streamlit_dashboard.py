import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import koreanize_matplotlib
import warnings
import seaborn as sns
import os
import random
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# =====================================================================
# Streamlit 페이지 설정
# =====================================================================
st.set_page_config(page_title="K-pop 급성장 예측 v2.1 (Instagram 기반)", layout="wide")

# 세션 상태 초기화
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'prev_gp' not in st.session_state:
    st.session_state.prev_gp = None
if 'prev_pt' not in st.session_state:
    st.session_state.prev_pt = None

# =====================================================================
# 데이터 로딩 함수 (캐시 사용)
# =====================================================================
@st.cache_data
def load_data():
    """세 개의 기본 데이터 파일을 로드합니다."""
    with st.spinner("기본 데이터 로딩 중... (최초 1회 실행)"):
        # 사용자의 기존 파일 경로를 유지합니다. 환경에 맞게 수정해주세요.
        try:
            df = pd.read_csv('C:/Users/gimyo/OneDrive/Desktop/KAIST_kpopradar_instagram_20230930/kpopradar_master_followers_wide.csv')
            album_release_df = pd.read_csv('C:/Users/gimyo/OneDrive/Desktop/KAIST_kpopradar_instagram_20230930/240105_artist_album_release.csv')
            youtube_daily_df = pd.read_csv('C:/Users/gimyo/OneDrive/Desktop/KAIST_kpopradar_instagram_20230930/KAIST_kpopradar_youtube_artists_daily_20230930.csv')
        except FileNotFoundError:
            st.error("데이터 파일 경로를 찾을 수 없습니다. 스크립트의 `load_data` 함수 내 경로를 확인해주세요.")
            return None, None, None
            
    # 날짜 형식 변환
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
    album_release_df['publish_date'] = pd.to_datetime(album_release_df['publish_date'], format='%Y%m%d', errors='coerce')
    youtube_daily_df['date'] = pd.to_datetime(youtube_daily_df['date'], format='%Y%m%d', errors='coerce')
    
    df.sort_values(['artist_id','date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df, album_release_df, youtube_daily_df

# =====================================================================
# 개선된 분석 로직을 포함한 전처리 및 학습 함수
# =====================================================================
@st.cache_data
def run_analysis_pipeline(gp_fixed, pt_fixed):
    """개선된 분석 파이프라인을 실행하여 모델 결과와 시각화 데이터를 생성합니다."""
    
    with st.spinner("데이터 전처리 및 모델 학습 중... (시간이 다소 소요될 수 있습니다)"):
        df_raw, album_release_df, youtube_daily_df = load_data()
        if df_raw is None: return None

        EPS = 1e-8
        TARGET_COL = 'instagram_followers'
        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        df = df_raw.copy()
        df.sort_values(['artist_id','date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 1차 필터링
        ig_missing_ratio = df.groupby('artist_id')[TARGET_COL].apply(lambda x: x.isna().mean())
        valid_ids = ig_missing_ratio[ig_missing_ratio <= 0.20].index
        df = df[df['artist_id'].isin(valid_ids)].copy()

        obs_counts = df.groupby('artist_id')['date'].nunique()
        valid_ids = obs_counts[obs_counts >= 365].index
        df = df[df['artist_id'].isin(valid_ids)].copy()
        df.sort_values(['artist_id','date'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 피처 엔지니어링
        def calculate_rsi(series, window: int = 14):
            diff = series.diff()
            gain = diff.clip(lower=0)
            loss = (-diff).clip(lower=0)
            avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
            avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
            rs = avg_gain / (avg_loss + EPS)
            return 100 - (100 / (1 + rs))

        def add_platform_features(df_, col, prefix):
            grp = df_.groupby('artist_id')[col]
            df_[f'{prefix}_diff'] = grp.diff()
            df_[f'{prefix}_pct_change'] = grp.pct_change(fill_method=None)
            df_[f'{prefix}_rolling_mean_7'] = grp.transform(lambda x: x.rolling(7, 1).mean())
            df_[f'{prefix}_rolling_mean_30'] = grp.transform(lambda x: x.rolling(30, 1).mean())
            df_[f'{prefix}_rolling_std_7'] = grp.transform(lambda x: x.rolling(7, 1).std())
            df_[f'{prefix}_rolling_std_30'] = grp.transform(lambda x: x.rolling(30, 1).std())
            for lag in (1, 7, 30):
                df_[f'{prefix}_lag_{lag}d'] = grp.shift(lag)

        # Instagram 중심 피처
        df['ig_diff'] = df.groupby('artist_id')[TARGET_COL].diff()
        df['ig_pct_change'] = df.groupby('artist_id')[TARGET_COL].pct_change()
        df['ig_rolling_mean_7'] = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.rolling(7, 1).mean())
        df['ig_rolling_mean_30'] = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.rolling(30, 1).mean())
        df['ig_rsi_14'] = df.groupby('artist_id')[TARGET_COL].transform(calculate_rsi)
        df['ig_ema_12'] = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.ewm(span=12, adjust=False).mean())
        df['ig_ema_26'] = df.groupby('artist_id')[TARGET_COL].transform(lambda x: x.ewm(span=26, adjust=False).mean())
        df['ig_macd'] = df['ig_ema_12'] - df['ig_ema_26']
        df['ig_signal'] = df.groupby('artist_id')['ig_macd'].transform(lambda x: x.ewm(span=9, adjust=False).mean())

        # 요일/월
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        # 기타 플랫폼 보조 피처
        platform_cols = {
            'youtube': 'youtube_followers',
            'twitter': 'twitter_followers',
            'tiktok': 'tiktok_followers',
            'spotify': 'spotify_followers',
            'fancafe': 'fancafe_followers'
        }
        for pfx, col in platform_cols.items():
            add_platform_features(df, col, pfx)

        # 모멘텀 & 가속도 피처
        df['ig_momentum_7_30'] = df['ig_rolling_mean_7'] / (df['ig_rolling_mean_30'] + EPS)
        df['yt_momentum_7_30'] = df['youtube_rolling_mean_7'] / (df['youtube_rolling_mean_30'] + EPS)
        df['ig_diff_acceleration'] = df.groupby('artist_id')['ig_diff'].diff()
        df['yt_diff_acceleration'] = df.groupby('artist_id')['youtube_diff'].diff()

        # 성장 플래그 & 멀티플랫폼 성장 카운트
        growing_flags = ['ig_growing']
        df['ig_growing'] = (df['ig_diff'] > 0).astype(int)
        for pfx in platform_cols.keys():
            base = f'{pfx}_diff'
            flag = f'{pfx}_growing'
            df[flag] = (df[base] > 0).astype(int)
            growing_flags.append(flag)
        df['num_platforms_growing'] = df[growing_flags].sum(axis=1)

        # cross-platform ratio
        platform_raw_cols = [TARGET_COL] + list(platform_cols.values())
        for i in range(len(platform_raw_cols)):
            for j in range(i + 1, len(platform_raw_cols)):
                col_i, col_j = platform_raw_cols[i], platform_raw_cols[j]
                ratio_name = f'ratio_{col_i.split("_")[0]}_to_{col_j.split("_")[0]}'
                df[ratio_name] = df[col_i] / (df[col_j] + EPS)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 외부 테이블 병합
        album_df = album_release_df[['artist_id', 'publish_date']].copy()
        album_df['artist_id'] = pd.to_numeric(album_df['artist_id'], errors='coerce')
        album_df.dropna(subset=['artist_id'], inplace=True)
        album_df.rename(columns={'publish_date': 'date'}, inplace=True)
        album_df['album_release_flag'] = 1

        df = df.merge(album_df, on=['artist_id', 'date'], how='left')
        df['album_release_flag'] = df['album_release_flag'].fillna(0)

        df['last_album_date'] = df.groupby('artist_id').apply(
            lambda g: g['date'].where(g['album_release_flag'] == 1).ffill()
        ).reset_index(level=0, drop=True)
        df['days_since_last_album'] = (df['date'] - df['last_album_date']).dt.days
        max_gap = df['days_since_last_album'].max(skipna=True)
        df['days_since_last_album'].fillna(max_gap + 1, inplace=True)
        df.drop(columns='last_album_date', inplace=True)

        df['album_freq_365'] = df.groupby('artist_id')['album_release_flag'].transform(lambda x: x.rolling(365, 1).sum())

        youtube_daily_df.rename(columns={'id': 'artist_id'}, inplace=True)
        youtube_daily_df.sort_values(['artist_id', 'date'], inplace=True)
        youtube_daily_df['yt_view_std_7'] = youtube_daily_df.groupby('artist_id')['viewcount'].transform(lambda x: x.rolling(7, 1).std())
        youtube_daily_df['yt_view_std_30'] = youtube_daily_df.groupby('artist_id')['viewcount'].transform(lambda x: x.rolling(30, 1).std())
        vol_cols = ['artist_id', 'date', 'yt_view_std_7', 'yt_view_std_30']
        df = df.merge(youtube_daily_df[vol_cols], on=['artist_id', 'date'], how='left')

        df['yt_volatility_momentum_7_30'] = df['yt_view_std_7'] / (df['yt_view_std_30'] + EPS)

        # 날짜 분할
        LATEST_DATE = df['date'].max()
        TEST_DAYS = 365
        VAL_DAYS = 180

        TEST_START = LATEST_DATE - pd.DateOffset(days=TEST_DAYS - 1)
        VAL_START = TEST_START - pd.DateOffset(days=VAL_DAYS)

        df_train = df[df['date'] < VAL_START].copy()
        df_val = df[(df['date'] >= VAL_START) & (df['date'] < TEST_START)].copy()
        df_test = df[df['date'] >= TEST_START].copy()

        # 결측치 처리
        train_missing_ratio = df_train.isna().mean()
        high_missing_cols = train_missing_ratio[train_missing_ratio > 0.9].index.tolist()
        mid_missing_cols = train_missing_ratio[(train_missing_ratio > 0.4) & (train_missing_ratio <= 0.9)].index.tolist()

        def apply_missing_policy(df_part, drop_cols, mid_cols):
            df_part = df_part.copy()
            for col in mid_cols:
                if col in df_part.columns:
                    df_part[f'{col}_missing'] = df_part[col].isna().astype(int)
            drop_actual = [c for c in drop_cols if c in df_part.columns]
            df_part.drop(columns=drop_actual, inplace=True, errors='ignore')
            return df_part

        df_train = apply_missing_policy(df_train, high_missing_cols, mid_missing_cols)
        df_val = apply_missing_policy(df_val, high_missing_cols, mid_missing_cols)
        df_test = apply_missing_policy(df_test, high_missing_cols, mid_missing_cols)

        common_cols = set(df_train.columns) & set(df_val.columns) & set(df_test.columns)
        df_train = df_train[[c for c in df_train.columns if c in common_cols]]
        df_val = df_val[[c for c in df_val.columns if c in common_cols]]
        df_test = df_test[[c for c in df_test.columns if c in common_cols]]

        # 라벨 생성 함수
        def label_growth(df_subset, growth_days, perc_threshold=None, fixed_threshold=None, min_pos=50, target_col=TARGET_COL):
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

        # 저베이스 필터
        LOW_BASE_CUT_PT = 0.80
        base_cut_value = df_train[TARGET_COL].quantile(LOW_BASE_CUT_PT)

        def apply_low_base_filter(df_part, cut_value):
            return df_part[df_part[TARGET_COL] <= cut_value].copy()

        df_train_low = apply_low_base_filter(df_train, base_cut_value)
        df_val_low = apply_low_base_filter(df_val, base_cut_value)
        df_test_low = apply_low_base_filter(df_test, base_cut_value)

        # 단일 gp/pt로 모델링 (사이드바 선택 사용)
        tr_lab, thr_growth = label_growth(df_train_low, gp_fixed, perc_threshold=pt_fixed, min_pos=50, target_col=TARGET_COL)
        if tr_lab is None:
            return None

        va_lab, _ = label_growth(df_val_low, gp_fixed, fixed_threshold=thr_growth, min_pos=1, target_col=TARGET_COL)
        if va_lab is None:
            return None

        EXCLUDE = {'artist_id','artist_name','date','target_future','growth_pct','is_rapid_growth'}
        FEATURES = [c for c in tr_lab.columns if c not in EXCLUDE]

        X_tr, y_tr = tr_lab[FEATURES], tr_lab['is_rapid_growth']
        X_va, y_va = va_lab[FEATURES], va_lab['is_rapid_growth']

        counts = y_tr.value_counts()
        scale_pos_weight = (counts.get(0,0) / counts.get(1,1)) if 1 in counts else 1.0

        BASE_PARAMS = dict(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=0.01,
            random_state=RANDOM_SEED,
            objective='binary'
        )

        clf = lgb.LGBMClassifier(**BASE_PARAMS, scale_pos_weight=scale_pos_weight)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])

        y_va_prob = clf.predict_proba(X_va)[:, 1]
        prec, rec, thr_arr = precision_recall_curve(y_va, y_va_prob)
        beta = 1.0
        fbeta = (1+beta**2) * prec * rec / np.clip((beta**2) * prec + rec, EPS, None)
        thr_prob = float(thr_arr[np.argmax(fbeta[:-1])] if len(thr_arr) > 1 else 0.5)

        trainval_low = pd.concat([df_train_low, df_val_low], ignore_index=True)
        trainval_lab, _ = label_growth(trainval_low, gp_fixed, fixed_threshold=thr_growth, min_pos=1, target_col=TARGET_COL)
        test_lab, _ = label_growth(df_test_low, gp_fixed, fixed_threshold=thr_growth, min_pos=1, target_col=TARGET_COL)

        if trainval_lab is None or test_lab is None:
            return None

        X_tv, y_tv = trainval_lab[FEATURES], trainval_lab['is_rapid_growth']
        X_te, y_te = test_lab[FEATURES], test_lab['is_rapid_growth']

        counts_tv = y_tv.value_counts()
        scale_pos_weight_tv = (counts_tv.get(0,0) / counts_tv.get(1,1)) if 1 in counts_tv else 1.0

        final_params = BASE_PARAMS.copy()
        final_params['n_estimators'] = clf.best_iteration_
        final_clf = lgb.LGBMClassifier(**final_params, scale_pos_weight=scale_pos_weight_tv)
        final_clf.fit(X_tv, y_tv)

        y_te_prob = final_clf.predict_proba(X_te)[:, 1]

        test_lab['prediction'] = (y_te_prob >= thr_prob).astype(int)
        test_lab['prediction_proba'] = y_te_prob
        test_lab['actual_label'] = test_lab['is_rapid_growth']

        predicted_rapid_growth = test_lab[test_lab['prediction'] == 1].copy()
        recent_rapid_growth = predicted_rapid_growth[predicted_rapid_growth['date'] == predicted_rapid_growth['date'].max()]

        precision, recall, thresholds = precision_recall_curve(y_te, y_te_prob)
        pr_auc_score = auc(recall, precision)

        # 피처 중요도
        feature_importances = pd.DataFrame({
            'feature': X_tv.columns,
            'importance': final_clf.feature_importances_
        }).sort_values('importance', ascending=False)

        results = {
            "predicted_rapid_growth": predicted_rapid_growth,
            "recent_rapid_growth": recent_rapid_growth,
            "precision_curve": precision,
            "recall_curve": recall,
            "pr_thresholds": np.append(thresholds, 1),
            "pr_auc": pr_auc_score,
            "growth_threshold": thr_growth,
            "optimal_prob_threshold": thr_prob,
            "latest_prediction_date": test_lab['date'].max(),
            "feature_importances": feature_importances,
            "low_base_cut": base_cut_value,
            "target_col": TARGET_COL
        }
        return results

# =====================================================================
# 사이드바 설정
# =====================================================================
with st.sidebar:
    st.header("⚙️ 모델 설정")
    gp_select = st.selectbox("성장 관찰 기간 (Growth Period)", [50], index=0)
    pt_select = st.selectbox("성장률 기준 분위수 (Percentile)", [0.80, 0.90, 0.95], 
                             format_func=lambda x: f"상위 {(1-x)*100:.0f}%", index=0)

# 선택 변경 감지 및 재계산 트리거
if gp_select != st.session_state.prev_gp or pt_select != st.session_state.prev_pt:
    st.session_state.model_results = None
    st.session_state.prev_gp = gp_select
    st.session_state.prev_pt = pt_select

# 모델 결과 로드/실행
if st.session_state.model_results is None:
    st.session_state.model_results = run_analysis_pipeline(gp_select, pt_select)

# =====================================================================
# 대시보드 UI 렌더링
# =====================================================================

st.title("📈 K-pop 아티스트 급성장 예측 (v2.1, Instagram 기반)")

if st.session_state.model_results:
    res = st.session_state.model_results
    st.write(f"**분석 기간 (Test Set):** `{res['predicted_rapid_growth']['date'].min().strftime('%Y-%m-%d')}` ~ `{res['predicted_rapid_growth']['date'].max().strftime('%Y-%m-%d')}`")
    st.write(f"**라벨링 기준 성장률 (Train 기준):** `{res['growth_threshold']:.2f}%` (상위 {(1-pt_select)*100:.0f}%)")
    st.write(f"**최적 예측 확률 임계값 (Val 기준):** `{res['optimal_prob_threshold']:.3f}`")
    st.write(f"**저베이스 컷 (Train 80% 분위):** `{res['low_base_cut']:,.0f}` Instagram 팔로워 이하")
    st.markdown("---")

    st.header("🔍 전체 급성장 예측 결과 탐색")
    with st.container():
        st.subheader("필터")
        col1, col2 = st.columns([2, 2])
        
        all_artists = sorted(res['predicted_rapid_growth']['artist_name'].unique())
        with col1:
            artist_filter = st.multiselect("아티스트 선택", options=all_artists, placeholder="모든 아티스트")
        
        date_min_val = res['predicted_rapid_growth']['date'].min().date()
        date_max_val = res['predicted_rapid_growth']['date'].max().date()
        with col2:
            date_filter = st.date_input("날짜 범위 선택", [date_min_val, date_max_val], min_value=date_min_val, max_value=date_max_val)

        filtered_df = res['predicted_rapid_growth'].copy()
        if artist_filter:
            filtered_df = filtered_df[filtered_df['artist_name'].isin(artist_filter)]
        if len(date_filter) == 2:
            filtered_df = filtered_df[(filtered_df['date'].dt.date >= date_filter[0]) & (filtered_df['date'].dt.date <= date_filter[1])]

        filtered_display = filtered_df.copy()
        filtered_display_renamed = filtered_display.rename(columns={
            'artist_name': '아티스트', 'date': '날짜', 'instagram_followers': '당시 팔로워',
            'growth_pct': '실제 성장률(%)', 'prediction_proba': '예측 확률', 'actual_label': '실제 결과'
        })
        filtered_display_renamed['날짜'] = filtered_display_renamed['날짜'].dt.strftime('%Y-%m-%d')
        filtered_display_renamed['실제 성장률(%)'] = filtered_display_renamed['실제 성장률(%)'].round(2)
        filtered_display_renamed['예측 확률'] = filtered_display_renamed['예측 확률'].round(3)
        filtered_display_renamed['실제 결과'] = filtered_display_renamed['실제 결과'].map({1: '✅ 실제 급성장', 0: '❌ 해당 없음'})
        st.dataframe(filtered_display_renamed[['아티스트', '날짜', '당시 팔로워', '실제 성장률(%)', '예측 확률', '실제 결과']], use_container_width=True, hide_index=True)

        csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("필터링된 데이터 CSV 다운로드", csv, "filtered_predictions.csv", "text/csv")
    st.markdown("---")
    
    st.header("📊 모델 성능: 정밀도-재현율(PR) 곡선")
    prob_threshold_slider = st.slider("확률 임계값 조정", 0.0, 1.0, res['optimal_prob_threshold'], 0.01, format="%.2f",
                                      help="이 값을 기준으로 '급성장' 여부를 판단합니다. 값이 높을수록 더 확실한 경우만 예측합니다.")

    idx = np.argmin(np.abs(res['pr_thresholds'] - prob_threshold_slider))
    selected_precision = res['precision_curve'][idx]
    selected_recall = res['recall_curve'][idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['recall_curve'], res['precision_curve'], marker='.', label='정밀도-재현율 곡선', zorder=1)
    ax.scatter(selected_recall, selected_precision, color='red', s=100, label=f'선택된 임계값 ({prob_threshold_slider:.2f})', zorder=2)
    
    ax.set_xlabel('재현율 (Recall) - 실제 급성장을 얼마나 잡아냈는가')
    ax.set_ylabel('정밀도 (Precision) - 예측이 얼마나 정확했는가')
    ax.set_title(f'정밀도-재현율 곡선 (Test Set PR-AUC: {res["pr_auc"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.5)
    
    ax.annotate(f'정밀도: {selected_precision:.2f}\n재현율: {selected_recall:.2f}',
                xy=(selected_recall, selected_precision),
                xytext=(10, -40), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7))
    st.pyplot(fig)

    st.markdown("---")
    st.header("🧩 최종 모델 피처 중요도 (Top 30)")
    fig_fi, ax_fi = plt.subplots(figsize=(10, 15))
    sns.barplot(x='importance', y='feature', data=res['feature_importances'].head(30), ax=ax_fi)
    ax_fi.set_title('Top 30 Feature Importances (Final Model)')
    st.pyplot(fig_fi)

else:
    st.error("분석 파이프라인 실행 중 오류가 발생했습니다. 터미널/콘솔 로그를 확인해주세요.")
