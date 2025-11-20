import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from outlier_detectors import DiffDetector, AdaptiveVarianceDetector
from time_series_models import MAModel, ARModel, ARMAModel
import argparse
import datetime

# 仅分析，不修改原始代码逻辑；输出建议到 analysis/ 目录

DIFF_LAMBDA_GRID = [5, 8, 10, 12, 15, 18, 20, 25]
DIFF_K_GRID = [0, 2, 5, 8, 12]
AV_ALPHA_GRID = [0.001, 0.003, 0.005, 0.007, 0.01]
AV_QUANTILE_GRID = [0.99, 0.995, 0.997, 0.999]

TARGET_MIN = 0.005  # 0.5%
TARGET_MAX = 0.05   # 5%


def safe_read_csv(path: str, datetime_col: str = 'date_time'):
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1')
    if datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    return df


def ratio(labels, tag):
    if labels is None or len(labels) == 0:
        return 0.0
    return float((labels == tag).sum()) / float(len(labels))


def analyze_diff(values: np.ndarray):
    serie = pd.Series(values).dropna()
    if len(serie) < 30:
        return {"recommendation": None, "note": "too_few_points", "grid": []}

    results = []
    for lam in DIFF_LAMBDA_GRID:
        for k in DIFF_K_GRID:
            det = DiffDetector(lambda_centrada=lam, k=k)
            out = det.detect(serie.values)
            labels = out['labels']
            out_ratio = ratio(labels, 'outlier')
            results.append({
                'lambda_centrada': lam,
                'k': k,
                'outlier_ratio': out_ratio,
                'threshold_used': out.get('threshold_usado'),
                'std_used': out.get('std_usado')
            })
    # 过滤落入区间
    in_band = [r for r in results if TARGET_MIN <= r['outlier_ratio'] <= TARGET_MAX]
    if in_band:
        # 选择最小 lambda 提供更敏感但仍合理
        rec = sorted(in_band, key=lambda r: (r['lambda_centrada'], r['k']))[0]
        rec['note'] = 'in_target_band'
    else:
        mid = (TARGET_MIN + TARGET_MAX) / 2
        rec = min(results, key=lambda r: abs(r['outlier_ratio'] - mid))
        rec['note'] = 'closest_to_target'
    return {"recommendation": rec, "note": rec['note'], "grid": results}


def analyze_adaptive(values: np.ndarray):
    serie = pd.Series(values).dropna().astype(float)
    if len(serie) < 80:
        return {"recommendation": None, "note": "too_few_points", "grid": []}

    # 默认使用 MA(q=2) 获取残差（简化）
    residuals = MAModel(q=2).get_residuals(serie.values)
    results = []
    for alpha in AV_ALPHA_GRID:
        for quantile in AV_QUANTILE_GRID:
            det = AdaptiveVarianceDetector(alpha=alpha, quantile=quantile,
                                           factor_olvido=0.02, lag_cambio=2,
                                           suavizado=7, change_quantile=0.99)
            out = det.detect(serie.values, residuals)
            labels = out['labels']
            out_ratio = ratio(labels, 'outlier')
            change_ratio = ratio(labels, 'change')
            results.append({
                'alpha': alpha,
                'quantile': quantile,
                'outlier_ratio': out_ratio,
                'change_ratio': change_ratio
            })
    # 优先 outlier 位于区间且 change_ratio <= 1%
    candidates = [r for r in results if TARGET_MIN <= r['outlier_ratio'] <= TARGET_MAX and r['change_ratio'] <= 0.01]
    if candidates:
        rec = sorted(candidates, key=lambda r: (r['alpha'], r['quantile']))[0]
        rec['note'] = 'in_target_band'
    else:
        mid = (TARGET_MIN + TARGET_MAX) / 2
        rec = min(results, key=lambda r: (abs(r['outlier_ratio'] - mid), r['change_ratio']))
        rec['note'] = 'closest_to_target'
    return {"recommendation": rec, "note": rec['note'], "grid": results}


def main():
    parser = argparse.ArgumentParser(description='分析现有数据的阈值与建议（不修改源代码）')
    parser.add_argument('csv', help='输入CSV路径')
    parser.add_argument('--config', required=True, help='现有配置文件路径')
    parser.add_argument('--datetime-col', default='date_time', help='时间列名')
    parser.add_argument('--output-prefix', default='threshold_report', help='输出文件前缀')
    parser.add_argument('--columns', nargs='+', default=None, help='只分析指定列')
    args = parser.parse_args()

    df = safe_read_csv(args.csv, args.datetime_col)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 选择列
    if args.columns:
        candidate_cols = [c for c in args.columns if c in df.columns]
    else:
        candidate_cols = [c for c in config.keys() if c in df.columns]

    summary_rows = []
    detailed = {}

    for col in candidate_cols:
        detector_type = config[col].get('outlier_detector', 'diff')
        values = df[col].values
        if detector_type == 'diff':
            analysis = analyze_diff(values)
        else:
            analysis = analyze_adaptive(values)
        rec = analysis['recommendation']
        summary_rows.append({
            'column': col,
            'detector': detector_type,
            'status': analysis['note'],
            **({} if rec is None else rec)
        })
        detailed[col] = analysis

    ts = datetime.datetime.utcnow().isoformat()
    summary_df = pd.DataFrame(summary_rows)

    out_dir = Path('config_parametros')
    out_dir.mkdir(exist_ok=True)

    summary_path = out_dir / f'{args.output_prefix}_summary.csv'
    detail_path = out_dir / f'{args.output_prefix}_details.json'

    summary_df.to_csv(summary_path, index=False)
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump({"generated_at_utc": ts, "target_band": [TARGET_MIN, TARGET_MAX], "details": detailed}, f, indent=2, ensure_ascii=False)

    print(f'✓ Summary escrito: {summary_path}')
    print(f'✓ Details escrito: {detail_path}')
    print('完成阈值分析，不修改原配置。')

if __name__ == '__main__':
    main()
