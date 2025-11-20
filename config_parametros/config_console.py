import json
import argparse
from pathlib import Path
import shutil

# 简单交互控制台：查看、修改并保存新的 config，不改原文件
# 使用示例:
# python analysis/config_console.py --config config.json --save-as config_edit.json

ACTIONS = ['list', 'show', 'set', 'bulk_set', 'copy', 'save', 'exit', 'help']

HELP_TEXT = {
    'list': '列出所有可编辑列名',
    'show': '显示某列当前配置: show <col>',
    'set': '修改某列单一参数: set <col> <param_path> <value> (param_path 例如 ts_params.q 或 outlier_params.lambda_centrada)',
    'bulk_set': '批量对多列设置同一参数: bulk_set <param_path> <value> <col1> <col2> ...',
    'copy': '复制一个列的整个配置到另一列: copy <src_col> <dst_col>',
    'save': '保存当前工作副本',
    'exit': '退出控制台 (不会自动保存)',
    'help': '显示此帮助或某命令帮助: help [command]'
}

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def deep_set(d: dict, path: str, value):
    parts = path.split('.')
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f'路径不存在或非字典: {p}')
        cur = cur[p]
    last = parts[-1]
    if last not in cur:
        raise KeyError(f'最终键不存在: {last}')
    # 尝试类型保持
    orig = cur[last]
    if isinstance(orig, int):
        try:
            value = int(value)
        except ValueError:
            pass
    elif isinstance(orig, float):
        try:
            value = float(value)
        except ValueError:
            pass
    cur[last] = value


def main():
    parser = argparse.ArgumentParser(description='交互式配置编辑控制台')
    parser.add_argument('--config', required=True, help='原始配置文件')
    parser.add_argument('--save-as', default='config_edited.json', help='保存目标文件')
    parser.add_argument('--backup', action='store_true', help='保存前先对原始文件做备份')
    args = parser.parse_args()

    original_path = Path(args.config)
    work = load_config(args.config)
    print(f'已加载配置: {args.config} (列: {len(work)})')
    print('输入 help 查看命令, exit 退出.')

    while True:
        try:
            raw = input('cfg> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n退出.')
            break
        if not raw:
            continue
        parts = raw.split()
        cmd = parts[0]
        if cmd not in ACTIONS:
            print(f'未知命令: {cmd}'); continue
        if cmd == 'exit':
            print('未保存更改 (如需保存请先执行 save)。')
            break
        if cmd == 'help':
            if len(parts) == 1:
                for k,v in HELP_TEXT.items():
                    print(f'{k}: {v}')
            else:
                target = parts[1]
                print(HELP_TEXT.get(target, f'无帮助: {target}'))
            continue
        if cmd == 'list':
            for i,col in enumerate(work.keys(),1):
                print(f'{i}. {col}')
            continue
        if cmd == 'show':
            if len(parts) != 2:
                print('用法: show <col>'); continue
            col = parts[1]
            if col not in work:
                print('列不存在'); continue
            print(json.dumps(work[col], indent=2, ensure_ascii=False))
            continue
        if cmd == 'set':
            if len(parts) < 4:
                print('用法: set <col> <param_path> <value>'); continue
            col, path, value = parts[1], parts[2], ' '.join(parts[3:])
            if col not in work:
                print('列不存在'); continue
            try:
                deep_set(work[col], path, value)
                print(f'已更新 {col}.{path} = {value}')
            except KeyError as e:
                print(f'错误: {e}')
            continue
        if cmd == 'bulk_set':
            if len(parts) < 5:
                print('用法: bulk_set <param_path> <value> <col1> <col2> ...'); continue
            path = parts[1]; value = parts[2]; cols = parts[3:]
            ok = 0
            for c in cols:
                if c not in work: print(f'跳过不存在列: {c}'); continue
                try:
                    deep_set(work[c], path, value); ok += 1
                except KeyError as e:
                    print(f'{c} 失败: {e}')
            print(f'批量更新完成, 成功 {ok}/{len(cols)}')
            continue
        if cmd == 'copy':
            if len(parts) != 3:
                print('用法: copy <src_col> <dst_col>'); continue
            src, dst = parts[1], parts[2]
            if src not in work or dst not in work:
                print('src 或 dst 不存在'); continue
            work[dst] = json.loads(json.dumps(work[src]))
            print(f'已复制 {src} -> {dst}')
            continue
        if cmd == 'save':
            out_path = Path(args.save_as)
            if args.backup and original_path.exists():
                backup_path = original_path.parent / (original_path.stem + '.bak.json')
                shutil.copyfile(str(original_path), str(backup_path))
                print(f'已备份原文件到 {backup_path}')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(work, f, indent=2, ensure_ascii=False)
            print(f'已保存到 {out_path}')
            continue

if __name__ == '__main__':
    main()
