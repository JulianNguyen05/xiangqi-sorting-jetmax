#!/usr/bin/env python3
# Đọc/ghi file cấu hình YAML theo đường dẫn tính từ gốc repo.
import os
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def path(rel):
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def load(rel):
    with open(path(rel)) as fh:
        return yaml.safe_load(fh)


def save(rel, data, header=None):
    p = path(rel)
    with open(p, 'w') as fh:
        if header:
            for line in header.strip().splitlines():
                fh.write('# ' + line + '\n')
        try:
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True, default_flow_style=None)
        except TypeError:                     # PyYAML < 5.1 trên Jetson cũ không có sort_keys
            yaml.safe_dump(data, fh, allow_unicode=True, default_flow_style=None)
    return p


def _merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


LOCAL = 'config/robot.local.yaml'


def robot_cfg():
    """config/robot.yaml (mặc định, đi theo code) + config/robot.local.yaml (số đo trên robot, ghi đè).
    Sửa thông số trên robot thì sửa robot.local.yaml: chép code mới từ laptop sang không làm mất số đã đo."""
    cfg = load('config/robot.yaml')
    if os.path.exists(path(LOCAL)):
        _merge(cfg, load(LOCAL) or {})
    return cfg


def calib(cfg=None):
    cfg = cfg or robot_cfg()
    p = path(cfg.get('calib_file', 'config/calib.yaml'))
    if not os.path.exists(p):
        return None
    return load(p)
