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


def robot_cfg():
    return load('config/robot.yaml')


def calib(cfg=None):
    cfg = cfg or robot_cfg()
    p = path(cfg.get('calib_file', 'config/calib.yaml'))
    if not os.path.exists(p):
        return None
    return load(p)
