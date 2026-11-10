#!/bin/sh
# CÀI ĐẶT TRÊN ROBOT JETMAX — chạy sau mỗi lần chép gói mới sang (tự bỏ qua bước đã làm):
#   cd ~ && python3 -m zipfile -e ~/Desktop/xiangqi_to_robot.zip ~/
#   cd ~/xiangqi && sh install.sh
# Không cần mạng. Không ghi đè config/robot.local.yaml, config/calib.yaml, models/best.engine (nếu khớp model).
cd "$(dirname "$0")" || exit 1
ok()   { echo "  [OK]   $*"; }
bad()  { echo "  [LỖI]  $*"; FAIL=1; }
FAIL=0
echo "== Gói"
[ -f VERSION.txt ] && sed 's/^/  /' VERSION.txt

echo "== Cấu hình riêng của robot"
if [ ! -f config/robot.local.yaml ] && [ -f config/robot.local.example.yaml ]; then
  cp config/robot.local.example.yaml config/robot.local.yaml && ok "tạo config/robot.local.yaml (lần đầu)"
else
  ok "giữ nguyên config/robot.local.yaml"
fi

echo "== Engine cờ tướng (Pikafish)"
if [ -f engine/pikafish ]; then
  chmod +x engine/pikafish                       # python -m zipfile làm mất quyền chạy
  if ( (printf 'uci\nisready\nposition startpos\ngo depth 4\n'; sleep 3; printf 'quit\n') | ./engine/pikafish 2>&1 | grep -q bestmove ); then
    ok "$(printf 'uci\nquit\n' | ./engine/pikafish | grep 'id name' | sed 's/id name //')"
  else
    bad "engine/pikafish không chạy (sai kiến trúc? thiếu engine/pikafish.nnue?)"
  fi
else
  bad "gói không có engine/pikafish -> thư viện sẽ dùng engine Python (yếu)"
fi

echo "== Mô hình YOLO -> TensorRT (lần đầu ~15 phút)"
if [ ! -f models/best.onnx ]; then
  bad "thiếu models/best.onnx"
elif [ -f models/best.engine ] && python3 -c "
import sys; sys.path.insert(0, '.')
from vision.pieces import engine_matches
sys.exit(0 if engine_matches('models/best.onnx', 'models/best.engine') else 1)"; then
  ok "models/best.engine đã có và khớp best.onnx -> bỏ qua"
elif [ -f /etc/nv_tegra_release ]; then
  echo "  build models/best.engine (đừng tắt cửa sổ)..."
  python3 tools/build_engine.py || bad "build engine lỗi -> thử: python3 tools/build_engine.py --fp32"
else
  ok "không phải Jetson -> dùng best.onnx (onnxruntime / OpenCV)"
fi

echo "== Tự kiểm tra"
ENG=""; [ -x engine/pikafish ] && ENG="--engine engine/pikafish"
python3 tests/test_rules.py $ENG > /tmp/xq_test_rules.log 2>&1 \
  && ok "tests/test_rules.py" || { bad "tests/test_rules.py (xem /tmp/xq_test_rules.log)"; tail -5 /tmp/xq_test_rules.log; }
python3 tests/test_brain.py $ENG > /tmp/xq_test_brain.log 2>&1 \
  && ok "tests/test_brain.py" || { bad "tests/test_brain.py (xem /tmp/xq_test_brain.log)"; tail -5 /tmp/xq_test_brain.log; }

echo
if [ "$FAIL" = 0 ]; then
  echo "CÀI ĐẶT XONG. Thử: python3 tools/brain_demo.py samples/start_position.jpg"
  echo "Đo tốc độ engine: python3 tools/bench_engine.py --engine engine/pikafish"
else
  echo "CÓ LỖI ở trên. Xem RUNBOOK.md mục Xử lý sự cố."
fi
exit $FAIL
