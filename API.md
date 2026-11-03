# Hướng dẫn sử dụng thư viện `XiangqiBrain`

**Ảnh bàn cờ (+ thế cờ lượt trước) → nước đi tiếp theo** cho cờ tướng, chạy trên robot JetMax (Jetson Nano, Python 3.6, không cần Internet) và trên laptop / Colab.

```python
from brain.xiangqi_brain import XiangqiBrain

brain = XiangqiBrain()
res = brain.analyze(cv2.imread('ban_co.jpg'), side='w')
print(res['best_move_vn'])        # 'Pháo 2 bình 5'
```

---

## Mục lục
1. [Cài đặt](#1-cài-đặt)
2. [Quy ước dữ liệu](#2-quy-ước-dữ-liệu)
3. [Khởi tạo `XiangqiBrain`](#3-khởi-tạo-xiangqibrain)
4. [`analyze()`: một ảnh → nước tiếp theo](#4-analyze-một-ảnh--nước-tiếp-theo)
5. [`step()` / `confirm()`: robot chơi cả ván](#5-step--confirm-robot-chơi-cả-ván)
6. [`read_board()`: chỉ đọc thế cờ từ ảnh](#6-read_board-chỉ-đọc-thế-cờ-từ-ảnh)
7. [Cấu hình](#7-cấu-hình)
8. [Ví dụ đầy đủ](#8-ví-dụ-đầy-đủ)
9. [API tầng dưới](#9-api-tầng-dưới)
10. [Xử lý lỗi](#10-xử-lý-lỗi)
11. [Hiệu năng và giới hạn](#11-hiệu-năng-và-giới-hạn)
12. [Công cụ dòng lệnh](#12-công-cụ-dòng-lệnh)

---

## 1. Cài đặt

### Trên robot (không có Internet)
Gói `xiangqi_to_robot.zip` được tạo bởi `training/package.ipynb` (Bước 6). Chép sang robot qua NoMachine rồi:

```bash
cd ~ && python3 -m zipfile -e ~/Desktop/xiangqi_to_robot.zip ~/
cd ~/xiangqi && sh install.sh          # lần đầu ~15 phút (build models/best.engine)
```

Xong khi `install.sh` in `CÀI ĐẶT XONG`. Mọi ví dụ bên dưới chạy trong thư mục `~/xiangqi`.

### Trên laptop / Colab
```bash
git clone https://github.com/JulianNguyen05/xiangqi-sorting-jetmax && cd xiangqi-sorting-jetmax
pip install numpy opencv-python pyyaml onnxruntime
# đặt best.onnx vào models/ ; Pikafish: sh tools/build_pikafish.sh x86  (cần Linux)
```

### Thành phần cần có

| Thành phần | Trên robot | Trên laptop | Thiếu thì |
|---|---|---|---|
| `numpy`, `opencv`, `pyyaml` | có sẵn trong JetPack | `pip install` | không chạy được |
| Mô hình YOLO11n | `models/best.engine` (TensorRT, do `install.sh` build) | `models/best.onnx` + `onnxruntime` | không chạy được (trừ khi tự truyền `detector`) |
| Engine Pikafish | `engine/pikafish` + `engine/pikafish.nnue` (trong gói) | `dist/pikafish_x86/` | tự chuyển sang engine Python (yếu hơn nhiều) |

Trong code, thêm thư mục gốc vào `sys.path` nếu chạy script ở chỗ khác:
```python
import sys; sys.path.insert(0, '/home/jetmax/xiangqi')     # đường dẫn gói trên robot
```

---

## 2. Quy ước dữ liệu

| Khái niệm | Dạng | Ví dụ | Ghi chú |
|---|---|---|---|
| Ảnh | `numpy.ndarray` BGR (như `cv2.imread`) | `img = cv2.imread('a.jpg')` | Ảnh camera ROS: dùng `robot.camera.RosCamera.grab()` |
| Ô | chuỗi `cột + hàng` | `'e0'`, `'h2'` | Cột `a`–`i` (trái → phải nhìn từ phía Đỏ), hàng `0`–`9`; **Đỏ ở hàng 0–4** (phía robot) |
| Ô (dạng số) | `(f, r)` | `(4, 0)` = `e0` | `board.geometry.parse_square('e0')`, `square_name(4, 0)` |
| Nước đi | UCI: ô đi + ô đến | `'h2e2'` | Cùng ký hiệu với Pikafish |
| Nước đi (tiếng Việt) | ký hiệu truyền thống | `'Pháo 2 bình 5'`, `'Mã 8 tiến 7'`, `'Xe trước thoái 2'` | Cột đánh số 1–9 từ phải sang trái theo góc nhìn bên đi |
| Bên đi | `'w'` = Đỏ, `'b'` = Đen | | Như FEN của Pikafish |
| Thế cờ | FEN | `rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1` | Liệt kê từ hàng 9 xuống hàng 0 |
| Thế cờ (dạng dict) | `{(f, r): mã}` | `{(4, 0): 'K', ...}` | `board.rules.parse_fen()` / `to_fen()` |

**Mã quân** (chữ HOA = Đỏ, thường = Đen):

| Mã | Quân | Mã | Quân |
|---|---|---|---|
| `K` / `k` | Tướng | `R` / `r` | Xe |
| `A` / `a` | Sĩ | `C` / `c` | Pháo |
| `B` / `b` | Tượng | `N` / `n` | Mã |
| `P` / `p` | Tốt | | |

---

## 3. Khởi tạo `XiangqiBrain`

```python
XiangqiBrain(detector=None, engine=None, verbose=True, **cấu_hình)
```

| Tham số | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `detector` | đối tượng có `.detect(bgr) -> [Piece]` | `None` → YOLO11n theo cấu hình | Thay bằng bộ nhận diện khác (xem [8.4](#84-thay-bộ-nhận-diện-hoặc-engine)) |
| `engine` | đối tượng có `.best_move(...)`, **hoặc** đường dẫn file Pikafish | `None` → theo cấu hình | Đường dẫn: `XiangqiBrain(engine='~/pikafish/pikafish')` |
| `verbose` | `bool` | `True` | In dòng `[brain] ...` khi nạp mô hình / engine |
| `**cấu_hình` | | | Ghi đè bất kỳ khoá nào ở [mục 7](#7-cấu-hình), vd. `conf=0.3, depth=10, robot_side='w'` |

Khởi tạo mất vài giây (nạp TensorRT + khởi động Pikafish). **Tạo một lần, dùng cho cả chương trình.** Đóng engine khi xong:

```python
with XiangqiBrain(depth=10) as brain:
    ...
# hoặc brain.close()
```

---

## 4. `analyze()`: một ảnh → nước tiếp theo

Không giữ trạng thái: mỗi lần gọi là độc lập. Đây là giao diện đúng yêu cầu *"ảnh + thế cờ + lượt hiện tại → nước tiếp theo"*.

```python
res = brain.analyze(bgr, prev_fen=None, side=None, depth=None, movetime_ms=None)
```

| Tham số | Kiểu | Ý nghĩa |
|---|---|---|
| `bgr` | ảnh BGR | Ảnh bàn cờ hiện tại |
| `prev_fen` | `str` hoặc `None` | **Thế cờ đã xác nhận ở lượt trước**, bên đi trong FEN là bên **vừa đi** trên ảnh này. Có tham số này thì thư viện suy ra nước vừa đi và **tự sửa tối đa `max_cost` ô nhận diện sai** |
| `side` | `'w'` / `'b'` | Chỉ dùng khi **không** có `prev_fen`: bên đi tiếp. Mặc định `'w'` |
| `depth` | `int` | Độ sâu engine cho lần gọi này (mặc định theo cấu hình) |
| `movetime_ms` | `int` | Cho engine nghĩ đúng chừng này mili giây (ưu tiên hơn `depth`) |

### Có hay không có `prev_fen`?

| | `analyze(img, side='w')` | `analyze(img, prev_fen=fen_truoc)` |
|---|---|---|
| Dùng khi | Ảnh đầu tiên, hoặc không biết thế trước | Mọi lượt sau lượt đầu |
| Thế cờ dùng để tính | Đúng như nhận diện | Thế trước + nước hợp lệ **khớp ảnh nhất** |
| Ảnh sai 1–2 ô | Chỉ phát hiện được nếu thế cờ vô lý (`invalid`) | **Tự sửa** (`corrected` = số ô đã sửa) |
| Biết đối thủ vừa đi gì | Không | Có (`last_move`) |
| Độ chính xác đo trên tập test (Bước 4) | ảnh → FEN đúng ~78% | suy ra đúng nước ~95%, còn lại báo `retake`, **0% sai mà không biết** |

### Kết quả (dict)

| Khoá | Kiểu | Có khi | Ý nghĩa |
|---|---|---|---|
| `status` | `str` | luôn có | xem bảng dưới |
| `issues` | `list[str]` | luôn có | Lý do khi `status` khác `'ok'` (tiếng Việt) |
| `fen` | `str` | trừ `retake` | **FEN đã kiểm tra / đã sửa**, bên đi = bên đi tiếp. Lưu lại làm `prev_fen` cho lượt sau |
| `board` | `dict {(f, r): mã}` | trừ `retake` | Như `fen`, dạng dict |
| `last_move`, `last_move_vn` | `str` | có `prev_fen` và có người vừa đi | Nước vừa đi: `'h2e2'`, `'Pháo 2 bình 5'` |
| `corrected` | `int` | có `prev_fen` | Số ô nhận diện sai đã được sửa (0 = ảnh đúng hoàn toàn) |
| `best_move` | `str` | `status == 'ok'` | **Nước tiếp theo** dạng UCI, vd. `'b9c7'` |
| `best_move_vn` | `str` | `'ok'` | `'Mã 2 tiến 3'` |
| `best_move_from`, `best_move_to` | `str` | `'ok'` | `'b9'`, `'c7'`: ô tay máy gắp và ô đặt |
| `captures` | `str` / `None` | `'ok'` | Mã quân bị ăn ở ô đến (phải bỏ ra trước khi đặt) |
| `score_cp` | `int` / `None` | `'ok'` | Điểm theo góc nhìn bên đi (1/100 tốt; > 0 là có lợi) |
| `mate` | `int` / `None` | `'ok'` | Chiếu bí sau N nước (âm: bị chiếu bí) |
| `think_ms` | `float` | `'ok'` | Thời gian engine nghĩ |
| `read` | `dict` | luôn có | Kết quả nhận diện thô (xem [mục 6](#6-read_board-chỉ-đọc-thế-cờ-từ-ảnh)) |

### Giá trị `status`

| `status` | Nghĩa | Nên làm gì |
|---|---|---|
| `'ok'` | Có nước tiếp theo | Đi `best_move`; lưu `fen` |
| `'retake'` | Không định vị được bàn / quá ít quân / ảnh lệch quá `max_cost` ô so với mọi nước hợp lệ | Chụp lại (dịch camera, chờ tay người ra khỏi khung hình) |
| `'invalid'` | (Chỉ khi không có `prev_fen`) thế cờ nhận diện được vô lý: 3 pháo, sĩ ra khỏi cung… | Chụp lại; xem `issues` |
| `'game_over'` | Bên đi tiếp hết nước (chiếu bí hoặc hết nước đi — cờ tướng tính là thua) | Kết thúc ván |

> Có `prev_fen` mà ảnh **chưa có ai đi** (giống hệt thế trước): `last_move = None`, vẫn trả nước tiếp theo cho bên trong `prev_fen`.

### Ví dụ

```python
import cv2
from brain.xiangqi_brain import XiangqiBrain

brain = XiangqiBrain()
res = brain.analyze(cv2.imread('luot1.jpg'), side='w')      # lượt đầu: Đỏ đi
if res['status'] == 'ok':
    print('Đỏ nên đi:', res['best_move_vn'], res['best_move_from'], '->', res['best_move_to'])
    fen = res['fen']

# ... người chơi Đỏ đi một nước, chụp ảnh mới ...
# fen ở trên có bên đi = 'w' = bên VỪA đi trên ảnh mới -> truyền thẳng làm prev_fen
res = brain.analyze(cv2.imread('luot2.jpg'), prev_fen=fen)
print('Vừa đi:', res['last_move_vn'], '| đã sửa', res['corrected'], 'ô')
print('Đen nên đi:', res['best_move_vn'])
```

---

## 5. `step()` / `confirm()`: robot chơi cả ván

Thư viện **tự nhớ ván cờ** (thế cờ, lượt đi, lịch sử nước đi cho luật lặp nước của Pikafish). Robot chỉ cần đưa ảnh vào và báo lại khi đã đi xong.

| Hàm | Ý nghĩa |
|---|---|
| `new_game(fen=START, robot_side=None)` | Bắt đầu ván mới (mặc định thế khai cuộc, robot cầm bên `robot_side` trong cấu hình) |
| `step(bgr, depth=None, movetime_ms=None)` | Một lượt nhìn bàn cờ → dict, xem bảng dưới |
| `confirm(move_uci)` | Gọi **sau khi tay máy đã đi xong** nước của robot; trả về ký hiệu tiếng Việt |
| `brain.game.fen` | FEN hiện tại của ván |
| `brain.game.history` | Danh sách nước đã đi (UCI) |
| `brain.game.robot_to_move()` | Đang đến lượt robot? |

### Kết quả của `step()`

| `status` | Nghĩa | Khoá đi kèm |
|---|---|---|
| `'robot_move'` | Người vừa đi (hoặc robot đi trước) → robot cần đi | `human_move`: `{'uci', 'vn', 'corrected'}` (None nếu robot đi trước); `robot_move`: `{'uci', 'vn', 'from', 'to', 'captures', 'score_cp', 'mate', 'think_ms'}` |
| `'waiting'` | Chưa thấy nước đi mới | |
| `'retake'` | Ảnh không chắc chắn | `candidates`: 5 nước gần đúng nhất (để debug) |
| `'game_over'` | Hết ván | `result` (nếu phát hiện lúc đọc ảnh): `{'result': 'checkmate' / 'stalemate', 'loser': 'w' / 'b'}` |

Luôn có `fen` (FEN hiện tại của ván).

```
          ┌──────────── waiting / retake ─────────────┐
          ▼                                           │
   chụp ảnh ──► step(img) ──► robot_move ──► tay máy đi from → to ──► confirm(uci)
                      │                                                    │
                      └── game_over ──► dừng          ◄────────────────────┘
```

---

## 6. `read_board()`: chỉ đọc thế cờ từ ảnh

Không dùng luật cờ, không gọi engine. Dùng để kiểm tra nhận diện hoặc vẽ kết quả.

```python
rd = brain.read_board(bgr)
```

| Khoá | Ý nghĩa |
|---|---|
| `board` | `{(f, r): mã}` hoặc `None` nếu không định vị được bàn |
| `method` | `'board'` (bàn của đề tài, `vision/board_detect.py`) / `'lattice'` (bàn bất kỳ, `vision/lattice.py`) / `None` |
| `pieces` | Danh sách `Piece` mô hình nhận diện được: `.x .y .r` (pixel), `.code` (mã FEN), `.conf` |
| `n_pieces`, `n_on_board` | Số quân nhận diện được / số quân nằm trên giao điểm |

---

## 7. Cấu hình

Thứ tự ưu tiên: **tham số khi khởi tạo** > `config/robot.local.yaml` (số đo riêng trên robot) > `config/robot.yaml` (mục `brain:`) > mặc định trong code.
Trên robot **chỉ sửa `robot.local.yaml`**: chép gói mới sang không làm mất.

```yaml
# config/robot.local.yaml
brain:
  depth: 10
  red_hint: [320, 470]
```

| Khoá | Mặc định | Ý nghĩa | Nguồn |
|---|---|---|---|
| `model` | `models/best.onnx` | Mô hình YOLO; tự dùng `models/best.engine` nếu đã build và khớp | Bước 2, 3 |
| `conf` | `0.25` | Ngưỡng tin cậy của YOLO | Bước 4, mục 9 |
| `iou` | `0.5` | Ngưỡng NMS | |
| `rotations` | `[0]` | Các góc xoay ảnh khi chạy YOLO; `[0, 90, 180, 270]` chính xác hơn chút nhưng chậm gấp 4 | Bước 2 |
| `locate` | `auto` | `board` (bàn của đề tài) / `lattice` (bàn bất kỳ) / `auto` (thử `board` trước) | Bước 4 |
| `red_hint` | `null` | Điểm ảnh `[x, y]` nằm **gần phía quân Đỏ**; `null` = cạnh dưới ảnh | |
| `engine` | `[engine/pikafish, ~/pikafish/pikafish]` | Đường dẫn Pikafish, thử lần lượt | Bước 5 |
| `depth` | `8` | Độ sâu tìm kiếm | đo bằng `tools/bench_engine.py` |
| `movetime_ms` | `null` | Đặt số thì tính theo thời gian thay vì độ sâu | |
| `threads`, `hash_mb` | `2`, `64` | Tài nguyên cho Pikafish (RAM Jetson chỉ 4 GB) | |
| `robot_side` | `b` | Bên robot cầm trong `step()` | |
| `max_cost` | `2` | Số ô nhận diện sai tối đa còn tự sửa; lớn hơn → dễ sửa nhầm | Bước 5, mục 5 |

---

## 8. Ví dụ đầy đủ

### 8.1 Phân tích một ảnh từ dòng lệnh
```bash
python3 tools/brain_demo.py ban_co.jpg --side b
python3 tools/brain_demo.py luot2.jpg --prev "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
```

### 8.2 Theo dõi ván cờ bằng `analyze()` (tự giữ FEN)
```python
import cv2
from brain.xiangqi_brain import XiangqiBrain

brain = XiangqiBrain()
fen = None
for path in ['luot1.jpg', 'luot2.jpg', 'luot3.jpg']:
    img = cv2.imread(path)
    res = brain.analyze(img, prev_fen=fen) if fen else brain.analyze(img, side='w')
    if res['status'] != 'ok':
        print(path, res['status'], res['issues']); continue
    print(path, '| vừa đi:', res['last_move_vn'], '| nên đi:', res['best_move_vn'])
    fen = res['fen']          # thế cờ ĐÃ SỬA LỖI của ảnh này (chưa gồm best_move)
brain.close()
```

### 8.3 Robot JetMax chơi cả ván
```python
import rospy
from robot import config
from robot.camera import RosCamera
from brain.xiangqi_brain import XiangqiBrain

rospy.init_node('xq_play', anonymous=True)
cam = RosCamera(config.robot_cfg())
brain = XiangqiBrain(robot_side='b')          # người cầm Đỏ, đi trước

def pick_and_place(src, dst, captured):
    """Do chương trình điều khiển tay máy viết: ô 'b9' -> toạ độ mm
    (board/geometry.py: BoardGeometry.square_mm + config/calib.yaml) -> robot/motion.py."""
    ...

while not rospy.is_shutdown():
    res = brain.step(cam.grab())
    if res['status'] == 'robot_move':
        m = res['robot_move']
        if res['human_move']:
            print('Người:', res['human_move']['vn'])
        print('Robot:', m['vn'])
        pick_and_place(m['from'], m['to'], m['captures'])
        brain.confirm(m['uci'])
    elif res['status'] == 'game_over':
        print('Hết ván:', res.get('result')); break
    rospy.sleep(1.0)                          # 'waiting' / 'retake': chụp lại sau 1 giây
brain.close()
```

### 8.4 Thay bộ nhận diện hoặc engine
```python
from vision.pieces import Piece
import numpy as np

class MyDetector(object):
    def detect(self, bgr):
        probs = np.zeros(14); probs[10] = 0.9                  # chỉ số lớp theo training/README.md (10 = red_king)
        return [Piece(x=320, y=400, r=15, probs=probs)]

brain = XiangqiBrain(detector=MyDetector(), engine='~/pikafish/pikafish', depth=12)
```

Engine tự viết chỉ cần có `best_move(fen, moves=None, depth=None, movetime_ms=None, nodes=None)` trả về `brain.uci.EngineResult` (xem `brain/simple.py`).

---

## 9. API tầng dưới

Dùng riêng được, không cần ảnh hay mô hình.

### `board.rules` — luật cờ
| Hàm | Ý nghĩa |
|---|---|
| `parse_fen(fen) -> (board, side)` / `to_fen(board, side) -> str` | Đổi FEN ↔ dict |
| `legal_moves(board, side) -> [((f0, r0), (f1, r1))]` | Mọi nước hợp lệ (đã kiểm chứng perft + đối chiếu Pikafish) |
| `make_move(board, move) -> board` | Thế cờ mới sau nước đi |
| `validate(board, side=None) -> [(mã lỗi, ô, mô tả)]` | Thế cờ có thể xảy ra trong ván thật không |
| `infer_move(prev, side, observed) -> dict` | Thế trước + ảnh mới (có thể sai vài ô) → nước vừa đi |
| `game_status(board, side)` | `'ongoing'` / `'checkmate'` / `'stalemate'` |
| `move_to_uci(move)` / `uci_to_move('h2e2')` / `move_to_vn(board, move)` | Đổi ký hiệu nước đi |

```python
from board import rules as R
b, s = R.parse_fen(R.START)
print(len(R.legal_moves(b, s)))                               # 44
print(R.move_to_vn(b, R.uci_to_move('h2e2')))                 # Pháo 2 bình 5
```

### `brain.uci.UciEngine` — Pikafish
```python
from brain.uci import UciEngine
with UciEngine('engine/pikafish', eval_file='engine/pikafish.nnue', threads=2, hash_mb=64) as eng:
    r = eng.best_move(R.START, depth=10)                      # EngineResult(move, ponder, score_cp, mate, depth, nodes, pv, time_ms)
    r = eng.best_move(R.START, moves=['h2e2', 'h9g7'], movetime_ms=1000)
```

### `brain.game.GameTracker` — theo dõi ván ở mức thế cờ (không ảnh)
`observe(board)` → `{'status': 'moved' | 'corrected' | 'no_change' | 'uncertain' | 'game_over', 'uci', 'vn', 'cost', ...}`, `apply(move)`, `suggest(engine, depth=...)`.

### `vision.lattice` — thế cờ từ tâm quân (bàn bất kỳ)
`image_to_board(pts, codes, confs) -> (board, fit, placement)`; `pts`: mảng `(n, 2)` toạ độ pixel tâm quân.

---

## 10. Xử lý lỗi

| Ngoại lệ / dấu hiệu | Nguyên nhân | Cách xử lý |
|---|---|---|
| `RuntimeError: Không nạp được model` | Chưa có `best.engine` và không chạy được ONNX trên robot | `python3 tools/build_engine.py` (hoặc chạy lại `install.sh`) |
| `CẢNH BÁO: ... được build từ model KHÁC` | Đã đổi `best.onnx` nhưng chưa build lại | `python3 tools/build_engine.py` |
| `[brain] CẢNH BÁO: không có Pikafish -> dùng engine Python` | Thiếu `engine/pikafish` hoặc không chạy được | Kiểm tra `engine/`, chạy `sh install.sh` |
| `brain.uci.EngineError: Hết thời gian chờ` | `depth` quá lớn trên Jetson / engine treo | Giảm `depth` hoặc dùng `movetime_ms` |
| `ValueError: Nước ... không hợp lệ` (từ `confirm`) | Gọi `confirm` với nước khác nước đã trả về | Truyền đúng `res['robot_move']['uci']` |
| `status == 'retake'` liên tục | Bàn ra ngoài khung hình, tay người che, quá tối | Xem ảnh `runs/brain_*.jpg` của `brain_demo.py` |

---

## 11. Hiệu năng và giới hạn

| Đo | Giá trị | Ở đâu |
|---|---|---|
| Ảnh → FEN đúng hoàn toàn (tập test, bàn bất kỳ) | ~78% | Bước 4 |
| Có thế trước: suy ra đúng nước / chụp lại / sai mà không biết | ~95% / ~5% / 0% | Bước 4, mục 8 |
| Ảnh → thế cờ (ONNX, CPU Colab) | ~300 ms | Bước 6, mục 4.2 |
| Pikafish độ sâu 8 (CPU Colab) | ~10–40 ms | Bước 5, mục 6.3 |
| Trên Jetson Nano (TensorRT, Pikafish) | *đo bằng `tools/bench_engine.py`, `tools/brain_demo.py`* | giai đoạn 7 |

**Giới hạn**
- Chưa đo trên ảnh camera robot (giai đoạn 7). Số liệu trên là tập test công khai.
- `locate='lattice'` giả sử ô vuông đều, sông rộng 1 ô; thế cờ thưa (< 15 quân) hoặc ảnh chụp xiên mạnh có thể khớp lưới sai. Trên robot dùng `board_detect` (bàn của đề tài) làm chính.
- `board/rules.py` chưa xử lý luật cấm lặp nước / đuổi dai; trong `step()` Pikafish nhận đủ lịch sử nên tự tránh.
- Bàn cờ đối xứng: thư viện dựa vào vị trí quân + `red_hint` để biết phía Đỏ; đặt sai `red_hint` sẽ lật thế cờ.

---

## 12. Công cụ dòng lệnh

| Lệnh | Việc |
|---|---|
| `sh install.sh` | Cài đặt / cập nhật trên robot, chạy toàn bộ tự kiểm tra |
| `python3 tools/brain_demo.py ảnh.jpg [--side b] [--prev FEN]` | Ảnh → FEN → nước đi; lưu ảnh chú thích vào `runs/` |
| `python3 tools/brain_demo.py --camera` | Chơi cả ván bằng camera robot (người đi quân giúp robot) |
| `python3 tools/bench_engine.py [--depths 6 8 10 12]` | Đo tốc độ Pikafish → chọn `depth` |
| `python3 tests/test_rules.py [--engine engine/pikafish]` | Tự kiểm tra luật cờ, khớp lưới, engine |
| `python3 tests/test_brain.py [--engine engine/pikafish]` | Tự kiểm tra thư viện trên ảnh mẫu |
| `python3 tools/pack.py` | (Laptop) đóng gói `dist/xiangqi_to_robot.zip` |
| `py tools\demo_app.py` hoặc `run_demo.bat` | Giao diện demo trên laptop (xem [DEMO_WINDOWS.md](DEMO_WINDOWS.md)) |
| `python3 tools/pack.py --target windows` | Đóng gói `dist/xiangqi_demo_windows.zip` cho laptop Windows |
