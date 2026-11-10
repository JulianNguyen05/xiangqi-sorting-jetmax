# Yêu cầu 1 - Xây dựng cơ sở dữ liệu quân cờ tướng

> **Yêu cầu của bài thi:** xây dựng cơ sở dữ liệu mới hoặc kết hợp cơ sở dữ liệu có sẵn; dùng công cụ gán nhãn cho ảnh; viết báo cáo tóm tắt.
>
> **Nhóm thực hiện:** kết hợp 4 cơ sở dữ liệu có sẵn thành một bộ thống nhất tên `merged_v1`, phục vụ bài toán phát hiện 14 loại quân cờ tướng.

## 1. Tóm tắt kết quả

| Hạng mục | Giá trị |
|---|---|
| Số ảnh | **6 808** |
| Số nhãn (quân cờ) | **112 938** |
| Số lớp | **14** (7 loại quân x 2 màu) |
| Định dạng | YOLO (Ultralytics): mỗi ảnh một tệp `.txt`, mỗi dòng `lớp x_tâm y_tâm rộng cao`, toạ độ chuẩn hoá 0 - 1 |
| Chia tập | train **5 356** (78,7%) / valid **739** (10,9%) / test **713** (10,5%) |
| Nhóm ảnh trùng đã gom | 4 265 nhóm; không nhóm nào bị tách giữa các tập |

## 2. Nội dung thư mục

```
Nhom2_YeuCau1/
├── README.md               tệp này
├── build_dataset.ipynb     notebook Colab tạo ra cơ sở dữ liệu (đã có output và hình)
├── scripts/
│   ├── check_dataset.py    kiểm tra cơ sở dữ liệu
│   └── prelabel.py         gán nhãn sơ bộ cho ảnh mới
└── Nhom2_YeuCau1/          cơ sở dữ liệu
    ├── train/images, train/labels
    ├── valid/images, valid/labels
    ├── test/images,  test/labels
    └── data.yaml
```

## 3. 14 lớp (đúng thứ tự id trong `data.yaml`)

| id | Lớp | Quân | id | Lớp | Quân |
|---|---|---|---|---|---|
| 0 | black_advisor | Sĩ đen | 7 | red_advisor | Sĩ đỏ |
| 1 | black_bishop | Tượng đen | 8 | red_bishop | Tượng đỏ |
| 2 | black_cannon | Pháo đen | 9 | red_cannon | Pháo đỏ |
| 3 | black_king | Tướng đen | 10 | red_king | Tướng đỏ |
| 4 | black_knight | Mã đen | 11 | red_knight | Mã đỏ |
| 5 | black_pawn | Tốt đen | 12 | red_pawn | Tốt đỏ |
| 6 | black_rook | Xe đen | 13 | red_rook | Xe đỏ |

Các quân là đĩa tròn giống nhau, chỉ khác chữ Hán và màu mực. Lớp nhiều nhất (`red_pawn`, 15 807 quân) gấp 3,6 lần lớp ít nhất (`black_king`, 4 406 quân), đúng với tỉ lệ quân trong một ván cờ (5 Tốt, 1 Tướng mỗi bên).

## 4. Nguồn dữ liệu

Bốn bộ dữ liệu công khai trên Roboflow Universe:

| Tên trong đề tài | Dự án Roboflow | train | valid | test | Tổng |
|---|---|---|---|---|---|
| super | 312s-workspace/xiangqi-super-dataset | 1 837 | 240 | 241 | 2 318 |
| scara | 312s-workspace/xiangqi-scara-yolo26 | 1 335 | 216 | 183 | 1 734 |
| new_chinese_chess | quoc-rnmzr (v1) | 1 000 | 128 | 125 | 1 253 |
| duytranle | duytranle/xiangqi-ubjij (v18) | 1 184 | 155 | 164 | 1 503 |
| **Tổng** | | **5 356** | **739** | **713** | **6 808** |

Bốn bộ khác nhau về bộ quân, ánh sáng, góc chụp và nền bàn, nhờ đó mô hình không học thuộc một kiểu quân hay một kiểu bàn.

## 5. Quy trình xây dựng (5 bước trong `build_dataset.ipynb`)

| Bước | Việc làm | Lý do | Kết quả |
|---|---|---|---|
| 1 | Tải 4 bộ dữ liệu bằng Roboflow API | Lấy ảnh và nhãn gốc | 4 thư mục nguồn |
| 2 | Chuẩn hoá tên lớp về 14 lớp (ví dụ `Black-Guard`, `B_Shi` đều thành `black_advisor`); bỏ ảnh có lớp không xác định | Mỗi bộ đặt tên lớp khác nhau; sai thứ tự id là mô hình gọi nhầm tên toàn bộ quân | Bảng ánh xạ lớp |
| 3 | Đổi mọi kiểu nhãn về hộp bao: polygon (bộ scara) lấy hình chữ nhật nhỏ nhất chứa các đỉnh; pose (bộ duytranle) bỏ phần điểm đặc trưng; kẹp hộp vào khung ảnh | Mô hình phát hiện đối tượng của YOLO học hộp bao | 112 938 hộp bao |
| 4 | Gom ảnh trùng: cùng tên gốc trước `.rf.` hoặc băm cảm nhận pHash gần nhau, nối bằng union-find | Roboflow sinh nhiều bản tăng cường từ một ảnh gốc; nếu chia ngẫu nhiên thì ảnh gần trùng rơi vào cả train và test | 4 265 nhóm |
| 5 | Chia train / valid / test theo nhóm bằng `StratifiedGroupKFold`, giữ tỉ lệ 4 nguồn và 14 lớp | Tập test khách quan | 5 356 / 739 / 713 |

Điểm đáng chú ý: nếu giữ cách chia gốc của 4 bộ thì **1 044 ảnh bị rò rỉ** giữa các tập (riêng bộ scara 45,4% số ảnh). Chia theo nhóm loại bỏ hoàn toàn hiện tượng này.

## 6. Gán nhãn

- **Nhãn gốc** của 4 bộ được các tác giả gán và công bố trên nền tảng Roboflow, theo ba kiểu: hộp bao, đa giác (scara) và hộp bao kèm điểm đặc trưng (duytranle).
- **Nhóm không gán lại từ đầu** mà làm ba việc: chuẩn hoá tên lớp, đổi kiểu nhãn về hộp bao, kiểm tra nhãn sau khi gộp (kiểm tra tự động bằng `check_dataset.py` và kiểm tra bằng mắt trên ảnh lấy ngẫu nhiên, xem hình trong notebook).
- **Gán nhãn ảnh mới** (ví dụ ảnh chụp từ camera robot) theo quy trình có mô hình hỗ trợ:

```bat
:: 1. Mô hình gán nhãn sơ bộ, sinh labels\*.txt và classes.txt
pip install ultralytics
python scripts\prelabel.py D:\anh_moi --weights ..\Nhom2_YeuCau2_Demo\models\best.pt

:: 2. Rà soát và sửa bằng LabelImg
pip install labelImg
labelImg D:\anh_moi D:\anh_moi\classes.txt D:\anh_moi\labels
```

Trong LabelImg chọn định dạng **YOLO**, sửa hộp sai, vẽ thêm quân bị sót, lưu lại. Sau đó chép ảnh vào `train/images` và nhãn vào `train/labels`. Cũng có thể tải ảnh và nhãn sơ bộ lên Roboflow để sửa trực tuyến.

## 7. Tệp `data.yaml`

```yaml
train: train/images
val: valid/images
test: test/images

nc: 14
names:
  0: black_advisor
  1: black_bishop
  2: black_cannon
  3: black_king
  4: black_knight
  5: black_pawn
  6: black_rook
  7: red_advisor
  8: red_bishop
  9: red_cannon
  10: red_king
  11: red_knight
  12: red_pawn
  13: red_rook
```

## 8. Cách kiểm tra cơ sở dữ liệu

```bat
python scripts\check_dataset.py
```

Tập lệnh chỉ dùng thư viện chuẩn của Python. Nó in số ảnh, số tệp nhãn, số quân của từng tập, số quân theo lớp, và liệt kê các dòng nhãn lỗi (sai số cột, lớp ngoài 0 - 13, toạ độ ngoài 0 - 1). Số liệu để đối chiếu: train 5 356, valid 739, test 713 ảnh; tổng 112 938 quân.

## 9. Cách tạo lại từ đầu

Mở `build_dataset.ipynb` trên Google Colab, nhập API key của Roboflow ở ô đầu tiên rồi chạy toàn bộ. Notebook không cần GPU và ghi kết quả ra `MyDrive/datn/merged_v1.zip`.

## 10. Hạn chế

- Cơ sở dữ liệu chưa có ảnh chụp từ camera của robot JetMax (640 x 480, gắn trên tay máy). Đây là khoảng cách miền lớn nhất; hướng khắc phục là chụp thêm ảnh từ robot, gán nhãn theo mục 6 rồi tinh chỉnh mô hình.
- Mất cân bằng lớp ở mức tự nhiên của bộ cờ; lớp ít nhất vẫn trên 4 000 quân nên không cần cân bằng lại.
