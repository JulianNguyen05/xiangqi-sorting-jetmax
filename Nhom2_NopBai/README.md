# Nhóm 2 - Sản phẩm nộp bài thi chuyên đề

**Đề tài:** Xây dựng thư viện nhận dạng thế cờ tướng từ ảnh bằng YOLO11 và gợi ý nước đi cho robot JetMax
**Giảng viên hướng dẫn:** TS. Đinh Đồng Lưỡng
**Nhóm 2:** Nguyễn Hữu Trọng (65133958), Nguyễn Gia Khiêm (65131478), Huỳnh Thanh Trình (65133949)

Thư mục này gom mọi thứ cần nộp và cần chạy được, tổ chức theo đúng hai yêu cầu của bài thi.

## 1. Cấu trúc thư mục

```
Nhom2_NopBai/
├── README.md                      tệp này
├── Nhom2_BaoCaoCuoiKi.docx        báo cáo (in mang theo hôm vấn đáp)
│
├── Nhom2_YeuCau1/                 YÊU CẦU 1 - cơ sở dữ liệu
│   ├── README.md                  cách xây dựng, thống kê, cách kiểm tra
│   ├── build_dataset.ipynb        notebook tạo ra cơ sở dữ liệu (đã có output)
│   ├── scripts/
│   │   ├── check_dataset.py       đếm ảnh, nhãn, phát hiện nhãn lỗi
│   │   └── prelabel.py            gán nhãn sơ bộ ảnh mới để sửa bằng LabelImg / Roboflow
│   └── Nhom2_YeuCau1/             cơ sở dữ liệu định dạng YOLO
│       ├── train/ (images, labels)
│       ├── valid/ (images, labels)
│       ├── test/  (images, labels)
│       └── data.yaml
│
└── Nhom2_YeuCau2_Demo/            YÊU CẦU 2 - mô hình học sâu và ứng dụng
    ├── README.md                  mô hình, cách huấn luyện, kết quả, cách chạy demo
    ├── requirements.txt
    ├── models/                    best.pt, best.onnx (mô hình đã huấn luyện)
    ├── runs/detect/               log hai giai đoạn huấn luyện (args.yaml, results.csv, weights)
    ├── src/                       train_yolo11.py, evaluate.py, predict.py
    ├── notebooks/                 các notebook Colab đã chạy (huấn luyện, robot, ảnh - FEN, luật cờ, đóng gói)
    └── xiangqi/                   thư viện XiangqiBrain + giao diện demo (run_demo.bat)
```

## 2. Yêu cầu của bài thi và nơi đáp ứng

| Yêu cầu | Nội dung nhóm thực hiện | Thư mục | Chương trong báo cáo |
|---|---|---|---|
| **1.** Xây dựng cơ sở dữ liệu mới hoặc kết hợp cơ sở dữ liệu có sẵn | Kết hợp 4 bộ dữ liệu công khai trên Roboflow Universe thành `merged_v1`: 6 808 ảnh, 112 938 quân, 14 lớp | `Nhom2_YeuCau1/` | Chương 4 |
| **1.** Dùng công cụ gán nhãn | Nhãn gốc được gán trên Roboflow; nhóm chuẩn hoá về 14 lớp, đổi polygon / pose về hộp bao, kiểm tra nhãn. Ảnh mới gán nhãn bằng `prelabel.py` + LabelImg | `Nhom2_YeuCau1/scripts/` | Mục 4.2 |
| **1.** Báo cáo tóm tắt việc thực hiện | `Nhom2_YeuCau1/README.md` và Chương 4 | `Nhom2_YeuCau1/` | Chương 4 |
| **2.** Nghiên cứu và áp dụng mô hình học sâu vào bài toán thực tế liên ngành | YOLO11n (CNN phát hiện đối tượng), huấn luyện 2 giai đoạn, mAP@0.5 = 0,964; áp dụng cho robot JetMax chơi cờ tướng (học sâu + robot + luật cờ) | `Nhom2_YeuCau2_Demo/` | Chương 2, 3, 5, 6, 7 |
| **Chung.** Báo cáo và sản phẩm | Báo cáo có tờ bìa, thông tin nhóm; sản phẩm chạy được | cả thư mục | toàn bộ |

## 3. Cách tạo lại thư mục này từ repo

Thư mục được tạo bằng `make_submission.py` đặt ở thư mục gốc của repo `xiangqi-sorting-jetmax`:

```bat
python make_submission.py --dataset D:\tai-ve\merged_v1.zip --runs D:\tai-ve\runs --zip
```

- `--dataset`: tệp `merged_v1.zip` tải từ Google Drive `MyDrive/datn/` (do `build_dataset.ipynb` tạo ra).
- `--runs`: thư mục `runs/` tải từ Google Drive `MyDrive/datn/runs/` (chứa `yolo11n_640` và `yolo11n_640_rot`).
- `--zip`: nén thành `Nhom2_NopBai.zip` để nộp.

Tập lệnh chỉ chép, không sửa gì trong repo, và in ra danh sách mục còn thiếu.

## 4. Chạy nhanh để kiểm tra

```bat
:: Yêu cầu 1 - kiểm tra cơ sở dữ liệu (không cần cài gì)
cd Nhom2_YeuCau1
python scripts\check_dataset.py

:: Yêu cầu 2 - nhận dạng quân cờ trên một ảnh
cd ..\Nhom2_YeuCau2_Demo
pip install -r requirements.txt
python src\predict.py xiangqi\samples\start_position.jpg

:: Yêu cầu 2 - giao diện demo (ảnh bàn cờ - thế cờ - nước đi)
cd xiangqi
run_demo.bat
```

Chi tiết từng phần nằm trong README của mỗi thư mục.
