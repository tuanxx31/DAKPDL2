# BÁO CÁO ĐỒ ÁN NHÓM — KHAI PHÁ DỮ LIỆU

## Phát hiện hành vi bất thường theo session trong website thương mại điện tử

- Môn học: Khai phá dữ liệu
- Dataset: RetailRocket E-commerce events
- File notebook: `anomaly_detection_group_project.ipynb`
- Ngày lập báo cáo: 16/06/2026

### Thông tin nhóm

Phân công thuật toán:

| Thành viên                            | Thuật toán      | Loại             | Vai trò                                     |
| --------------------------------------- | ----------------- | ----------------- | -------------------------------------------- |
| **Đình Tuấn (nhóm trưởng)** | **XGBoost** | Có giám sát    | **Mô hình chính của đồ án**     |
| Lê Văn Anh                            | Decision Tree     | Có giám sát    | Dễ giải thích bằng cây quyết định    |
| Tuấn Anh                               | Random Forest     | Có giám sát    | Ensemble ổn định, có feature importance  |
| Thủy                                   | LightGBM          | Có giám sát    | Gradient boosting, đối chiếu với XGBoost |
| Đức Anh                               | Isolation Forest  | Không giám sát | Phát hiện điểm dị biệt, tham chiếu    |

> ⚠️ **Lưu ý về số liệu:** Các ô đánh dấu `___` là metric định lượng cần điền **sau khi chạy lại notebook** với cách chia dữ liệu mới (group split theo `visitorid`). Số liệu cũ (chia ngẫu nhiên) không còn dùng làm kết luận.

## 1. Mục tiêu bài toán

Mục tiêu của đồ án là phát hiện các session có hành vi đáng nghi trên website thương mại điện tử. Do dataset không có nhãn anomaly thật, nhóm xây dựng hệ thống business rules để gán pseudo-label, sau đó dùng các thuật toán khai phá dữ liệu để học và đánh giá mức độ phù hợp với pseudo-label.

Đơn vị phân tích chính là `session_id` (tách session bằng khoảng nghỉ 30 phút), vì hành vi bất thường trong TMĐT thường xảy ra gọn trong một phiên truy cập ngắn.

## 2. Dữ liệu và phạm vi phân tích

- Tổng số sự kiện sau tiền xử lý: 2.755.641
- Số visitor duy nhất: 1.407.580
- Số session profile: 1.761.675
- Tỉ lệ session bất thường theo business rules: 3,51%

## 3. Quy trình xử lý dữ liệu

1. Đọc `events.csv`, xóa dòng trùng lặp, giữ các event hợp lệ (`view`, `addtocart`, `transaction`).
2. Chuyển `timestamp` sang `datetime`, sắp xếp theo `visitorid`, `timestamp`.
3. Tạo session: mở session mới khi là event đầu của visitor hoặc khoảng cách với event trước lớn hơn 30 phút.
4. Gom nhóm theo `session_id` để tạo session behavior profile.
5. Gán nhãn bất thường bằng 12 business rules (multi-label).
6. Chia dữ liệu theo nhóm `visitorid`, train các mô hình, đánh giá và xuất báo cáo.

## 4. Đặc trưng chính (session behavior profile)

- Tần suất hành vi: `total_events`, `n_view`, `n_addtocart`, `n_transaction`.
- Độ đa dạng: `unique_items`, `unique_categories`, `unique_parent_categories`.
- Thời gian & nhịp: `session_duration_sec`, `events_per_minute`, `std_interval_sec`, `max_interval_sec`, `min_interval_sec`, `hour_entropy`.
- Tốc độ thao tác: `rapid_fire_count`, `rapid_ratio`.
- Hành vi lặp: `max_same_item_view`, `max_same_item_atc`.
- Tỉ lệ hành vi: `view_rate`, `atc_rate`, `buy_rate`, `night_ratio`.

## 5. Hệ thống business rules

| Mã  | Loại bất thường | Ý nghĩa                                                 |
| ---- | ------------------- | --------------------------------------------------------- |
| BR01 | Bot scraper         | Session có nhiều event hoặc tốc độ thao tác cao    |
| BR02 | Ghost buyer         | Có transaction nhưng không có addtocart               |
| BR03 | Click fraud         | View nhiều nhưng không addtocart/transaction           |
| BR04 | Rapid-fire          | Nhiều thao tác xảy ra quá nhanh                       |
| BR05 | Night crawler       | Session chủ yếu hoạt động trong khung 0h–5h         |
| BR06 | Item hoarding       | Thêm cùng một item vào giỏ nhiều lần               |
| BR07 | Session bomb        | Xem/quét rất nhiều item khác nhau trong session       |
| BR08 | Sequence violation  | Transaction xuất hiện trước view/addtocart cùng item |
| BR09 | Transaction burst   | Có nhiều transaction trong một session                 |
| BR10 | Cart abandonment    | Addtocart nhiều nhưng không transaction                |
| BR11 | Repeated view spam  | Xem lặp lại cùng một item quá nhiều lần            |
| BR12 | Category scanning   | Quét nhiều danh mục sản phẩm                         |

## 6. Chuẩn bị dữ liệu cho mô hình

- Mẫu supervised: 300.000 session (lấy mẫu phân tầng từ toàn bộ).
- **Chia dữ liệu theo nhóm `visitorid` (GroupShuffleSplit):** đảm bảo các session của cùng một visitor không xuất hiện ở nhiều tập, tránh rò rỉ ngầm ở mức người dùng.
- Tỉ lệ Train / Validation / Test ≈ 60 / 20 / 20 (khoảng 180.000 / 60.000 / 60.000 session).
- Đã kiểm tra tự động: **0 visitor bị trùng** giữa train, validation và test.
- Số đặc trưng: `safe_features` = 15 (bảng metric chính), `full_features` = 37 (chỉ minh họa rule-mimic/leakage).

## 7. Kiểm soát chất lượng mô hình

Đồ án áp dụng ba lớp kiểm soát để kết quả đáng tin cậy:

1. **Tách safe_features và full_features.** `safe_features` loại bỏ các đặc trưng trực tiếp tạo nên luật, dùng làm kết quả chính; `full_features` chỉ để minh họa hiện tượng mô hình "học lại luật".
2. **Group split theo visitorid** (mục 6) để tránh rò rỉ ở mức người dùng.
3. **Shuffle-label sanity check:** train với nhãn bị xáo trộn để xác nhận pipeline không có lỗi nghiêm trọng (điểm phải tụt mạnh).

## 8. Kết quả định lượng (tập test, `safe_features`)

| Mô hình                  | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
| -------------------------- | --------: | -----: | -------: | ------: | -----: |
| Dummy baseline             |       ___ |    ___ |      ___ |     ___ |    ___ |
| Decision Tree              |       ___ |    ___ |      ___ |     ___ |    ___ |
| Random Forest              |       ___ |    ___ |      ___ |     ___ |    ___ |
| **XGBoost**          |       ___ |    ___ |      ___ |     ___ |    ___ |
| LightGBM                   |       ___ |    ___ |      ___ |     ___ |    ___ |
| Isolation Forest (overlap) |       ___ |    ___ |      ___ |     ___ |    ___ |

Nhận xét (điền sau khi có số liệu):

- So sánh nhóm Gradient Boosting (XGBoost, LightGBM) với cây đơn (Decision Tree) và ensemble (Random Forest).
- Dummy baseline phải rất thấp để xác nhận bài toán có ý nghĩa học máy.
- Isolation Forest giữ vai trò tham chiếu unsupervised, không cạnh tranh trực tiếp với supervised.

## 9. Phân tích riêng XGBoost — phần của nhóm trưởng (Đình Tuấn)

XGBoost là mô hình chính của đồ án, do nhóm trưởng phụ trách.

**Cấu hình chính:**

- `objective='binary:logistic'`, `tree_method='hist'`
- `n_estimators=120`, `max_depth=4`, `learning_rate=0.08`
- `subsample=0.85`, `colsample_bytree=0.85`
- `scale_pos_weight = neg/pos` để xử lý mất cân bằng lớp

**Chọn ngưỡng dự đoán:** tối ưu F1 trên tập validation, sau đó áp dụng cố định lên test (không dò ngưỡng trên test).

- Ngưỡng đã chọn: `threshold = ___`

**Kết quả:**

| Tập                           | Precision | Recall | F1-score |
| ------------------------------ | --------: | -----: | -------: |
| Validation (`safe_features`) |       ___ |    ___ |      ___ |
| Test (`safe_features`)       |       ___ |    ___ |      ___ |

Diễn giải (điền sau khi có số liệu):

- Recall thể hiện khả năng bắt được session bất thường theo pseudo-label.
- Precision thể hiện mức cảnh báo sai còn tồn tại.
- Khoảng cách validation–test nhỏ cho thấy mô hình ổn định, chưa overfit rõ rệt.

**Top feature importance của XGBoost (`safe_features`):**

| Đặc trưng             | Importance |
| ------------------------ | ---------: |
| `std_interval_sec`     |        ___ |
| `max_interval_sec`     |        ___ |
| `session_duration_sec` |        ___ |
| `peak_ratio`           |        ___ |
| `unique_event_types`   |        ___ |

Ý nghĩa nghiệp vụ: mô hình học mạnh từ mẫu thời gian và nhịp hành vi trong session (độ lệch khoảng cách giữa các event, khoảng cách tối đa, độ dài session), phù hợp logic phát hiện bot / rapid-fire / session bất thường.

## 10. Kiểm soát leakage và sanity check

**Bảng phụ `full_features`** (có chứa feature tạo rule) — dùng để minh họa rule-mimic do leakage, **không** dùng để kết luận hiệu năng triển khai:

- XGBoost test (`full_features`): F1 = ___, ROC-AUC = ___, PR-AUC = ___

**Shuffle-label sanity check** (train với nhãn xáo trộn ngẫu nhiên):

- XGBoost shuffled-label test: F1 = ___, PR-AUC = ___
- Kỳ vọng: điểm giảm mạnh, xác nhận pipeline đánh giá hợp lý.

## 11. Cơ cấu anomaly type theo business rules

| Anomaly type       | Số session | Tỉ lệ |
| ------------------ | ----------: | ------: |
| Click fraud        |      31.213 |   1,77% |
| Night crawler      |      30.130 |   1,71% |
| Bot scraper        |      10.884 |   0,62% |
| Rapid-fire         |       8.052 |   0,46% |
| Session bomb       |       5.112 |   0,29% |
| Item hoarding      |       3.212 |   0,18% |
| Ghost buyer        |       2.365 |   0,13% |
| Sequence violation |       1.573 |   0,09% |
| Transaction burst  |       1.558 |   0,09% |
| Category scanning  |         628 |   0,04% |
| Cart abandonment   |         224 |   0,01% |
| Repeated view spam |          49 |  0,003% |

Nhóm anomaly tập trung vào: luồng thao tác nhiều/nhanh bất thường, mẫu hành vi xem nhiều ít chuyển đổi, và hoạt động lệch khung giờ bình thường.

## 12. Kết luận

- Đồ án xây dựng được quy trình anomaly detection session-level hoàn chỉnh: Rule → Model → Đánh giá → Export report.
- Điểm mạnh về phương pháp: tách `safe_features`, **chia dữ liệu theo nhóm visitorid** để tránh rò rỉ, threshold tuning trên validation, và shuffle-label sanity check.
- XGBoost là mô hình chính của nhóm; kết quả định lượng cuối cùng sẽ được cập nhật sau khi chạy lại với group split.
- Hạn chế cần nêu rõ: nhãn là pseudo-label sinh từ business rules, nên metric phản ánh mức độ "khớp với luật" hơn là phát hiện bất thường thật. Đây là giới hạn nền tảng của bài toán khi không có nhãn thật.

## 13. Đề xuất cải tiến

1. Đánh giá theo thời gian (time-based split) để mô phỏng production drift.
2. Calibration cho score (Platt / Isotonic) trước khi đặt ngưỡng cảnh báo.
3. Tối ưu ngưỡng theo mục tiêu nghiệp vụ (ưu tiên recall hay precision tùy chi phí false alert).
4. Nếu có thể, thu thập nhãn anomaly thật để đánh giá ngoài pseudo-label.

## 14. Tệp đầu ra kèm theo

- `group_model_metrics.csv`
- `group_leakage_audit.csv`
- `group_anomaly_type_breakdown.csv`
- `session_anomaly_tags.csv`
- `session_anomaly_report.csv`
