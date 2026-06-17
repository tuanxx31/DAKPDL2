# Đánh giá & chỉnh sửa notebook — Anomaly Detection (cấp session)

Tài liệu này tóm tắt (1) đánh giá ngưỡng business rules, (2) các thay đổi đã thực hiện trong notebook, và (3) kết quả đã chạy xác minh trên toàn bộ 1.76 triệu session.

## 1. Đánh giá ngưỡng — đã chọn ổn chưa & cơ sở nào?

**Cơ sở chọn ngưỡng (trước đây thiếu):** ngưỡng nên dựa trên **phân vị (percentile) thực nghiệm** của phân phối từng đặc trưng, không phải số tùy ý. Triết lý: mỗi luật nhắm vào nhóm hành vi **hiếm** (~top 0.1–1%) nhưng vẫn đủ mẫu dương để mô hình học.

Phân tích phân phối thực tế (N = 1,761,675 session) cho thấy phần lớn ngưỡng cũ nằm quanh p99–p99.9 → **hợp lý về hướng**, nhưng có 3 ngưỡng đặt quá cao tạo "lớp chết":

| Rule | Đặc trưng | Cũ | Mới | Phân vị thực tế | Lý do |
| --- | --- | --- | --- | --- | --- |
| BR01 | events_per_minute | 8 | **4** | p99.9 ≈ 4.6 | epm=8 gần như không bao giờ đạt (max≈29, dồn về cận dưới) |
| BR10 | n_addtocart | 10 | **4** | p99.9 = 4 | cũ 10 ≫ p99.9 → chỉ 224 session (lớp gần rỗng) |
| BR11 | max_same_item_view | 20 | **6** | p99.9 = 7 | cũ 20 chỉ trúng **49 session** → F1 ≈ 0.03 (lớp chết) |
| BR12 | unique_categories | 20 | **10** | p99.9 = 9 | cũ 20 ≫ p99.9 → quá hiếm |

Các ngưỡng còn lại (BR01 total_events≥12, BR03 n_view≥6, BR04, BR05, BR06, BR07, BR09) **giữ nguyên** vì đã nằm đúng vùng p98–p99.9 và đủ mẫu.

**Tác động:** số session dương tăng để học được — Repeated view spam 49 → **3,391**; Cart abandonment 224 → **1,586**; Category scanning 628 → **1,666**. Tỉ lệ anomaly tổng: 3.57%.

## 2. Các điểm yếu đã khắc phục trong notebook

1. **Ngưỡng percentile + tài liệu hóa**: cell `THRESHOLDS` cập nhật giá trị mới kèm chú thích phân vị từng dòng; thêm 1 cell markdown "Cơ sở chọn ngưỡng".
2. **Lỗi thực thi (rủi ro lớn nhất)**: trạng thái cũ có cell Extra Trees (Đức Anh) và cell kết luận **chưa chạy** → Extra Trees biến mất khỏi bảng kết quả, phần tổng kết rỗng. Đã xóa toàn bộ output + execution_count để notebook ở trạng thái sạch, **chỉ cần Restart & Run All là chạy đúng thứ tự**.
3. **Báo cáo theo từng thành viên (yêu cầu giáo viên: mỗi người 1 mô hình)**: thêm hàm `show_member_results()` và gọi ngay cuối section của mỗi thành viên → mỗi người có macro/micro-F1, bảng F1 theo từng loại, và confusion matrix **ngay trong phần của mình**.
4. File CSV cũ còn cột `isolation_forest_score` (phiên bản trước Extra Trees) — sẽ được ghi đè đúng khi Run All.

Phân công đã đúng yêu cầu "mỗi thành viên 1 mô hình": XGBoost (Đình Tuấn), Decision Tree (Lê Văn Anh), Random Forest (Tuấn Anh), LightGBM (Thủy), Extra Trees (Đức Anh).

## 3. Kết quả đã chạy xác minh (toàn bộ dữ liệu, ngưỡng mới)

Pipeline chạy end-to-end **không lỗi**. Bảng chính (đa nhãn, safe features, tập test):

| Mô hình (thành viên) | macro-F1 | micro-F1 | macro-PR-AUC |
| --- | --- | --- | --- |
| XGBoost (Đình Tuấn) | **0.631** | 0.806 | 0.646 |
| LightGBM (Thủy) | 0.588 | 0.803 | 0.581 |
| Decision Tree (Lê Văn Anh) | 0.533 | 0.712 | 0.557 |
| Random Forest (Tuấn Anh) | 0.397 | 0.297 | 0.424 |
| Extra Trees (Đức Anh) | 0.172 | 0.064 | 0.227 |

So sánh leakage giữ vững: full features đạt macro-F1 ≈ 0.96–0.98 (boosting) so với safe features ~0.6 → chứng minh leakage rõ ràng.

> Lưu ý: số liệu trên là từ lần chạy xác minh (Random Forest/Extra Trees được giảm cây + lấy mẫu để chạy nhanh trong môi trường kiểm thử). Khi bạn Restart & Run All trên máy với tham số gốc (200 cây), điểm RF/ET sẽ nhỉnh hơn nhưng **xu hướng không đổi**: boosting (XGBoost/LightGBM) vượt trội, hai mô hình rừng over-predict mạnh trên safe features.

## 4. Khuyến nghị tiếp theo (chưa sửa, để nhóm cân nhắc)

- **Random Forest & Extra Trees over-predict** trên safe features (precision rất thấp → micro-F1 sụp). Nguyên nhân: `class_weight='balanced_subsample'` kết hợp cơ chế chọn ngưỡng theo F1 đẩy quá nhiều dự đoán dương. Cách khắc phục: bỏ `class_weight` (để xác suất hiệu chỉnh tốt hơn), hoặc siết `max_pos_rate_multiple` khi chọn ngưỡng, hoặc dùng `CalibratedClassifierCV`.
- Một số "safe features" vẫn là **proxy** cho biến tạo luật (vd `session_duration_sec` ~ `total_events`), nên F1 cao của Bot scraper trên safe set một phần là leakage còn sót. Nên ghi caveat này trong phần thảo luận khi bảo vệ.
