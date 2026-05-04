# Phát hiện hành vi bất thường theo session trong TMĐT

## 1. Mục tiêu đề tài

- Dataset: RetailRocket E-Commerce.
- Bài toán: phát hiện session có hành vi đáng nghi trong website thương mại điện tử.
- Đơn vị phân tích: `session_id`, tạo từ `visitorid` với ngưỡng ngắt phiên 30 phút.
- Nhãn bất thường là pseudo-label từ luật nghiệp vụ, không phải nhãn thật có sẵn trong dataset.

## 2. Quy trình xử lý dữ liệu

1. Đọc `events.csv`, xóa dòng trùng lặp và giữ các event hợp lệ.
2. Chuyển `timestamp` sang `datetime`.
3. Sắp xếp theo `visitorid`, `timestamp`.
4. Tạo session:
   - Nếu là event đầu tiên của visitor thì mở session mới.
   - Nếu khoảng cách với event trước lớn hơn 30 phút thì mở session mới.
5. Gom nhóm theo `session_id` để tạo session behavior profile.

## 3. Đặc trưng chính

- Tần suất hành vi: `total_events`, `n_view`, `n_addtocart`, `n_transaction`.
- Độ đa dạng: `unique_items`, `unique_categories`, `unique_parent_categories`.
- Thời gian: `session_duration_sec`, `events_per_minute`, `hour_entropy`.
- Tốc độ thao tác: `min_interval_sec`, `rapid_fire_count`, `rapid_ratio`.
- Hành vi lặp: `max_same_item_view`, `max_same_item_atc`.
- Tỉ lệ hành vi: `view_rate`, `atc_rate`, `buy_rate`, `night_ratio`.

## 4. Hệ thống luật nghiệp vụ

| Mã | Loại bất thường | Ý nghĩa |
| --- | --- | --- |
| BR01 | Bot scraper | Session có nhiều event hoặc tốc độ thao tác cao. |
| BR02 | Ghost buyer | Có transaction nhưng không có addtocart. |
| BR03 | Click fraud | View nhiều nhưng không addtocart/transaction. |
| BR04 | Rapid-fire | Nhiều thao tác xảy ra quá nhanh. |
| BR05 | Night crawler | Session chủ yếu hoạt động trong khung 0h-5h. |
| BR06 | Item hoarding | Thêm cùng một item vào giỏ nhiều lần. |
| BR07 | Session bomb | Xem/quét nhiều item khác nhau trong session. |
| BR08 | Sequence violation | Transaction xuất hiện trước view/addtocart cùng item. |
| BR09 | Transaction burst | Có nhiều transaction trong một session. |
| BR10 | Cart abandonment | Addtocart nhiều nhưng không transaction. |
| BR11 | Repeated view spam | Xem lặp lại cùng một item quá nhiều lần. |
| BR12 | Category scanning | Quét nhiều danh mục sản phẩm. |

## 5. Kết quả gắn nhãn

- Tổng số session: `1,761,675`.
- Session bất thường theo luật: `61,849`, chiếm `3.51%`.
- File tag chính: `session_anomaly_tags.csv`.
- File report tổng hợp rule + model: `session_anomaly_report.csv`.

Phân bố anomaly type:

| Anomaly type | Số session | Tỉ lệ |
| --- | ---: | ---: |
| Click fraud | 31,213 | 1.77% |
| Night crawler | 30,130 | 1.71% |
| Bot scraper | 10,884 | 0.62% |
| Rapid-fire | 8,052 | 0.46% |
| Session bomb | 5,112 | 0.29% |
| Item hoarding | 3,212 | 0.18% |
| Ghost buyer | 2,365 | 0.13% |
| Sequence violation | 1,573 | 0.09% |
| Transaction burst | 1,558 | 0.09% |
| Category scanning | 628 | 0.04% |
| Cart abandonment | 224 | 0.01% |
| Repeated view spam | 49 | 0.003% |

## 6. Phân công thuật toán

| Người | Thuật toán | Loại | Vai trò |
| --- | --- | --- | --- |
| 1 | XGBoost | Có giám sát | Mô hình chính, hiệu quả tốt nhất. |
| 2 | Decision Tree | Có giám sát | Dễ giải thích bằng cây quyết định. |
| 3 | Random Forest | Có giám sát | Ensemble ổn định, có feature importance. |
| 4 | K-Means | Không giám sát | Khám phá cụm session bất thường. |
| 5 | Isolation Forest | Không giám sát | Phát hiện điểm dị biệt. |

## 7. Kết quả mô hình

Metric chính dùng `safe_features`, đã loại các feature trực tiếp tạo nhãn luật để giảm label leakage.

| Model | Precision | Recall | F1-score | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.5420 | 0.9734 | 0.6963 | 0.9219 |
| Decision Tree | 0.4948 | 0.9720 | 0.6558 | 0.7310 |
| Random Forest | 0.4812 | 0.9748 | 0.6444 | 0.8954 |
| K-Means | 0.3343 | 0.4761 | 0.3928 | 0.3660 |
| Isolation Forest | 0.3261 | 0.4657 | 0.3836 | 0.3555 |

## 8. Đánh giá

- XGBoost có kết quả tốt nhất tổng thể, F1-score và PR-AUC cao nhất.
- Random Forest gần XGBoost, phù hợp để giải thích bằng feature importance.
- Decision Tree thấp hơn nhưng dễ trình bày, phù hợp minh họa logic phân loại.
- K-Means và Isolation Forest thấp hơn vì không dùng nhãn khi huấn luyện; kết quả dùng để đánh giá overlap với pseudo-label.
- Accuracy không nên là metric chính vì dữ liệu mất cân bằng.
- PR-AUC, precision, recall và F1-score quan trọng hơn.

## 9. Lưu ý về label leakage

- Nếu dùng trực tiếp các feature tạo luật như `total_events`, `n_view`, `n_transaction`, `rapid_ratio`, mô hình sẽ học lại luật và metric sẽ quá đẹp.
- Notebook tách:
  - `full_features`: dùng để minh họa rule-mimic/leakage.
  - `safe_features`: dùng cho bảng metric chính.
- Khi báo cáo, chỉ dùng bảng `safe_features` làm kết luận mô hình.

## 10. Kết luận

Phiên bản session-level phù hợp hơn visitor-level vì hành vi bất thường trong TMĐT thường xảy ra trong một phiên truy cập ngắn. Hệ thống rule-based giúp giải thích loại bất thường, còn 5 thuật toán khai phá dữ liệu giúp so sánh khả năng mô hình hóa và phát hiện hành vi đáng nghi.
