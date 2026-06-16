# GIẢI THÍCH MÔ HÌNH XGBOOST — Ý TƯỞNG, CODE & KẾT QUẢ

> Tài liệu dành cho phần của nhóm trưởng (Đình Tuấn). Đọc xong là hiểu: (1) ý tưởng vì sao dùng XGBoost, (2) cách chạy, (3) từng dòng code làm gì, (4) cách đọc và đánh giá kết quả đầu ra. Mọi số liệu lấy từ `group_model_metrics.csv`.

---

## 1. Ý TƯỞNG

### XGBoost là gì (nói ngắn gọn để bảo vệ)

XGBoost (Extreme Gradient Boosting) là thuật toán **boosting cây quyết định**: nó xây nhiều cây nhỏ **nối tiếp nhau**, mỗi cây mới học để sửa lỗi còn lại của các cây trước. Tổng hợp nhiều cây "yếu" thành một mô hình "mạnh". Đầu ra là một **xác suất** từ 0 đến 1 cho biết một session khả nghi đến mức nào.

### Vì sao chọn XGBoost cho bài toán này

- **Dữ liệu dạng bảng (tabular):** mỗi session là một hàng với ~15–37 đặc trưng số — đúng "sân nhà" của cây boosting, thường mạnh hơn deep learning trên dữ liệu bảng.
- **Quan hệ phi tuyến & tương tác đặc trưng:** hành vi bất thường không tuyến tính (ví dụ "nhanh + nhiều + nửa đêm" mới đáng nghi); cây tự học các tương tác này.
- **Chịu được mất cân bằng:** có tham số `scale_pos_weight` để tăng trọng số cho lớp hiếm (~3,5% anomaly).
- **Chống overfit:** có regularization, giới hạn độ sâu cây, lấy mẫu ngẫu nhiên dòng/cột.
- **Giải thích được:** cho `feature_importance` để biết đặc trưng nào quan trọng — phục vụ bảo vệ.

### Vai trò trong đồ án

XGBoost là **mô hình chính** (main supervised). Nó được huấn luyện trên `safe_features` (15 đặc trưng đã loại biến tạo luật) để kết quả không bị "chép luật". Các mô hình khác (LightGBM, Random Forest, Decision Tree, Isolation Forest) dùng để đối chiếu.

### Đặc trưng đầu vào: bao nhiêu, gồm những gì, dùng gì để train

- Mỗi session sinh ra **37** đặc trưng số (`full_feature_cols`).
- XGBoost (và mọi mô hình chính) **chỉ train trên 15** đặc trưng an toàn (`safe_feature_cols`).
- 22 đặc trưng còn lại (biến đếm trực tiếp tạo luật) bị loại để tránh label leakage; chúng chỉ xuất hiện trong bảng phụ `full_features`.

**15 đặc trưng XGBoost dùng để train:** `session_duration_sec`, `active_hours`, `unique_event_types`, `event_type_entropy`, `hour_entropy`, `mean_interval_sec`, `median_interval_sec`, `std_interval_sec`, `max_interval_sec`, `peak_events`, `weekend_events`, `peak_ratio`, `weekend_ratio`, `unique_parent_categories`, `session_dayofweek`.

---

## 2. CÁCH DÙNG (CHẠY NHƯ THẾ NÀO)

**Yêu cầu thư viện:**

```bash
.venv/bin/python -m pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn
# macOS có thể cần: brew install libomp   (runtime OpenMP cho XGBoost/LightGBM)
```

**Chạy:** mở `anomaly_detection_group_project.ipynb`, chạy tuần tự từ trên xuống. Thứ tự các cell liên quan XGBoost:

1. Cell cấu hình (import, hằng số `RANDOM_STATE`, đường dẫn).
2. Cell định nghĩa `safe_feature_cols` / `full_feature_cols` (chống leakage).
3. Cell chia dữ liệu theo nhóm visitor (train/val/test).
4. Cell định nghĩa `choose_best_threshold`, `record_evaluation` và `train_and_evaluate_supervised_model` (hàm dùng chung).
5. **Cell huấn luyện XGBoost** (phần chính bên dưới).
6. Cell tổng hợp metric → xuất `group_model_metrics.csv`.

**Đầu ra:** điểm số `xgboost_score` cho từng session, bảng metric, và biểu đồ feature importance / confusion matrix.

---

## 3. GIẢI THÍCH TỪNG DÒNG CODE

### 3.1. Hàm tạo mô hình XGBoost

```python
def make_xgb_model(n_estimators=120, scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
```

Giải thích từng tham số:

- `def make_xgb_model(...)` — gói cấu hình vào một hàm để tái dùng (dùng lại y hệt cho bản `full_features` và bản shuffle-label, đảm bảo công bằng khi so sánh).
- `n_estimators=120` — số cây boosting (120 cây nối tiếp). Đủ để học, không quá nhiều gây overfit/chậm.
- `max_depth=4` — mỗi cây sâu tối đa 4 tầng. **Cây nông** → mỗi cây chỉ học mẫu đơn giản, chống overfit; sức mạnh đến từ việc cộng nhiều cây.
- `learning_rate=0.08` — tốc độ học (mỗi cây chỉ đóng góp 8% chỉnh sửa). Học **chậm mà chắc**, kết hợp với 120 cây cho kết quả ổn định.
- `subsample=0.85` — mỗi cây chỉ dùng ngẫu nhiên 85% **số dòng (session)** để huấn luyện → tăng tính tổng quát, giảm overfit.
- `colsample_bytree=0.85` — mỗi cây chỉ dùng ngẫu nhiên 85% **số cột (đặc trưng)** → tương tự, tránh phụ thuộc quá mức vào vài đặc trưng.
- `objective='binary:logistic'` — bài toán phân loại nhị phân (bình thường / bất thường), đầu ra là xác suất.
- `eval_metric='logloss'` — hàm đo lỗi trong lúc train (log loss, phù hợp xác suất).
- `tree_method='hist'` — thuật toán dựng cây dạng histogram, **nhanh** và tiết kiệm bộ nhớ trên dữ liệu lớn.
- `random_state=RANDOM_STATE` — cố định hạt ngẫu nhiên (=10) để **kết quả tái lập được** mỗi lần chạy.
- `n_jobs=-1` — dùng tất cả nhân CPU để train song song.
- `scale_pos_weight=scale_pos_weight` — **trọng số bù mất cân bằng**: phạt nặng hơn khi bỏ sót anomaly (xem 3.2).

### 3.2. Tính trọng số mất cân bằng

```python
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_pos_weight = neg_count / max(pos_count, 1)
```

- `neg_count` — đếm số session **bình thường** (nhãn 0) trong tập train.
- `pos_count` — đếm số session **bất thường** (nhãn 1) trong tập train.
- `scale_pos_weight = neg/pos` — tỉ lệ âm/dương (≈ 27 vì anomaly chỉ ~3,5%). Đưa vào XGBoost nghĩa là: "một lỗi ở lớp bất thường đáng giá bằng ~27 lỗi ở lớp bình thường" → mô hình **chú ý bắt anomaly** thay vì bỏ qua. `max(pos_count, 1)` để tránh chia cho 0.

### 3.3. Huấn luyện mô hình

```python
xgb_safe_model = make_xgb_model(n_estimators=120, scale_pos_weight=scale_pos_weight)
xgb_safe_model.fit(X_safe_train, y_train)
```

- Dòng 1 — tạo mô hình với cấu hình trên + trọng số mất cân bằng vừa tính.
- Dòng 2 — `.fit(...)` huấn luyện trên **tập train**, dùng `X_safe_train` (15 đặc trưng an toàn) và nhãn `y_train`. Chỉ học trên train, **không** đụng tới validation/test.

### 3.4. Chọn ngưỡng quyết định trên VALIDATION

```python
xgb_safe_val_score = xgb_safe_model.predict_proba(X_safe_val)[:, 1]
xgb_safe_threshold = choose_best_threshold(y_val, xgb_safe_val_score, 'XGBoost safe_features')
xgb_safe_val_pred = (xgb_safe_val_score >= xgb_safe_threshold).astype(int)
```

- `predict_proba(X_safe_val)[:, 1]` — lấy **xác suất bất thường** (cột thứ 2) cho từng session trong tập validation. Kết quả là số thực 0–1.
- `choose_best_threshold(...)` — quét nhiều ngưỡng và chọn ngưỡng cho **F1 cao nhất trên validation**. Vì sao không mặc định 0,5? Vì dữ liệu mất cân bằng, ngưỡng tối ưu thường khác 0,5 (ở đây = **0,869**). Chọn trên validation chứ **không** trên test để giữ test khách quan.
- `(score >= threshold).astype(int)` — quy đổi xác suất thành nhãn 0/1: vượt ngưỡng thì gắn cờ bất thường (1).

> Hàm `choose_best_threshold` bên trong: tạo danh sách ngưỡng ứng viên từ các phân vị của điểm số, với mỗi ngưỡng tính F1, rồi giữ ngưỡng cho F1 tốt nhất (hòa thì ưu tiên precision cao hơn, rồi recall). Trả về một con số ngưỡng duy nhất.

### 3.5. Đánh giá trên TEST (số báo cáo cuối)

```python
xgb_safe_test_score = xgb_safe_model.predict_proba(X_safe_test)[:, 1]
xgb_safe_test_pred = (xgb_safe_test_score >= xgb_safe_threshold).astype(int)
add_evaluation('XGBoost', 'safe_features', 'Main supervised ...', 'test',
               y_test, xgb_safe_test_pred, xgb_safe_test_score,
               threshold=xgb_safe_threshold)
```

- Tính xác suất cho **tập test**.
- **Dùng lại đúng ngưỡng 0,869 đã chốt ở validation** (không dò lại trên test) → đánh giá trung thực.
- `add_evaluation(...)` — hàm dùng chung: tính accuracy, precision, recall, F1, ROC-AUC, PR-AUC, lưu vào bảng kết quả và in `classification_report`.

### 3.6. Gán điểm cho toàn bộ dữ liệu (để xuất báo cáo)

```python
anomalies['xgboost_score'] = xgb_safe_model.predict_proba(X_safe)[:, 1]
anomalies['xgboost_pred']  = (anomalies['xgboost_score'] >= xgb_safe_threshold).astype(int)
```

- Chấm điểm cho **tất cả 1,76 triệu session** (không chỉ test), thêm 2 cột: `xgboost_score` (xác suất) và `xgboost_pred` (cờ 0/1 theo ngưỡng 0,869). Hai cột này dùng cho file `session_anomaly_report.csv` và bỏ phiếu tổng hợp giữa các mô hình.

### 3.7. Các bước nền (để hiểu code chạy được)

**Chia dữ liệu theo nhóm visitor (chống rò rỉ):**

```python
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
trainval_pos, test_pos = next(gss_test.split(supervised_indices, y_sup, groups_sup))
```

- `GroupShuffleSplit` chia dữ liệu theo **nhóm** (`groups_sup` = `visitorid`), đảm bảo **mọi session của một visitor nằm trọn trong một tập** — không lọt vừa train vừa test. Sau đó chia tiếp validation từ phần còn lại (60/20/20). Cuối cell có `assert` kiểm tra 0 visitor trùng giữa các tập.

**Lấy mẫu phân tầng:**

```python
supervised_indices, _ = train_test_split(all_indices, train_size=MAX_SUPERVISED_ROWS,
                                          random_state=RANDOM_STATE, stratify=y)
```

- Rút 300.000 session để train cho nhẹ, `stratify=y` giữ đúng tỉ lệ ~3,5% anomaly.

---

## 4. ĐÁNH GIÁ KẾT QUẢ ĐẦU RA

### 4.1. Vì sao không nhìn accuracy

Dữ liệu mất cân bằng (~3,5%): đoán "tất cả bình thường" đã ~96,5% accuracy nhưng vô dụng. Vì vậy nhìn **Precision, Recall, F1, PR-AUC**.

- **Precision** (độ chính xác cảnh báo): trong số session bị gắn cờ, bao nhiêu % thực sự bất thường → đo mức **báo động sai**.
- **Recall** (độ phủ): trong số session bất thường thật, bắt được bao nhiêu % → đo mức **bỏ sót**.
- **F1**: trung bình điều hòa của precision và recall.
- **PR-AUC**: diện tích dưới đường precision–recall, **chỉ số đáng tin nhất cho lớp hiếm**.

### 4.2. Kết quả XGBoost (`safe_features`)

| Tập | Precision | Recall | F1 | ROC-AUC | PR-AUC | Ngưỡng |
|---|--:|--:|--:|--:|--:|--:|
| Validation | 0,814 | 0,929 | 0,868 | 0,994 | 0,928 | 0,869 |
| **Test** | **0,812** | **0,918** | **0,862** | **0,993** | **0,922** | 0,869 |

**Đọc kết quả:**

- **Recall 0,918** — bắt được ~92% session bất thường (theo nhãn luật). Tốt cho bài toán phát hiện, vì bỏ sót anomaly nguy hiểm hơn báo nhầm.
- **Precision 0,812** — trong các session bị gắn cờ, ~81% là bất thường thật, ~19% báo nhầm. Chấp nhận được cho lớp sàng lọc.
- **F1 0,862**, **PR-AUC 0,922** — cao và cân bằng.
- **Khoảng cách validation–test rất nhỏ** (F1 0,868 → 0,862) → mô hình **ổn định, chưa overfit rõ rệt**.

### 4.3. Ma trận nhầm lẫn (tập test, 60.000 session)

Suy ra từ các chỉ số trên (support anomaly = 2.116, dự đoán anomaly = 2.392):

|  | Dự đoán: Bình thường | Dự đoán: Bất thường |
|---|--:|--:|
| **Thật: Bình thường** | TN ≈ 57.434 | FP ≈ 450 |
| **Thật: Bất thường** | FN ≈ 174 | TP ≈ 1.942 |

- **TP 1.942 / FN 174:** bắt đúng 1.942, bỏ sót 174 (recall 0,918).
- **FP 450:** báo nhầm 450 session bình thường (giá phải trả của precision 0,812).

### 4.4. So với các mô hình khác (test, `safe_features`)

| Mô hình | Precision | Recall | F1 | PR-AUC |
|---|--:|--:|--:|--:|
| Dummy baseline | 0,041 | 0,040 | 0,040 | 0,035 |
| Decision Tree | 0,684 | 0,775 | 0,727 | 0,747 |
| Random Forest | 0,799 | 0,911 | 0,851 | 0,902 |
| **XGBoost (chính)** | 0,812 | 0,918 | **0,862** | 0,922 |
| **LightGBM** | 0,820 | 0,934 | **0,873** | 0,935 |
| Isolation Forest | 0,291 | 0,674 | 0,406 | 0,358 |

- **Dummy ~0** → bài toán thật sự có ý nghĩa học máy, không phải đoán mò.
- **XGBoost & LightGBM dẫn đầu**; LightGBM nhỉnh hơn chút (F1 0,873 > 0,862) nhưng cùng họ boosting, kết quả sát nhau củng cố độ tin cậy. XGBoost là mô hình chính theo phân công.
- **Isolation Forest thấp** vì không dùng nhãn khi train — chỉ là tham chiếu unsupervised.

### 4.5. Bằng chứng kết quả KHÔNG bị "ảo"

- **`full_features` (chứa biến tạo luật):** XGBoost test F1 = **0,994**, PR-AUC ≈ 1,0 — gần tuyệt đối. Đây là minh chứng *label leakage* (mô hình chép luật) → **không dùng làm kết luận**, chỉ để so sánh.
- **Shuffle-label sanity check:** train với nhãn xáo trộn ngẫu nhiên, XGBoost test F1 tụt còn **0,209**, PR-AUC 0,092 → đúng kỳ vọng, xác nhận pipeline lành mạnh (không có rò rỉ ẩn).

### 4.6. Đặc trưng quan trọng nhất

Feature importance của XGBoost cho thấy nhóm dẫn đầu là các đặc trưng **thời gian/nhịp thao tác**: `std_interval_sec` (độ lệch khoảng cách giữa các event), `max_interval_sec`, `session_duration_sec`. Hợp lý về nghiệp vụ: bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

---

## 5. TÓM TẮT 30 GIÂY (để nói khi bảo vệ)

"XGBoost là mô hình chính, huấn luyện trên 15 đặc trưng an toàn đã loại biến tạo luật. Em dùng `scale_pos_weight` để xử lý mất cân bằng, cây nông `max_depth=4` chống overfit, và chọn ngưỡng 0,869 tối ưu F1 trên validation rồi cố định áp lên test. Kết quả test: F1 0,862, recall 0,918, PR-AUC 0,922 — bắt được ~92% session bất thường với độ chính xác cảnh báo ~81%. Khoảng cách validation–test rất nhỏ nên mô hình ổn định, và bài kiểm tra shuffle-label (F1 tụt còn 0,21) xác nhận kết quả không bị rò rỉ."
