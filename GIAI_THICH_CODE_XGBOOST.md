# GIẢI THÍCH MÔ HÌNH XGBOOST — Ý TƯỞNG, CODE & KẾT QUẢ

> Tài liệu dành cho phần của nhóm trưởng (Đình Tuấn). Đọc xong là hiểu: (1) ý tưởng vì sao dùng XGBoost, (2) cách chạy, (3) từng dòng code làm gì, (4) cách đọc và đánh giá kết quả. Bài toán là **phân loại ĐA NHÃN** (multi-label): XGBoost dự đoán mỗi session thuộc loại bất thường nào trong 12 loại. Mọi số liệu lấy từ `group_model_metrics.csv` và `group_multilabel_type_metrics.csv`.

---

## 1. Ý TƯỞNG

### XGBoost là gì (nói ngắn gọn để bảo vệ)

XGBoost (Extreme Gradient Boosting) là thuật toán **boosting cây quyết định**: nó xây nhiều cây nhỏ **nối tiếp nhau**, mỗi cây mới học để sửa lỗi còn lại của các cây trước. Tổng hợp nhiều cây "yếu" thành một mô hình "mạnh". Với mỗi loại bất thường, đầu ra là một **xác suất** từ 0 đến 1 cho biết session khả nghi đến mức nào ở loại đó.

### Vì sao chọn XGBoost cho bài toán này

- **Dữ liệu dạng bảng (tabular):** mỗi session là một hàng với ~15–37 đặc trưng số — đúng "sân nhà" của cây boosting, thường mạnh hơn deep learning trên dữ liệu bảng.
- **Quan hệ phi tuyến & tương tác đặc trưng:** hành vi bất thường không tuyến tính (ví dụ "nhanh + nhiều + nửa đêm" mới đáng nghi); cây tự học các tương tác này.
- **Chịu được mất cân bằng:** kết hợp với cơ chế chọn ngưỡng riêng từng nhãn để bắt lớp hiếm (~3,6% anomaly, từng loại còn hiếm hơn).
- **Chống overfit:** có regularization, giới hạn độ sâu cây, lấy mẫu ngẫu nhiên dòng/cột.
- **Giải thích được:** cho `feature_importance` để biết đặc trưng nào quan trọng — phục vụ bảo vệ.

### Vai trò trong đồ án

XGBoost là **mô hình chính** (main supervised). Nó được huấn luyện trên `safe_features` (15 đặc trưng đã loại biến tạo luật) để kết quả không bị "chép luật". Bốn mô hình còn lại (LightGBM, Random Forest, Decision Tree, Extra Trees) dùng để đối chiếu — tất cả đều là **mô hình có giám sát, đa nhãn** (bản trước có Isolation Forest không giám sát, nay đã thay bằng Extra Trees).

### Bài toán đa nhãn & cách XGBoost xử lý

Một session có thể mang nhiều loại bất thường cùng lúc, nên đây là **đa nhãn 12 loại**. XGBoost gốc chỉ phân loại nhị phân, nên bọn em bọc nó trong `MultiOutputClassifier`: tạo **một bộ XGBoost cho mỗi loại** (One-vs-Rest), tất cả dùng chung 15 đặc trưng an toàn.

### Đặc trưng đầu vào: bao nhiêu, gồm những gì, dùng gì để train

- Mỗi session sinh ra **37** đặc trưng số (`full_feature_cols`).
- XGBoost (và cả 5 mô hình chính) **chỉ train trên 15** đặc trưng an toàn (`safe_feature_cols`).
- 22 đặc trưng còn lại (biến đếm trực tiếp tạo luật) bị loại để tránh label leakage; chúng chỉ xuất hiện trong bảng phụ `full_features`.

**15 đặc trưng XGBoost dùng để train:** `session_duration_sec`, `active_hours`, `unique_event_types`, `event_type_entropy`, `hour_entropy`, `mean_interval_sec`, `median_interval_sec`, `std_interval_sec`, `max_interval_sec`, `peak_events`, `weekend_events`, `peak_ratio`, `weekend_ratio`, `unique_parent_categories`, `session_dayofweek`.

---

## 2. CÁCH DÙNG (CHẠY NHƯ THẾ NÀO)

**Yêu cầu thư viện:**

```bash
.venv/bin/python -m pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn
# macOS có thể cần: brew install libomp   (runtime OpenMP cho XGBoost/LightGBM)
```

**Chạy:** mở `anomaly_detection_group_project.ipynb`, chạy tuần tự từ trên xuống (Restart & Run All). Thứ tự các cell liên quan XGBoost:

1. Cell cấu hình (import, hằng số `RANDOM_STATE`, đường dẫn).
2. Cell định nghĩa `safe_feature_cols` / `full_feature_cols` (chống leakage).
3. Cell chia dữ liệu theo nhóm visitor (train/val/test).
4. Cell **hạ tầng đa nhãn dùng chung** (`choose_thresholds_per_label`, `record_multilabel`, `train_and_evaluate_multilabel`).
5. **Cell huấn luyện XGBoost** (phần chính bên dưới).
6. Cell tổng hợp metric → xuất `group_model_metrics.csv`, `group_multilabel_type_metrics.csv`.

**Đầu ra:** điểm số `xgboost_score` cho từng session, bảng metric đa nhãn, F1 theo từng loại, và biểu đồ feature importance / confusion matrix.

---

## 3. GIẢI THÍCH TỪNG DÒNG CODE

### 3.1. Hàm tạo mô hình XGBoost & bọc đa nhãn

```python
def make_xgboost_model(n_estimators=120):
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
    )

# One-vs-Rest: 1 bộ XGBoost cho mỗi loại bất thường
xgboost_model = MultiOutputClassifier(make_xgboost_model(n_estimators=120), n_jobs=-1)
```

Giải thích từng tham số:

- `def make_xgboost_model(...)` — gói cấu hình vào một hàm để tái dùng (dùng lại y hệt cho bản `full_features` và bản shuffle-label, đảm bảo công bằng khi so sánh).
- `n_estimators=120` — số cây boosting (120 cây nối tiếp). Đủ để học, không quá nhiều gây overfit/chậm.
- `max_depth=4` — mỗi cây sâu tối đa 4 tầng. **Cây nông** → mỗi cây chỉ học mẫu đơn giản, chống overfit; sức mạnh đến từ việc cộng nhiều cây.
- `learning_rate=0.08` — tốc độ học (mỗi cây chỉ đóng góp 8% chỉnh sửa). Học **chậm mà chắc**, kết hợp với 120 cây cho kết quả ổn định.
- `subsample=0.85` — mỗi cây chỉ dùng ngẫu nhiên 85% **số dòng (session)** → tăng tính tổng quát, giảm overfit.
- `colsample_bytree=0.85` — mỗi cây chỉ dùng ngẫu nhiên 85% **số cột (đặc trưng)** → tránh phụ thuộc quá mức vào vài đặc trưng.
- `objective='binary:logistic'` — mỗi bộ con phân loại nhị phân (loại này có/không), đầu ra là xác suất.
- `eval_metric='logloss'` — hàm đo lỗi trong lúc train (log loss, phù hợp xác suất).
- `tree_method='hist'` — thuật toán dựng cây dạng histogram, **nhanh** và tiết kiệm bộ nhớ trên dữ liệu lớn.
- `random_state=RANDOM_STATE` — cố định hạt ngẫu nhiên (=10) để **kết quả tái lập được**.
- `n_jobs=-1` — dùng tất cả nhân CPU để train song song.
- `MultiOutputClassifier(...)` — bọc XGBoost lại thành mô hình đa nhãn: huấn luyện **một bộ XGBoost độc lập cho mỗi trong 12 loại**.

> **Khác bản nhị phân cũ:** không còn `scale_pos_weight`. Vì mỗi loại có tỉ lệ dương khác nhau, mất cân bằng được xử lý bằng **chọn ngưỡng riêng cho từng nhãn** (mục 3.4) — linh hoạt hơn một trọng số chung.

### 3.2. Huấn luyện + đánh giá qua hàm dùng chung

```python
(
    xgboost_model, xgboost_thresholds,
    xgboost_scores_all, xgboost_predictions_all,
) = train_and_evaluate_multilabel(
    model=xgboost_model, model_name='XGBoost',
    features_train=features_safe_train, labels_train_matrix=labels_multi_train_model,
    features_validation=features_safe_validation, labels_validation_matrix=labels_multi_validation_model,
    features_test=features_safe_test, labels_test_matrix=labels_multi_test_model,
    features_all=features_safe, label_names=model_label_names,
    feature_set='safe_features', evaluation_type=MULTILABEL_MAIN_EVALUATION,
)
```

- Hàm `train_and_evaluate_multilabel` (dùng chung cho cả 5 mô hình) lần lượt: `.fit` trên train; chọn ngưỡng riêng từng nhãn trên validation; đánh giá validation + test; chấm điểm cho **toàn bộ session**.
- Trả về: mô hình, **vector 12 ngưỡng**, ma trận điểm và ma trận dự đoán 0/1 cho mọi session.
- `labels_multi_train_model` là **ma trận nhãn 12 cột** (mỗi cột một loại bất thường).

### 3.3. Bên trong: huấn luyện trên TRAIN

```python
model.fit(features_train, labels_train_matrix)   # học trên 15 đặc trưng an toàn + 12 nhãn
```

Chỉ học trên tập train (`features_safe_train`), **không** đụng tới validation/test. `MultiOutputClassifier` tự train 12 bộ XGBoost, mỗi bộ cho một loại.

### 3.4. Chọn ngưỡng quyết định RIÊNG cho từng nhãn trên VALIDATION

```python
validation_scores = proba_matrix(model, features_validation, n_labels)  # ma trận xác suất 12 cột
thresholds = choose_thresholds_per_label(labels_validation_matrix, validation_scores, 'XGBoost')
```

- `proba_matrix(...)` — lấy **xác suất dương cho cả 12 loại** (ma trận `n_samples × 12`).
- `choose_thresholds_per_label(...)` — với **mỗi loại**, quét nhiều ngưỡng và chọn ngưỡng cho **F1 cao nhất trên validation**, kèm **ràng buộc chống over-predict**: không cho tỉ lệ dự đoán dương vượt ~3 lần tỉ lệ nền của loại đó (`max_pos_rate_multiple=3.0`). Vì sao không mặc định 0,5? Vì dữ liệu mất cân bằng, ngưỡng tối ưu mỗi loại thường khác 0,5. Chọn trên validation chứ **không** trên test để giữ test khách quan.

### 3.5. Đánh giá trên TEST (số báo cáo cuối)

```python
test_scores = proba_matrix(model, features_test, n_labels)
test_predictions = predict_with_thresholds(test_scores, thresholds)  # dùng lại đúng 12 ngưỡng đã chốt ở val
record_multilabel('XGBoost', 'safe_features', ..., 'test',
                  labels_test_matrix, test_predictions, label_names, score_matrix=test_scores)
```

- Tính xác suất cho **tập test**, áp **đúng bộ ngưỡng đã chốt ở validation** (không dò lại trên test) → đánh giá trung thực.
- `record_multilabel(...)` tính macro-F1, micro-F1, macro-PR-AUC, Hamming loss, subset-accuracy và **F1 theo từng loại**, rồi in `classification_report`.

### 3.6. Gán điểm cho toàn bộ dữ liệu (để xuất báo cáo)

```python
sessions['xgboost_pred']  = xgboost_predictions_all.max(axis=1)     # có loại nào bị gắn cờ không (0/1)
sessions['xgboost_score'] = xgboost_scores_all.max(axis=1)          # điểm cao nhất trong 12 loại
sessions['xgboost_pred_types'] = predicted_types_string(xgboost_predictions_all, model_label_names)
```

- Chấm điểm cho **tất cả 1,76 triệu session**, suy ra cờ nhị phân (`.max` theo hàng), điểm đại diện, và **chuỗi tên các loại dự đoán**. Vì XGBoost là mô hình chính, `xgboost_pred_types` được lấy làm `predicted_anomaly_types` trong báo cáo cuối.

### 3.7. Các bước nền (để hiểu code chạy được)

**Chia dữ liệu theo nhóm visitor (chống rò rỉ):**

```python
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
trainval_pos, test_pos = next(gss_test.split(supervised_indices, labels_sup, visitor_ids_sup))
```

- `GroupShuffleSplit` chia theo **nhóm** (`visitorid`), đảm bảo **mọi session của một visitor nằm trọn trong một tập**. Sau đó chia tiếp validation từ phần còn lại (60/20/20). Cuối cell có `assert` kiểm tra 0 visitor trùng giữa các tập.

**Dùng toàn bộ session:**

```python
if MAX_SUPERVISED_ROWS is not None and len(labels) > MAX_SUPERVISED_ROWS:
    supervised_indices, _ = train_test_split(all_indices, train_size=MAX_SUPERVISED_ROWS, ...)
else:
    supervised_indices = all_indices            # trần 3 triệu > 1,76 triệu -> dùng hết
```

---

## 4. ĐÁNH GIÁ KẾT QUẢ ĐẦU RA

### 4.1. Vì sao không nhìn accuracy

Dữ liệu mất cân bằng (~3,6%): đoán "tất cả bình thường" đã ~96% accuracy nhưng vô dụng. Vì vậy nhìn **macro-F1, micro-F1, macro-PR-AUC**.

- **macro-F1**: trung bình F1 của 12 loại, coi mọi loại quan trọng như nhau → làm nổi các loại hiếm.
- **micro-F1**: gộp toàn bộ cặp (session, loại) rồi tính một F1 chung → bị chi phối bởi loại phổ biến.
- **macro-PR-AUC**: trung bình diện tích dưới đường precision–recall của từng loại, **đáng tin cho lớp hiếm**.

### 4.2. Kết quả XGBoost (`safe_features`)

| Tập | macro-F1 | micro-F1 | macro-PR-AUC |
|---|--:|--:|--:|
| Validation | 0,621 | 0,792 | 0,639 |
| **Test** | **0,628** | **0,790** | **0,646** |

**Đọc kết quả:**

- **macro-F1 0,628** — trung bình tốt qua 12 loại, đáng kể trên đặc trưng an toàn (đã loại biến tạo luật).
- **micro-F1 0,790** — cao hơn macro vì các loại lớn (Click fraud, Bot scraper) được XGBoost bắt rất tốt (F1 ~0,97–0,99).
- **Khoảng cách validation–test rất nhỏ** (macro-F1 0,621 → 0,628) → mô hình **ổn định, chưa overfit rõ rệt**.

### 4.3. F1 theo từng loại (test) — điểm mạnh/yếu của XGBoost

| Loại | F1 XGBoost | support | Nhận xét |
|---|--:|--:|---|
| Click fraud | 0,991 | 6.319 | tín hiệu hành vi rõ → gần như hoàn hảo |
| Bot scraper | 0,974 | 2.598 | rất tốt |
| Category scanning | 0,922 | 345 | tốt dù hiếm |
| Transaction burst | 0,809 | 339 | tốt |
| Session bomb | 0,770 | 1.022 | khá |
| Night crawler | 0,715 | 6.111 | khá |
| Cart abandonment | 0,710 | 320 | khá |
| Rapid-fire | 0,531 | 1.578 | trung bình |
| Item hoarding | 0,421 | 662 | yếu |
| Repeated view spam | 0,261 | 716 | yếu |
| Sequence violation | 0,236 | 286 | yếu |
| Ghost buyer | 0,197 | 471 | yếu nhất |

Các loại yếu (Ghost buyer, Sequence violation, Repeated view spam) có bản chất luật nằm ở **biến đếm đã bị loại khỏi safe set**, nên rất khó tách chỉ bằng đặc trưng thời gian/bối cảnh — đây là cái giá có chủ đích của việc chống leakage.

### 4.4. Confusion matrix "có bất thường hay không" (test, 352.326 session)

`show_member_results` quy đa nhãn về nhị phân (gắn ≥ 1 loại = bất thường) để vẽ confusion matrix. Với XGBoost (support bất thường = 12.701):

|  | Dự đoán: Bình thường | Dự đoán: Bất thường |
|---|--:|--:|
| **Thật: Bình thường** | TN ≈ 337.812 | FP ≈ 1.813 |
| **Thật: Bất thường** | FN ≈ 1.836 | TP ≈ 10.865 |

- **TP 10.865 / FN 1.836:** bắt đúng ~85,5% session bất thường (recall ≈ 0,855).
- **FP 1.813:** báo nhầm 1.813 session bình thường (precision ≈ 0,857).

### 4.5. So với các mô hình khác (test, `safe_features`)

| Mô hình (thành viên) | macro-F1 | micro-F1 | macro-PR-AUC |
|---|--:|--:|--:|
| Dummy baseline | 0,004 | 0,011 | 0,005 |
| **LightGBM (Thủy)** | **0,642** | **0,816** | **0,659** |
| **XGBoost (chính)** | **0,628** | 0,790 | 0,646 |
| Decision Tree (Lê Văn Anh) | 0,536 | 0,718 | 0,550 |
| Random Forest (Tuấn Anh) | 0,361 | 0,251 | 0,417 |
| Extra Trees (Đức Anh) | 0,227 | 0,148 | 0,267 |

- **Dummy ~0** → bài toán thật sự có ý nghĩa học máy, không phải đoán mò.
- **XGBoost & LightGBM dẫn đầu**; LightGBM nhỉnh hơn chút (macro-F1 0,642 > 0,628) nhưng cùng họ boosting, kết quả sát nhau củng cố độ tin cậy. XGBoost là mô hình chính theo phân công.
- **Random Forest & Extra Trees over-predict** trên safe features (gắn cờ quá nhiều → precision thấp → micro-F1 sụp). Không phải lỗi pipeline, hướng khắc phục là calibration / siết ràng buộc tỉ lệ dương.

### 4.6. Bằng chứng kết quả KHÔNG bị "ảo"

- **`full_features` (chứa biến tạo luật):** XGBoost macro-F1 test = **0,982**, micro-F1 ≈ 0,997 — gần tuyệt đối. Đây là minh chứng *label leakage* (mô hình chép luật) → **không dùng làm kết luận**, chỉ để so sánh. Khoảng cách full (~0,98) → safe (~0,63) cho thấy việc loại biến tạo luật đã chặn phần lớn leakage.
- **Shuffle-label sanity check:** train với nhãn xáo trộn ngẫu nhiên, XGBoost macro-F1 test tụt còn **0,082** (micro-F1 0,145) → đúng kỳ vọng, xác nhận pipeline lành mạnh (không có rò rỉ ẩn).

### 4.7. Đặc trưng quan trọng nhất

Feature importance (trung bình qua 12 bộ con đa nhãn) cho thấy nhóm dẫn đầu là các đặc trưng **thời gian/nhịp thao tác**: `std_interval_sec` (độ lệch khoảng cách giữa các event), `session_duration_sec`, `mean_interval_sec`, `median_interval_sec`, `max_interval_sec`. Hợp lý về nghiệp vụ: bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

---

## 5. TÓM TẮT 30 GIÂY (để nói khi bảo vệ)

"XGBoost là mô hình chính, huấn luyện đa nhãn trên 15 đặc trưng an toàn đã loại biến tạo luật, bọc trong MultiOutputClassifier để dự đoán đồng thời 12 loại bất thường. Em xử lý mất cân bằng bằng cách chọn ngưỡng tối ưu F1 riêng cho từng loại trên validation (có chặn over-predict), cây nông max_depth=4 chống overfit. Kết quả test: macro-F1 0,628, micro-F1 0,790, macro-PR-AUC 0,646 — các loại có tín hiệu rõ như click fraud, bot scraper đạt F1 ~0,97–0,99. Khoảng cách validation–test rất nhỏ nên mô hình ổn định, và bài kiểm tra shuffle-label (macro-F1 tụt còn 0,08) xác nhận kết quả không bị rò rỉ."
