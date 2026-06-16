# TÀI LIỆU BẢO VỆ ĐỒ ÁN — BẢN CHI TIẾT (KÈM CODE)

## Phát hiện hành vi bất thường theo session — RetailRocket E-commerce

> Đây là bản mở rộng của `TAI_LIEU_BAO_VE_TONG_HOP.md`: bổ sung **code thật trong notebook** ở từng bước và một mục riêng giải thích kỹ **phương pháp xử lý mất cân bằng dữ liệu**. Mọi số liệu lấy từ `group_model_metrics.csv`. File notebook: `anomaly_detection_group_project.ipynb`.
>
> Cách dùng: học bản tổng hợp để nắm ý; mở bản này khi cần giải thích "code chỗ này làm gì" hoặc bị hỏi sâu về kỹ thuật.

---

## MỤC LỤC

1. Tóm tắt & mục tiêu
2. Dữ liệu & tạo session (code)
3. Xây đặc trưng session profile (code)
4. Hệ thống 12 luật nghiệp vụ (code)
5. Chống rò rỉ nhãn — safe vs full features (code)
6. Chia dữ liệu theo nhóm visitor (code)
7. **Xử lý mất cân bằng dữ liệu (chi tiết + code)**
8. Hàm dùng chung: chọn ngưỡng & đánh giá (code)
9. Năm mô hình (code từng mô hình)
10. Kết quả & đánh giá đầu ra
11. Bằng chứng kết quả không bị "ảo"
12. Điểm yếu, hướng phát triển, phân công

---

## 1. TÓM TẮT & MỤC TIÊU

"Đồ án phát hiện các phiên truy cập (session) có hành vi đáng nghi trên website TMĐT, dùng dataset RetailRocket. Vì dataset **không có nhãn bất thường thật**, nhóm xây 12 luật nghiệp vụ để sinh nhãn giả (pseudo-label), rồi huấn luyện 5 thuật toán khai phá dữ liệu để học lại và đánh giá. Quy trình: **Luật → Mô hình → Đánh giá → Xuất báo cáo.**"

**Quy mô dữ liệu:** 2.755.641 events · 1.407.580 visitor · 1.761.675 session · **3,51% session bất thường** (mất cân bằng nặng — xem Phần 7).

**Cấu hình toàn cục (cell 3):**

```python
RANDOM_STATE = 10                 # cố định hạt ngẫu nhiên -> kết quả tái lập được
DATA_PATH = Path('./data/events.csv')
SESSION_GAP_SEC = 30 * 60         # ngưỡng tách session = 30 phút (1800 giây)
MAX_SUPERVISED_ROWS = 3_000_000   # trần số session đưa vào train; lớn hơn tổng số session (~1,76 tr) -> DÙNG TOÀN BỘ session, không lấy mẫu con
IFOREST_TRAIN_ROWS = 250_000      # số dòng train cho Isolation Forest
```

---

## 2. DỮ LIỆU & TẠO SESSION (cell 5)

**Ý tưởng:** chuyển dữ liệu từ mức "sự kiện rời rạc" sang mức "phiên truy cập (session)", vì hành vi bất thường thường gói gọn trong một phiên ngắn. Một session mới bắt đầu khi là event đầu của visitor hoặc cách event trước **hơn 30 phút**.

```python
events = pd.read_csv(DATA_PATH)

# 1) Làm sạch: bỏ trùng lặp, chỉ giữ 3 loại event hợp lệ
events = events.drop_duplicates().reset_index(drop=True)
events = events[events['event'].isin(['view', 'addtocart', 'transaction'])].copy()

# 2) Đổi timestamp (mili-giây) sang datetime, tách giờ/thứ
events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')
events['hour'] = events['datetime'].dt.hour
events['dayofweek'] = events['datetime'].dt.dayofweek

# 3) Sắp xếp theo visitor + thời gian, tính khoảng cách với event trước
events = events.sort_values(['visitorid', 'timestamp', 'event', 'itemid']).reset_index(drop=True)
events['prev_timestamp'] = events.groupby('visitorid')['timestamp'].shift(1)
events['time_diff_sec'] = (events['timestamp'] - events['prev_timestamp']) / 1000

# 4) Mở session mới khi: event đầu của visitor (NaN) HOẶC nghỉ > 30 phút
new_session = events['time_diff_sec'].isna() | (events['time_diff_sec'] > SESSION_GAP_SEC)
events['session_number'] = new_session.groupby(events['visitorid']).cumsum().astype('int32')
events['session_id'] = events['visitorid'].astype(str) + '_S' + events['session_number'].astype(str)
```

Giải thích các dòng then chốt:

- `drop_duplicates()` — loại dòng log trùng để không thổi phồng số liệu.
- `isin(['view','addtocart','transaction'])` — chỉ giữ 3 loại hành vi có nghĩa.
- `pd.to_datetime(..., unit='ms')` — timestamp gốc là mili-giây → đổi sang datetime để lấy giờ/thứ phục vụ luật night-crawler, weekend...
- `groupby('visitorid')['timestamp'].shift(1)` — lấy thời điểm event liền trước **của cùng visitor**.
- `new_session = isna | (> SESSION_GAP_SEC)` — đây chính là logic cắt phiên 30 phút.
- `cumsum()` — cộng dồn cờ "mở phiên mới" để đánh số thứ tự phiên trong mỗi visitor; ghép `visitorid + '_S' + số` thành `session_id` duy nhất.

---

## 3. XÂY ĐẶC TRƯNG SESSION PROFILE (cell 8)

**Ý tưởng:** mỗi session được mô tả bằng ~37 đặc trưng số thuộc 6 nhóm. Code gom theo `session_id` rồi tính:

```python
g = df.groupby('session_id', sort=False)
profile['total_events'] = g.size()                    # tần suất
profile['unique_items']  = g['itemid'].nunique()      # độ đa dạng
profile['session_duration_sec'] = (g['timestamp'].max() - g['timestamp'].min())/1000

# Đếm theo loại event -> n_view, n_addtocart, n_transaction
event_counts = df.groupby(['session_id','event']).size().unstack(fill_value=0)

# Entropy giờ hoạt động (đo độ "trải đều" theo giờ)
hour_probs = hour_counts.div(hour_counts.sum(axis=1).clip(lower=1), axis=0)
profile['hour_entropy'] = -(hour_probs.replace(0,np.nan)*np.log2(...)).sum(axis=1)

# Nhịp thao tác: thống kê khoảng cách thời gian giữa các event trong phiên
speed_stats = time_diffs.groupby('session_id')['time_diff_session_sec'].agg(
    min_interval_sec='min', mean_interval_sec='mean', median_interval_sec='median',
    std_interval_sec='std', max_interval_sec='max')

# Hành vi "bắn liên thanh": số cặp event cách nhau < 1 giây
profile['rapid_fire_count'] = time_diffs[time_diffs['time_diff_session_sec']<1]\
                                .groupby('session_id').size()

# Tỉ lệ hành vi
profile['events_per_minute'] = profile['total_events'] / profile['duration_min']
profile['night_ratio'] = profile['night_events'] / profile['total_events']
profile['rapid_ratio'] = profile['rapid_fire_count'] / profile['total_events']
```

6 nhóm đặc trưng:

- **Tần suất:** `total_events`, `n_view`, `n_addtocart`, `n_transaction`.
- **Độ đa dạng:** `unique_items`, `unique_categories`, `unique_parent_categories`.
- **Thời gian/nhịp:** `session_duration_sec`, `events_per_minute`, `std/max/min_interval_sec`, `hour_entropy`.
- **Tốc độ:** `rapid_fire_count`, `rapid_ratio`.
- **Hành vi lặp:** `max_same_item_view`, `max_same_item_atc`.
- **Tỉ lệ:** `view_rate`, `atc_rate`, `buy_rate`, `night_ratio`.

> `std_interval_sec` (độ lệch khoảng cách giữa các event) rất quan trọng: bot bấm đều tăm tắp nên độ lệch thấp bất thường, khác hẳn người thật.

---

## 4. HỆ THỐNG 12 LUẬT NGHIỆP VỤ (cell 11)

**Ý tưởng:** không có nhãn thật → mã hóa "bất thường" thành 12 điều kiện đo được (mỗi luật là một *labeling function* theo tinh thần weak supervision / Snorkel). **Multi-label:** một phiên có thể dính nhiều luật; chỉ cần vi phạm ≥ 1 luật là bị coi là bất thường.

Ngưỡng được khai báo tập trung trong một dict để dễ chỉnh:

```python
THRESHOLDS = {
    'BR01_min_events_bot': 12, 'BR01_min_events_per_minute': 8, 'BR01_min_events_for_speed': 5,
    'BR03_min_views_for_click_fraud': 6,
    'BR04_rapid_fire_count': 1, 'BR04_rapid_fire_ratio': 0.20, 'BR04_min_events_for_rapid': 3,
    'BR05_night_ratio': 0.75, 'BR05_min_events_for_night': 4,
    'BR06_max_same_item_atc': 1, 'BR07_min_unique_items_session_bomb': 12,
    'BR09_min_transactions_burst': 3, 'BR10_min_addtocart_abandonment': 10,
    'BR11_max_same_item_view': 20, 'BR12_min_unique_categories': 20,
}
anomalies = profile.copy()
```

Vài luật tiêu biểu (code thật):

```python
# BR01 Bot scraper: quá nhiều event HOẶC thao tác quá nhanh
anomalies['flag_BR01_bot_scraper'] = (
    (anomalies['total_events'] >= 12)
    | ((anomalies['events_per_minute'] >= 8) & (anomalies['total_events'] >= 5))
).astype(int)

# BR02 Ghost buyer: có mua nhưng chưa từng thêm giỏ (phi logic)
anomalies['flag_BR02_ghost_buyer'] = (
    (anomalies['n_transaction'] > 0) & (anomalies['n_addtocart'] == 0)
).astype(int)

# BR08 Sequence violation: giao dịch xảy ra TRƯỚC mọi view/addtocart cùng item
interaction_event = df['event'].isin(['view','addtocart']).astype('int8')
prior = interaction_event.groupby([df['session_id'], df['itemid']]).cumsum()
viol = df.loc[(df['event']=='transaction') & (prior==0), 'session_id'].unique()
anomalies['flag_BR08_sequence_violation'] = anomalies.index.isin(viol).astype(int)
```

Gộp tất cả cờ lại để ra nhãn cuối:

```python
flag_cols = list(anomaly_type_map.keys())          # 12 cột flag_BR..
anomalies['total_flags'] = anomalies[flag_cols].sum(axis=1)
anomalies['is_anomaly_rule'] = (anomalies['total_flags'] > 0).astype(int)   # >=1 luật = bất thường
```

**Phân bố loại bất thường** (`group_anomaly_type_breakdown.csv`):

| Loại | Số session | Tỉ lệ |
|---|---:|---:|
| Click fraud | 31.213 | 1,77% |
| Night crawler | 30.130 | 1,71% |
| Bot scraper | 10.884 | 0,62% |
| Rapid-fire | 8.052 | 0,46% |
| Session bomb | 5.112 | 0,29% |
| Item hoarding | 3.212 | 0,18% |
| Ghost buyer | 2.365 | 0,13% |
| Sequence violation | 1.573 | 0,09% |
| Transaction burst | 1.558 | 0,09% |
| Category scanning | 628 | 0,04% |
| Cart abandonment | 224 | 0,01% |
| Repeated view spam | 49 | 0,003% |

Tổng (gộp đa nhãn): **61.849 session bất thường = 3,51%**.

---

## 5. CHỐNG RÒ RỈ NHÃN — SAFE vs FULL FEATURES (cell 14)

**Vấn đề:** nếu cho mô hình học bằng chính các biến tạo ra luật (vd `total_events`, `n_view`...), nó sẽ "chép" lại luật → metric đẹp giả tạo (label leakage).

**Giải pháp:** định nghĩa 2 bộ đặc trưng và **kiểm toán tự động** xem bộ an toàn còn sót biến tạo luật không:

```python
safe_feature_cols = [  # 15 đặc trưng — KHÔNG chứa biến trực tiếp tạo luật
    'session_duration_sec','active_hours','unique_event_types','event_type_entropy',
    'hour_entropy','mean_interval_sec','median_interval_sec','std_interval_sec',
    'max_interval_sec','peak_events','weekend_events','peak_ratio','weekend_ratio',
    'unique_parent_categories','session_dayofweek',
]
full_feature_cols = [ ... 37 đặc trưng, gồm cả biến tạo luật ... ]

leaky_feature_set = set(full_feature_cols).difference(safe_feature_cols)
leakage_audit_df['inputs_remaining_in_safe_features'] = leakage_audit_df['rule_input_features'].apply(
    lambda cols: [c for c in cols if c in safe_feature_cols])

# CHẶN CỨNG: nếu safe_features còn chứa biến tạo luật -> báo lỗi, dừng chạy
if leakage_audit_df['inputs_remaining_in_safe_features'].str.len().sum() != 0:
    raise ValueError('safe_feature_cols vẫn chứa đặc trưng trực tiếp tạo nhãn luật')
```

- `safe_features` (15) → dùng làm **kết quả chính**.
- `full_features` (37) → chỉ để minh họa hiện tượng "học lại luật" (Phần 11).
- File `group_leakage_audit.csv` ghi rõ mỗi luật dùng biến gì và biến nào đã bị loại.

### 5.1. Tổng kết số lượng đặc trưng

| Bộ đặc trưng | Số lượng | Vai trò |
| --- | --- | --- |
| `full_feature_cols` | **37** | Toàn bộ đặc trưng số của session; chỉ dùng cho **bảng phụ** minh họa label leakage (rule-mimic). |
| `safe_feature_cols` | **15** | **Bộ dùng để TRAIN** mọi mô hình báo cáo chính (XGBoost, Decision Tree, Random Forest, LightGBM, Isolation Forest). |
| Bị loại khỏi safe (leaky) | **22** | Là `full − safe`: các biến trực tiếp/gián tiếp tạo ra luật, **không** đưa vào train chính. |

### 5.2. 15 đặc trưng DÙNG ĐỂ TRAIN (`safe_feature_cols`)

| # | Đặc trưng | Ý nghĩa |
| --- | --- | --- |
| 1 | `session_duration_sec` | Thời lượng phiên (giây). |
| 2 | `active_hours` | Số khung giờ khác nhau có hoạt động trong phiên. |
| 3 | `unique_event_types` | Số loại sự kiện khác nhau (view/addtocart/transaction). |
| 4 | `event_type_entropy` | Entropy phân bố loại sự kiện — độ đa dạng hành vi. |
| 5 | `hour_entropy` | Entropy phân bố giờ hoạt động — mức trải đều theo giờ. |
| 6 | `mean_interval_sec` | Khoảng cách trung bình giữa các thao tác. |
| 7 | `median_interval_sec` | Khoảng cách trung vị giữa các thao tác. |
| 8 | `std_interval_sec` | Độ lệch chuẩn khoảng cách thao tác — bot bấm đều nên rất thấp. |
| 9 | `max_interval_sec` | Khoảng cách lớn nhất giữa hai thao tác liên tiếp. |
| 10 | `peak_events` | Số sự kiện trong khung giờ cao điểm (9h–21h). |
| 11 | `weekend_events` | Số sự kiện vào cuối tuần. |
| 12 | `peak_ratio` | Tỉ lệ sự kiện rơi vào giờ cao điểm. |
| 13 | `weekend_ratio` | Tỉ lệ sự kiện rơi vào cuối tuần. |
| 14 | `unique_parent_categories` | Số nhóm danh mục cha khác nhau đã chạm tới. |
| 15 | `session_dayofweek` | Thứ trong tuần của phiên. |

> Đây là các đặc trưng **về thời gian, nhịp thao tác và bối cảnh**, không phải biến đếm cấu thành luật → mô hình buộc phải học "dáng hành vi" thay vì chép lại điều kiện luật.

### 5.3. 22 đặc trưng BỊ LOẠI khỏi train chính (chỉ có trong `full_features`)

`total_events`, `unique_items`, `n_view`, `n_addtocart`, `n_transaction`, `events_per_minute`, `duration_min`, `min_interval_sec`, `rapid_fire_count`, `rapid_ratio`, `night_events`, `night_ratio`, `max_same_item_view`, `max_same_item_atc`, `unique_categories`, `view_rate`, `atc_rate`, `buy_rate`, `items_per_event`, `view_to_cart_ratio`, `cart_to_transaction_ratio`, `session_start_hour`.

Lý do loại: đây là các biến **trực tiếp tạo ra luật** — ví dụ `total_events`→BR01, `n_transaction`/`n_addtocart`→BR02/BR09/BR10, `rapid_fire_count`→BR04, `night_ratio`→BR05, `max_same_item_atc`→BR06, `unique_items`→BR07, `max_same_item_view`→BR11, `unique_categories`→BR12. Nếu cho train sẽ gây label leakage.

---

## 6. CHIA DỮ LIỆU THEO NHÓM VISITOR (cell 16)

**Vấn đề:** một visitor có nhiều session. Nếu chia ngẫu nhiên theo session, các session của cùng người có thể nằm cả ở train lẫn test → mô hình "thấy trước" người đó → metric cao giả.

**Giải pháp:** `GroupShuffleSplit` theo `visitorid` + lấy mẫu phân tầng giữ tỉ lệ anomaly:

```python
# Trần MAX_SUPERVISED_ROWS = 3 triệu > tổng ~1,76 triệu session -> DÙNG TOÀN BỘ session.
# (Nhánh lấy mẫu phân tầng chỉ kích hoạt nếu số session vượt trần)
if MAX_SUPERVISED_ROWS is not None and len(labels) > MAX_SUPERVISED_ROWS:
    supervised_indices, _ = train_test_split(all_indices, train_size=MAX_SUPERVISED_ROWS,
                                             random_state=RANDOM_STATE, stratify=labels)
else:
    supervised_indices = all_indices            # hiện tại đi vào nhánh này

groups_sup = anomalies['visitorid'].to_numpy()[supervised_indices]

# Tách test 20% theo NHÓM visitor (cả cụm session của 1 visitor đi cùng nhau)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
trainval_pos, test_pos = next(gss_test.split(supervised_indices, y_sup, groups_sup))

# Tách tiếp validation (25% của phần còn lại = 20% tổng) -> 60/20/20
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)

# Kiểm tra tự động: 0 visitor trùng giữa các tập
assert not (overlap_tv or overlap_tt or overlap_vt), 'Van con visitor bi ro ri giua cac tap!'
```

Kết quả kiểm tra in ra: **0 visitor trùng** giữa train/validation/test. Quy mô chia 60/20/20 trên toàn bộ session: **train 1.056.863 · validation 352.486 · test 352.326** (tương ứng 844.548 / 281.516 / 281.516 visitor). Tỉ lệ anomaly giữ xấp xỉ nhau: **3,51% / 3,48% / 3,55%**. Sau đó dùng `RobustScaler` (chuẩn hóa chịu được outlier) fit trên train rồi áp cho cả 3 tập — fit chỉ trên train để không rò rỉ thông tin test.

---

## 7. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU (CHI TIẾT)

Đây là vấn đề cốt lõi: chỉ **~3,51%** session là bất thường. Nếu không xử lý, mô hình sẽ "lười" dự đoán tất cả là bình thường (đã đạt ~96,5% accuracy nhưng vô dụng). Nhóm xử lý ở **4 tầng**, **không dùng oversampling tổng hợp (SMOTE)**.

### Tầng 1 — Cân bằng ở mức thuật toán (trọng số lớp) — cách chính

Tính tỉ lệ âm/dương từ tập train rồi đưa vào mô hình:

```python
neg_count = int((y_train == 0).sum())          # số session bình thường
pos_count = int((y_train == 1).sum())          # số session bất thường
scale_pos_weight = neg_count / max(pos_count, 1)   # ≈ 27 (vì anomaly ~3,5%)
```

- **XGBoost & LightGBM:** truyền `scale_pos_weight ≈ 27`. Nghĩa là "một lỗi bỏ sót anomaly bị phạt nặng gấp ~27 lần lỗi báo nhầm" → mô hình chú ý bắt lớp hiếm.

```python
XGBClassifier(..., scale_pos_weight=scale_pos_weight)
LGBMClassifier(..., scale_pos_weight=scale_pos_weight)
```

- **Decision Tree:** `class_weight='balanced'` — tự động tăng trọng số lớp hiếm tỉ lệ nghịch với tần suất.

```python
DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, class_weight='balanced', ...)
```

- **Random Forest:** `class_weight='balanced_subsample'` — như trên nhưng tính lại trọng số cho từng mẫu bootstrap của mỗi cây.

```python
RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=50,
                       class_weight='balanced_subsample', ...)
```

### Tầng 2 — Lấy mẫu phân tầng (stratified sampling) khi cần

Nhóm để sẵn cơ chế lấy mẫu phân tầng `stratify=y` nhằm **giữ đúng tỉ lệ ~3,5% anomaly** nếu phải rút bớt session. Hiện tại trần `MAX_SUPERVISED_ROWS = 3 triệu` **lớn hơn** tổng ~1,76 triệu session nên code đi vào nhánh **dùng toàn bộ session** — không cần lấy mẫu con:

```python
if MAX_SUPERVISED_ROWS is not None and len(labels) > MAX_SUPERVISED_ROWS:
    supervised_indices, _ = train_test_split(all_indices, train_size=MAX_SUPERVISED_ROWS,
                                             random_state=RANDOM_STATE, stratify=labels)
else:
    supervised_indices = all_indices            # hiện tại: train trên toàn bộ session
```

Notebook in ra tỉ lệ anomaly của từng tập để kiểm chứng chúng xấp xỉ nhau (**3,51% / 3,48% / 3,55%** cho train/validation/test), nhờ chia nhóm theo visitor.

### Tầng 3 — Tinh chỉnh ngưỡng quyết định (threshold tuning)

Không cắt mặc định 0,5. Thay vào đó quét ngưỡng tối ưu **F1 trên validation** rồi cố định áp lên test. Vì lớp dương hiếm, ngưỡng tối ưu thường khác 0,5 (XGBoost = **0,8845**). Đây là cách bù mất cân bằng ở khâu ra quyết định (xem hàm `choose_best_threshold` ở Phần 8).

### Tầng 4 — Chọn metric phù hợp lớp hiếm

Bỏ accuracy, dùng **Precision / Recall / F1 / PR-AUC**. PR-AUC là chỉ số đáng tin nhất cho lớp hiếm vì nó tập trung vào khả năng bắt đúng lớp dương.

### Vì sao KHÔNG dùng SMOTE / oversampling tổng hợp?

Hai lý do, nên nói nếu giám khảo hỏi: (1) các kỹ thuật trọng số lớp đã đủ hiệu quả và rẻ hơn về tính toán; (2) SMOTE tạo session "nhân tạo" bằng nội suy giữa các mẫu, có thể sinh ra mẫu hành vi **phi thực tế** làm méo các tín hiệu thời gian/nhịp vốn rất nhạy của bài toán này. Đây là lựa chọn có chủ đích.

> Tóm tắt 1 câu: "Bọn em xử lý mất cân bằng ở 4 tầng — trọng số lớp (`scale_pos_weight` cho boosting, `class_weight='balanced'` cho cây), lấy mẫu phân tầng, tinh chỉnh ngưỡng trên validation, và chọn metric PR-AUC/F1 — không dùng SMOTE để tránh tạo hành vi nhân tạo."

---

## 8. HÀM DÙNG CHUNG: CHỌN NGƯỠNG & ĐÁNH GIÁ (cell 17)

**`choose_best_threshold`** — quét ngưỡng, chọn ngưỡng cho F1 cao nhất trên validation:

```python
def choose_best_threshold(y_true, y_score, label, metric='f1'):
    # ứng viên ngưỡng = các phân vị của điểm số (1%..99%) + mốc 0.5
    candidate_thresholds = np.unique(np.r_[0.5, np.quantile(y_score, np.linspace(0.01,0.99,99))])
    best_threshold, best_value = 0.5, -1.0
    for threshold in candidate_thresholds:
        y_pred = (y_score >= threshold).astype(int)
        value = f1_score(y_true, y_pred, zero_division=0)
        if value > best_value:           # giữ ngưỡng cho F1 tốt nhất (hòa thì ưu tiên precision)
            best_threshold, best_value = float(threshold), float(value)
    return best_threshold
```

**`add_evaluation`** — tính đủ bộ metric và lưu vào bảng kết quả:

```python
record = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, zero_division=0),
    'recall': recall_score(y_true, y_pred, zero_division=0),
    'f1_score': f1_score(y_true, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_true, y_score),
    'pr_auc': average_precision_score(y_true, y_score),   # PR-AUC
    ...
}
```

Mọi mô hình đều đi qua đúng 2 hàm này → so sánh công bằng.

---

## 9. NĂM MÔ HÌNH (code từng mô hình)

### 9.1. XGBoost — mô hình chính (cell 21)

```python
def make_xgb_model(n_estimators=120, scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        objective='binary:logistic', eval_metric='logloss', tree_method='hist',
        random_state=RANDOM_STATE, n_jobs=-1, scale_pos_weight=scale_pos_weight)

xgb_safe_model = make_xgb_model(120, scale_pos_weight)
xgb_safe_model.fit(X_safe_train, y_train)                          # train

val_score = xgb_safe_model.predict_proba(X_safe_val)[:, 1]         # xác suất bất thường
threshold = choose_best_threshold(y_val, val_score, 'XGBoost')     # chốt ngưỡng = 0,8845 trên val
test_score = xgb_safe_model.predict_proba(X_safe_test)[:, 1]
test_pred  = (test_score >= threshold).astype(int)                 # áp NGUYÊN ngưỡng đó lên test
```

Ý nghĩa tham số: `max_depth=4` cây nông chống overfit; `learning_rate=0.08`+`n_estimators=120` học chậm mà chắc; `subsample/colsample=0.85` lấy mẫu ngẫu nhiên dòng/cột tăng tổng quát; `scale_pos_weight` bù mất cân bằng; `tree_method='hist'` train nhanh. (Giải thích chi tiết hơn xem `GIAI_THICH_CODE_XGBOOST.md`.)

### 9.2. Decision Tree (cell 23) — dễ giải thích

```python
dt_safe_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100,
                                       class_weight='balanced', random_state=RANDOM_STATE)
```

Cây đơn, vẽ được sơ đồ quyết định để trình bày trực quan. `class_weight='balanced'` xử lý mất cân bằng.

### 9.3. Random Forest (cell 25) — ensemble

```python
rf_safe_model = RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=50,
                                       class_weight='balanced_subsample', n_jobs=-1, random_state=RANDOM_STATE)
```

120 cây song song (bagging), ổn định, cho feature importance.

### 9.4. LightGBM (cell 27) — đối chiếu XGBoost

```python
def make_lgbm_model(n_estimators=160, scale_pos_weight=1.0):
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.85, colsample_bytree=0.85, objective='binary',
        scale_pos_weight=scale_pos_weight, reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1)
```

Cũng là gradient boosting nhưng mọc lá theo kiểu leaf-wise → nhanh hơn. Dùng để đối chiếu với XGBoost.

### 9.5. Isolation Forest (cell 35) — tham chiếu unsupervised

```python
iforest_model = IsolationForest(n_estimators=150, contamination='auto',
                                random_state=RANDOM_STATE, n_jobs=-1)
iforest_model.fit(X_safe_scaled[iforest_train_indices])     # KHÔNG dùng nhãn khi train
iforest_score = -iforest_model.score_samples(X_safe_scaled) # điểm càng cao càng bất thường
```

Hoàn toàn không dùng nhãn — chỉ tìm điểm dị biệt theo phân bố. Dùng để báo mức overlap với pseudo-label, không kỳ vọng cạnh tranh trực tiếp với supervised.

---

## 10. KẾT QUẢ & ĐÁNH GIÁ ĐẦU RA

**Vì sao không dùng accuracy:** mất cân bằng (~3,5%), đoán "tất cả bình thường" đã ~96,5% accuracy nhưng vô dụng → dùng Precision/Recall/F1/PR-AUC.

### Bảng kết quả chính — tập TEST, `safe_features` (`group_model_metrics.csv`)

| Mô hình | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|--:|--:|--:|--:|--:|
| Dummy baseline | 0,033 | 0,033 | 0,033 | 0,499 | 0,035 |
| Decision Tree | 0,789 | 0,707 | 0,746 | 0,988 | 0,791 |
| Random Forest | 0,810 | 0,935 | 0,868 | 0,993 | 0,919 |
| **XGBoost (chính)** | 0,805 | 0,931 | **0,864** | 0,994 | 0,929 |
| **LightGBM** | 0,818 | 0,951 | **0,879** | 0,995 | 0,942 |
| Isolation Forest | 0,302 | 0,606 | 0,403 | 0,935 | 0,347 |

- **Dummy ~0** → bài toán có ý nghĩa học máy thật.
- **LightGBM dẫn đầu** (F1 0,879); **Random Forest (0,868) và XGBoost (0,864) bám sát nhau** — ba mô hình boosting/ensemble đều mạnh, recall ~0,93–0,95 (bắt được phần lớn anomaly), precision ~0,81–0,82. XGBoost vẫn được chọn làm **mô hình chính** theo phân công.
- **Isolation Forest thấp** vì unsupervised.

### XGBoost — validation vs test

| Tập | Precision | Recall | F1 | PR-AUC | Ngưỡng |
|---|--:|--:|--:|--:|--:|
| Validation | 0,805 | 0,925 | 0,861 | 0,929 | 0,8845 |
| Test | 0,805 | 0,931 | 0,864 | 0,929 | 0,8845 |

Khoảng cách val–test rất nhỏ (F1 0,861 → 0,864) → **ổn định, chưa overfit**.

### Ma trận nhầm lẫn XGBoost (test, 352.326 session)

|  | Dự đoán Bình thường | Dự đoán Bất thường |
|---|--:|--:|
| **Thật: Bình thường** | TN ≈ 337.020 | FP ≈ 2.815 |
| **Thật: Bất thường** | FN ≈ 861 | TP ≈ 11.630 |

Bắt đúng 11.630 / bỏ sót 861 (recall 0,931); báo nhầm 2.815 (precision 0,805). (Suy ra từ support=12.491, predicted=14.445; khớp accuracy 0,9896.)

### Đặc trưng quan trọng nhất (feature importance)

Dẫn đầu áp đảo là nhóm thời gian/nhịp: với XGBoost, `std_interval_sec` chiếm ~0,84 importance, bỏ xa phần còn lại; tiếp theo là `session_duration_sec`, `mean_interval_sec`, `median_interval_sec`, `peak_ratio`. Random Forest/LightGBM cũng xếp `std_interval_sec`, `session_duration_sec`, `mean_interval_sec` lên đầu — hợp lý vì bot/rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

### Báo cáo bất thường tổng hợp (cell 40)

Bước cuối gộp tín hiệu **luật HOẶC mô hình** (`is_anomaly_final = is_rule_anomaly | is_model_anomaly`, với `is_model_anomaly` = có ít nhất 1 trong 5 mô hình gắn cờ) rồi gán mức độ nghiêm trọng. Kết quả: **159.205 / 1.761.675 session (9,04%)** bị đánh dấu trong báo cáo cuối — cao hơn 3,51% của riêng luật vì cộng thêm các phiên chỉ mô hình nghi ngờ. Phân bố mức độ: **CRITICAL 67.506 · MEDIUM 75.520 · HIGH 16.179**. Đây là đầu ra nghiệp vụ trong `session_anomaly_report.csv`, không phải metric đánh giá mô hình.

---

## 11. BẰNG CHỨNG KẾT QUẢ KHÔNG BỊ "ẢO"

**full_features (cell 29)** — train lại đúng các mô hình trên 37 đặc trưng (có biến tạo luật):

```python
xgb_full_model = make_xgb_model(120, scale_pos_weight)
xgb_full_model.fit(X_full_train, y_train)
```

→ XGBoost test F1 = **0,998**, PR-AUC ≈ 1,0 (LightGBM full F1 = 1,000). Gần tuyệt đối vì mô hình **chép lại luật** → đây là minh chứng leakage, **không dùng làm kết luận**.

**Shuffle-label sanity check (cell 31)** — train với nhãn bị xáo trộn ngẫu nhiên:

```python
y_train_shuffled = pd.Series(rng.permutation(y_train.to_numpy()), index=y_train.index)
xgb_shuffle_model = make_xgb_model(60, shuffle_scale_pos_weight)
xgb_shuffle_model.fit(X_safe_train, y_train_shuffled)
```

→ F1 tụt còn **0,272**, PR-AUC 0,147 (≈ tỉ lệ nền 3,55%). Đúng kỳ vọng (mô hình không thể học từ nhãn rác) → xác nhận pipeline không có rò rỉ ẩn. Notebook còn tự cảnh báo nếu PR-AUC shuffle cao bất thường.

---

## 12. ĐIỂM YẾU, HƯỚNG PHÁT TRIỂN, PHÂN CÔNG

**Điểm yếu & cách bảo vệ:**

1. *Nhãn là pseudo-label* → metric đo "mức khớp với luật", chưa khẳng định bắt gian lận thật. Thừa nhận thẳng + đề xuất thu thập nhãn thật.
2. *F1 safe vẫn cao (0,86) — hết leakage chưa?* → đã loại biến trực tiếp tạo luật; các biến thời gian còn lại vẫn tương quan gián tiếp nên còn một phần tín hiệu. Khoảng cách full (~1,0) → safe (0,86) cho thấy đã giảm leakage đáng kể nhưng chưa triệt để.
3. *LightGBM cao nhất, Random Forest ≈ XGBoost (F1 0,879 > 0,868 ≈ 0,864)* → cùng nhóm boosting/ensemble, sát nhau, củng cố độ tin cậy; chọn XGBoost làm chính theo phân công dù không phải cao nhất.
4. *Chưa đánh giá theo thời gian (drift).*

**Hướng phát triển:** time-based split; calibration (Platt/Isotonic); tối ưu ngưỡng theo chi phí nghiệp vụ; phân tích độ nhạy ngưỡng luật + label model (Snorkel); và quan trọng nhất là thu thập nhãn thật.

**Phân công nhóm:**

| Thành viên | Thuật toán | Vai trò |
|---|---|---|
| Đình Tuấn (nhóm trưởng) | XGBoost | Mô hình chính + điều phối |
| Lê Văn Anh | Decision Tree | Dễ giải thích trực quan |
| Tuấn Anh | Random Forest | Ensemble + feature importance |
| Thủy | LightGBM | Đối chiếu với XGBoost |
| Đức Anh | Isolation Forest | Tham chiếu unsupervised |

**Tệp đầu ra:** `group_model_metrics.csv`, `group_leakage_audit.csv`, `group_anomaly_type_breakdown.csv`, `session_anomaly_tags.csv`, `session_anomaly_report.csv`.
