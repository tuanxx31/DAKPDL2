# TÀI LIỆU BẢO VỆ ĐỒ ÁN — BẢN CHI TIẾT (KÈM CODE)

## Phát hiện hành vi bất thường theo session — RetailRocket E-commerce

> Đây là bản mở rộng của `TAI_LIEU_BAO_VE_TONG_HOP.md`: bổ sung **code thật trong notebook** ở từng bước và một mục riêng giải thích kỹ **phương pháp xử lý mất cân bằng dữ liệu**. Bài toán là **phân loại ĐA NHÃN** (multi-label): mỗi session được dự đoán thuộc loại bất thường nào trong 12 loại. Mọi số liệu lấy từ `group_model_metrics.csv` và `group_multilabel_type_metrics.csv`. File notebook: `anomaly_detection_group_project.ipynb`.
>
> Cách dùng: học bản tổng hợp để nắm ý; mở bản này khi cần giải thích "code chỗ này làm gì" hoặc bị hỏi sâu về kỹ thuật.

---

## MỤC LỤC

1. Tóm tắt & mục tiêu
2. Dữ liệu & tạo session (code)
3. Xây đặc trưng session profile (code)
4. Hệ thống 12 luật nghiệp vụ + cơ sở chọn ngưỡng (code)
5. Chống rò rỉ nhãn — safe vs full features (code)
6. Chia dữ liệu theo nhóm visitor (code)
7. **Xử lý mất cân bằng dữ liệu (chi tiết + code)**
8. Hạ tầng đa nhãn dùng chung: chọn ngưỡng & đánh giá (code)
9. Năm mô hình đa nhãn (code từng mô hình)
10. Kết quả & đánh giá đầu ra (đa nhãn)
11. Bằng chứng kết quả không bị "ảo"
12. Điểm yếu, hướng phát triển, phân công

---

## 1. TÓM TẮT & MỤC TIÊU

"Đồ án phát hiện các phiên truy cập (session) có hành vi đáng nghi trên website TMĐT, dùng dataset RetailRocket. Vì dataset **không có nhãn bất thường thật**, nhóm xây 12 luật nghiệp vụ để sinh nhãn giả (pseudo-label) **đa nhãn**, rồi huấn luyện 5 thuật toán khai phá dữ liệu để học lại và đánh giá. Quy trình: **Luật → Mô hình → Đánh giá → Xuất báo cáo.**"

**Bài toán học máy:** mỗi session có thể mang **đồng thời nhiều loại** bất thường (ví dụ vừa "bot scraper" vừa "category scanning"). Vì vậy đây là **phân loại đa nhãn 12 loại**, không phải nhị phân một cờ. Mỗi mô hình học cách gắn từng loại cho từng session.

**Quy mô dữ liệu:** 2.755.641 events · 1.407.580 visitor · 1.761.675 session · **3,57% session bất thường** theo luật (62.894 session — mất cân bằng nặng, xem Phần 7).

**Cấu hình toàn cục (cell 3):**

```python
RANDOM_STATE = 10                 # cố định hạt ngẫu nhiên -> kết quả tái lập được
DATA_PATH = Path('./data/events.csv')
SESSION_GAP_SEC = 30 * 60         # ngưỡng tách session = 30 phút (1800 giây)
MAX_SUPERVISED_ROWS = 3_000_000   # trần số session đưa vào train; lớn hơn tổng số session (~1,76 tr) -> DÙNG TOÀN BỘ session, không lấy mẫu con
PLOT_SAMPLE_ROWS = 30_000         # số dòng lấy mẫu khi vẽ biểu đồ cho nhẹ
```

> So với phiên bản trước: đã **bỏ Isolation Forest** (mô hình không giám sát) và đưa **Extra Trees** (có giám sát, đa nhãn) vào thay; nhóm có giám sát hiện gồm đủ 5 mô hình.

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

## 4. HỆ THỐNG 12 LUẬT NGHIỆP VỤ + CƠ SỞ CHỌN NGƯỠNG (cell 12)

**Ý tưởng:** không có nhãn thật → mã hóa "bất thường" thành 12 điều kiện đo được (mỗi luật là một *labeling function* theo tinh thần weak supervision / Snorkel). **Multi-label:** một phiên có thể dính nhiều luật; chỉ cần vi phạm ≥ 1 luật là bị coi là bất thường, và **mỗi luật là một nhãn loại riêng** để mô hình học dự đoán.

### 4.1. Cơ sở chọn ngưỡng (threshold rationale — cell 11)

Tất cả ngưỡng cường độ được chọn dựa trên **phân vị (percentile) thực nghiệm** của toàn bộ 1,76 triệu session, **không phải số tùy ý**. Triết lý: mỗi luật nhắm vào nhóm hành vi **hiếm** (~top 0,1–1%) nhưng vẫn đủ mẫu dương để mô hình học được.

So với bản trước, **ba ngưỡng đã được hạ xuống** vì đặt quá cao tạo "lớp chết" (gần như không có mẫu dương):

| Rule | Đặc trưng | Cũ | Mới | Phân vị thực tế | Lý do |
| --- | --- | --- | --- | --- | --- |
| BR10 | n_addtocart (& tx=0) | 10 | **4** | ≈ p99.9 = 4 | cũ 10 ≫ p99.9 → chỉ 224 session (lớp gần rỗng) |
| BR11 | max_same_item_view | 20 | **6** | ≈ p99.9 = 7 | cũ 20 chỉ trúng **49 session** → F1 ≈ 0,03 (lớp chết) |
| BR12 | unique_categories | 20 | **10** | ≈ p99.9 = 9 | cũ 20 ≫ p99.9 → quá hiếm |

Các ngưỡng còn lại (BR01 `total_events≥12` ~p99.5, BR03 `n_view≥6` ~p98, BR04, BR05, BR06, BR07, BR09) **giữ nguyên** vì đã nằm đúng vùng p98–p99.9 và đủ mẫu. Ngoài ra `events_per_minute` của BR01 hạ từ 8 → **4** (≈ p99.9 ≈ 4,6) vì mốc 8 gần như không bao giờ đạt.

**Tác động:** số session dương tăng để học được — Repeated view spam 49 → **3.391**; Cart abandonment 224 → **1.586**; Category scanning 628 → **1.666**.

### 4.2. Khai báo ngưỡng (code thật)

```python
THRESHOLDS = {
    'BR01_min_events_bot': 12,            # p99.5 của total_events
    'BR01_min_events_per_minute': 4,      # p99.9 của events_per_minute (~4.6); HẠ TỪ 8
    'BR01_min_events_for_speed': 5,
    'BR03_min_views_for_click_fraud': 6,  # ~p98 của n_view
    'BR04_rapid_fire_count': 1, 'BR04_rapid_fire_ratio': 0.20, 'BR04_min_events_for_rapid': 3,
    'BR05_night_ratio': 0.75, 'BR05_min_events_for_night': 4,
    'BR06_max_same_item_atc': 1, 'BR07_min_unique_items_session_bomb': 12,
    'BR09_min_transactions_burst': 3,
    'BR10_min_addtocart_abandonment': 4,  # SỬA: 10 -> 4 (~p99.9)
    'BR11_max_same_item_view': 6,         # SỬA: 20 -> 6 (~p99.9=7)
    'BR12_min_unique_categories': 10,     # SỬA: 20 -> 10 (~p99.9=9)
}
sessions = session_profile.copy()
```

Vài luật tiêu biểu (code thật):

```python
# BR01 Bot scraper: quá nhiều event HOẶC thao tác quá nhanh
sessions['flag_BR01_bot_scraper'] = (
    (sessions['total_events'] >= 12)
    | ((sessions['events_per_minute'] >= 4) & (sessions['total_events'] >= 5))
).astype(int)

# BR02 Ghost buyer: có mua nhưng chưa từng thêm giỏ (phi logic)
sessions['flag_BR02_ghost_buyer'] = (
    (sessions['n_transaction'] > 0) & (sessions['n_addtocart'] == 0)
).astype(int)

# BR08 Sequence violation: giao dịch xảy ra TRƯỚC mọi view/addtocart cùng item
interaction_event = session_events['event'].isin(['view','addtocart']).astype('int8')
prior = interaction_event.groupby([session_events['session_id'], session_events['itemid']]).cumsum()
viol = session_events.loc[(session_events['event']=='transaction') & (prior==0), 'session_id'].unique()
sessions['flag_BR08_sequence_violation'] = sessions.index.isin(viol).astype(int)
```

Gộp 12 cờ thành ma trận nhãn đa nhãn + nhãn nhị phân tổng:

```python
flag_cols = list(anomaly_type_map.keys())          # 12 cột flag_BR..
sessions['total_flags'] = sessions[flag_cols].sum(axis=1)
sessions['is_anomaly_rule'] = (sessions['total_flags'] > 0).astype(int)   # >=1 luật = bất thường
sessions['anomaly_types'] = [', '.join(...) if row.any() else 'Normal' for row in flag_matrix]
```

> `flag_cols` (12 cột 0/1) chính là **ma trận nhãn đa nhãn** mà các mô hình sẽ học dự đoán. `is_anomaly_rule` chỉ là nhãn nhị phân tổng "có bất thường hay không", dùng cho chia dữ liệu và so sánh phụ.

**Phân bố loại bất thường** (`group_anomaly_type_breakdown.csv`):

| Loại | Số session | Tỉ lệ |
|---|---:|---:|
| Click fraud | 31.213 | 1,77% |
| Night crawler | 30.130 | 1,71% |
| Bot scraper | 12.888 | 0,73% |
| Rapid-fire | 8.052 | 0,46% |
| Session bomb | 5.112 | 0,29% |
| Repeated view spam | 3.391 | 0,19% |
| Item hoarding | 3.212 | 0,18% |
| Ghost buyer | 2.365 | 0,13% |
| Category scanning | 1.666 | 0,09% |
| Cart abandonment | 1.586 | 0,09% |
| Sequence violation | 1.573 | 0,09% |
| Transaction burst | 1.558 | 0,09% |

Tổng (gộp đa nhãn, ≥1 luật): **62.894 session bất thường = 3,57%**.

---

## 5. CHỐNG RÒ RỈ NHÃN — SAFE vs FULL FEATURES (cell 15)

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

# CHẶN CỨNG: nếu safe_features còn chứa biến tạo luật -> báo lỗi, dừng chạy
if leakage_audit_df['inputs_remaining_in_safe_features'].str.len().sum() != 0:
    raise ValueError('safe_feature_cols vẫn chứa đặc trưng trực tiếp tạo nhãn luật')
```

- `safe_features` (15) → dùng làm **kết quả chính** cho cả 5 mô hình.
- `full_features` (37) → chỉ để minh họa hiện tượng "học lại luật" (Phần 11).
- File `group_leakage_audit.csv` ghi rõ mỗi luật dùng biến gì và biến nào đã bị loại.

### 5.1. Tổng kết số lượng đặc trưng

| Bộ đặc trưng | Số lượng | Vai trò |
| --- | --- | --- |
| `full_feature_cols` | **37** | Toàn bộ đặc trưng số của session; chỉ dùng cho **bảng phụ** minh họa label leakage (rule-mimic). |
| `safe_feature_cols` | **15** | **Bộ dùng để TRAIN** cả 5 mô hình báo cáo chính (XGBoost, Decision Tree, Random Forest, LightGBM, Extra Trees). |
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

## 6. CHIA DỮ LIỆU THEO NHÓM VISITOR (cell 17)

**Vấn đề:** một visitor có nhiều session. Nếu chia ngẫu nhiên theo session, các session của cùng người có thể nằm cả ở train lẫn test → mô hình "thấy trước" người đó → metric cao giả.

**Giải pháp:** `GroupShuffleSplit` theo `visitorid` (cùng cụm session của một visitor luôn đi cùng một tập):

```python
# Trần MAX_SUPERVISED_ROWS = 3 triệu > tổng ~1,76 triệu session -> DÙNG TOÀN BỘ session.
if MAX_SUPERVISED_ROWS is not None and len(labels) > MAX_SUPERVISED_ROWS:
    supervised_indices, _ = train_test_split(all_indices, train_size=MAX_SUPERVISED_ROWS,
                                             random_state=RANDOM_STATE, stratify=labels)
else:
    supervised_indices = all_indices            # hiện tại đi vào nhánh này

# Tách test 20% theo NHÓM visitor
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
trainval_pos, test_pos = next(gss_test.split(supervised_indices, labels_sup, visitor_ids_sup))

# Tách tiếp validation (25% của phần còn lại = 20% tổng) -> 60/20/20
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)

# Kiểm tra tự động: 0 visitor trùng giữa các tập
assert not (overlap_tv or overlap_tt or overlap_vt), 'Vẫn còn visitor bị rò rỉ giữa các tập!'
```

Kết quả kiểm tra in ra: **0 visitor trùng** giữa train/validation/test. Quy mô chia 60/20/20 trên toàn bộ session: **train 1.056.863 · validation 352.486 · test 352.326**. Tỉ lệ anomaly giữ xấp xỉ nhau (~3,5–3,6% mỗi tập, test ≈ 3,60% = 12.701 session). Sau đó dùng `RobustScaler` (chuẩn hóa chịu được outlier) fit trên train rồi áp cho cả 3 tập — fit chỉ trên train để không rò rỉ thông tin test.

---

## 7. XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU (CHI TIẾT)

Đây là vấn đề cốt lõi: chỉ **~3,57%** session là bất thường, và **trong từng loại còn hiếm hơn nhiều** (ví dụ Sequence violation chỉ ~0,09%). Nếu không xử lý, mô hình sẽ "lười" dự đoán tất cả là bình thường. Nhóm xử lý ở **4 tầng**, **không dùng oversampling tổng hợp (SMOTE)**.

### Tầng 1 — Cân bằng ở mức thuật toán (trọng số lớp)

- **Decision Tree:** `class_weight='balanced'` — tự động tăng trọng số lớp hiếm tỉ lệ nghịch với tần suất.
- **Random Forest & Extra Trees:** `class_weight='balanced_subsample'` — như trên nhưng tính lại trọng số cho từng mẫu bootstrap của mỗi cây.

```python
DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight='balanced', ...)
RandomForestClassifier(n_estimators=120, max_depth=16, min_samples_leaf=20,
                       class_weight='balanced_subsample', ...)
ExtraTreesClassifier(n_estimators=200, max_depth=16, min_samples_leaf=20,
                     class_weight='balanced_subsample', ...)
```

> **Thay đổi so với bản trước:** XGBoost/LightGBM **không còn** dùng một `scale_pos_weight` toàn cục. Vì bài toán nay là đa nhãn (mỗi loại có tỉ lệ dương khác nhau), mất cân bằng được xử lý chủ yếu ở **Tầng 3 — chọn ngưỡng RIÊNG cho từng nhãn** (xem dưới), linh hoạt hơn nhiều so với một trọng số chung.

### Tầng 2 — Lấy mẫu phân tầng (stratified sampling) khi cần

Nhóm để sẵn cơ chế lấy mẫu phân tầng `stratify=labels` nếu phải rút bớt session. Hiện tại trần `MAX_SUPERVISED_ROWS = 3 triệu` **lớn hơn** tổng ~1,76 triệu session nên code đi vào nhánh **dùng toàn bộ session** — không cần lấy mẫu con. Việc chia theo nhóm visitor cũng giữ tỉ lệ anomaly giữa các tập xấp xỉ nhau.

### Tầng 3 — Chọn ngưỡng quyết định RIÊNG cho từng nhãn (threshold tuning đa nhãn)

Không cắt mặc định 0,5. Với **mỗi trong 12 nhãn**, quét ngưỡng tối ưu **F1 trên validation** rồi cố định áp lên test. Quan trọng: có thêm **ràng buộc chống over-predict** — không cho tỉ lệ dự đoán dương vượt quá ~3 lần tỉ lệ nền của nhãn đó (`max_pos_rate_multiple=3.0`), tránh mô hình "bắn bừa" để ăn recall. Đây là cách bù mất cân bằng ở khâu ra quyết định (hàm `choose_thresholds_per_label`, Phần 8).

### Tầng 4 — Chọn metric phù hợp lớp hiếm & đa nhãn

Bỏ accuracy, dùng **macro-F1 / micro-F1 / macro-PR-AUC** (và Hamming loss). *macro-F1* tính trung bình F1 của 12 loại, coi mọi loại quan trọng như nhau (làm nổi loại hiếm); *micro-F1* gộp toàn bộ cặp (session, loại) nên thiên về loại phổ biến.

### Vì sao KHÔNG dùng SMOTE / oversampling tổng hợp?

Hai lý do, nên nói nếu giám khảo hỏi: (1) trọng số lớp + chọn ngưỡng riêng từng nhãn đã đủ hiệu quả và rẻ hơn về tính toán; (2) SMOTE tạo session "nhân tạo" bằng nội suy giữa các mẫu, có thể sinh ra mẫu hành vi **phi thực tế** làm méo các tín hiệu thời gian/nhịp vốn rất nhạy của bài toán này. Đây là lựa chọn có chủ đích.

> Tóm tắt 1 câu: "Bọn em xử lý mất cân bằng ở 4 tầng — trọng số lớp cho các mô hình cây, lấy mẫu phân tầng, **chọn ngưỡng tối ưu F1 riêng cho từng nhãn** (có chặn over-predict), và chọn metric macro/micro-F1 + PR-AUC — không dùng SMOTE để tránh tạo hành vi nhân tạo."

---

## 8. HẠ TẦNG ĐA NHÃN DÙNG CHUNG: CHỌN NGƯỠNG & ĐÁNH GIÁ (cell 19)

Mọi mô hình giám sát chạy qua cùng một bộ hàm để so sánh công bằng. Ma trận nhãn là 12 cột `flag_BR..`; chỉ mô hình hoá các loại có đủ 2 lớp và ≥ 30 mẫu dương trong train (`MIN_LABEL_POSITIVES=30`) — thực tế **cả 12 loại đều đạt**.

**`choose_thresholds_per_label`** — chọn ngưỡng F1 tối ưu cho TỪNG nhãn trên validation, có chặn over-predict:

```python
def choose_thresholds_per_label(true_matrix, score_matrix, label, max_pos_rate_multiple=3.0):
    for j in range(n_labels):
        base_rate = true_matrix[:, j].mean()
        cap = min(1.0, max(base_rate * max_pos_rate_multiple, base_rate + 0.005))
        candidates = np.unique(np.r_[0.5, np.quantile(scores, np.linspace(0.01, 0.999, 99))])
        for t in candidates:
            pred = (scores >= t).astype(int)
            if pred.mean() > cap:        # bỏ ngưỡng làm dự đoán dương quá nhiều
                continue
            # giữ ngưỡng cho F1 cao nhất của nhãn j
```

**`record_multilabel`** — tính đủ bộ metric đa nhãn và F1 theo từng loại:

```python
macro_f1 = f1_score(true, pred, average='macro')   # trung bình F1 của 12 loại
micro_f1 = f1_score(true, pred, average='micro')   # gộp toàn bộ cặp (session, loại)
macro_pr_auc = mean(average_precision_score theo từng loại)
hamming = hamming_loss(true, pred)
# + precision_recall_fscore_support(average=None) -> F1 từng loại để in bảng
```

**`proba_matrix`** chuẩn hoá đầu ra `predict_proba` của cả `MultiOutputClassifier` (trả list) lẫn mô hình đa nhãn gốc (Random Forest/Extra Trees) về một ma trận xác suất `n_samples × 12`. **`show_member_results`** in macro/micro-F1, bảng F1 theo loại và confusion matrix "có bất thường hay không" **ngay trong section của từng thành viên**.

---

## 9. NĂM MÔ HÌNH ĐA NHÃN (code từng mô hình)

Tất cả train trên `safe_features`, dự đoán đồng thời 12 loại. XGBoost, LightGBM, Decision Tree bọc trong `MultiOutputClassifier` (One-vs-Rest: 1 bộ con/loại); Random Forest và Extra Trees hỗ trợ đa nhãn gốc (truyền y 2D).

### 9.1. XGBoost — mô hình chính, Đình Tuấn (cell 23)

```python
def make_xgboost_model(n_estimators=120):
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=4, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        objective='binary:logistic', eval_metric='logloss', tree_method='hist',
        random_state=RANDOM_STATE, n_jobs=-1)

xgboost_model = MultiOutputClassifier(make_xgboost_model(120), n_jobs=-1)  # 1 bộ cho mỗi loại
```

`max_depth=4` cây nông chống overfit; `learning_rate=0.08`+`n_estimators=120` học chậm mà chắc; `subsample/colsample=0.85` lấy mẫu ngẫu nhiên dòng/cột tăng tổng quát; `tree_method='hist'` train nhanh. Mất cân bằng xử lý bằng ngưỡng riêng từng nhãn (không dùng `scale_pos_weight`). (Chi tiết hơn xem `GIAI_THICH_CODE_XGBOOST.md`.)

### 9.2. Decision Tree — Lê Văn Anh (cell 25)

```python
decision_tree_model = MultiOutputClassifier(
    DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight='balanced',
                           random_state=RANDOM_STATE), n_jobs=-1)
```

Một cây cho mỗi loại (One-vs-Rest). Ngoài ra cell còn vẽ một **cây đơn nhãn minh hoạ** (`is_anomaly`, max_depth=4) để xem các nhánh quyết định trên safe features (`output_group_decision_tree.png`).

### 9.3. Random Forest — Tuấn Anh (cell 27)

```python
random_forest_model = RandomForestClassifier(
    n_estimators=120, max_depth=16, min_samples_leaf=20,
    class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
```

120 cây song song (bagging), hỗ trợ đa nhãn gốc, có `feature_importances_` trực tiếp.

### 9.4. LightGBM — Thủy (cell 29)

```python
def make_lightgbm_model(n_estimators=160):
    return LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=80, subsample=0.85, colsample_bytree=0.85,
        objective='binary', reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=-1,
        verbosity=-1, force_col_wise=True)
lightgbm_model = MultiOutputClassifier(make_lightgbm_model(160), n_jobs=-1)
```

Cũng là gradient boosting nhưng mọc lá theo kiểu leaf-wise → nhanh hơn. Dùng để đối chiếu với XGBoost.

### 9.5. Extra Trees — Đức Anh (cell 31)

```python
extra_trees_model = ExtraTreesClassifier(
    n_estimators=200, max_depth=16, min_samples_leaf=20,
    class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1)
```

Extra Trees (Extremely Randomized Trees) là ensemble cây **ngẫu nhiên hoá cả đặc trưng lẫn ngưỡng cắt** → giảm phương sai so với Random Forest và train nhanh hơn. Đây là mô hình **có giám sát, đa nhãn**, train cùng safe features và so sánh trực tiếp với 4 mô hình kia bằng macro/micro-F1 (thay cho Isolation Forest ở bản cũ).

---

## 10. KẾT QUẢ & ĐÁNH GIÁ ĐẦU RA (ĐA NHÃN)

**Vì sao không dùng accuracy:** mất cân bằng (~3,6%), đoán "tất cả bình thường" đã ~96% accuracy nhưng vô dụng → dùng **macro-F1 / micro-F1 / macro-PR-AUC**.

### Bảng kết quả chính — tập TEST, `safe_features` (`group_model_metrics.csv`)

| Mô hình (thành viên) | macro-F1 | micro-F1 | macro-PR-AUC |
|---|--:|--:|--:|
| Dummy baseline | 0,004 | 0,011 | 0,005 |
| **LightGBM (Thủy)** | **0,642** | **0,816** | **0,659** |
| **XGBoost (Đình Tuấn, chính)** | **0,628** | 0,790 | 0,646 |
| Decision Tree (Lê Văn Anh) | 0,536 | 0,718 | 0,550 |
| Random Forest (Tuấn Anh) | 0,361 | 0,251 | 0,417 |
| Extra Trees (Đức Anh) | 0,227 | 0,148 | 0,267 |

- **Dummy ~0** → bài toán có ý nghĩa học máy thật.
- **Hai mô hình boosting dẫn đầu rõ rệt:** LightGBM (macro-F1 0,642) nhỉnh hơn XGBoost (0,628) một chút; cùng họ gradient boosting nên sát nhau, củng cố độ tin cậy. XGBoost vẫn là **mô hình chính** theo phân công.
- **Decision Tree** ở giữa (0,536).
- **Random Forest và Extra Trees over-predict mạnh trên safe features:** chúng gắn cờ quá nhiều → precision thấp → micro-F1 sụp (RF 0,251; ET 0,148). Đây là hạn chế đã ghi nhận (xem Phần 12).

### F1 theo TỪNG LOẠI bất thường — test, safe features (`group_multilabel_type_metrics.csv`)

| Loại | XGBoost | LightGBM | Decision Tree | Random Forest | Extra Trees | support test |
|---|--:|--:|--:|--:|--:|--:|
| Click fraud | 0,991 | 0,997 | 0,942 | 0,593 | 0,573 | 6.319 |
| Night crawler | 0,715 | 0,774 | 0,726 | 0,537 | 0,000 | 6.111 |
| Bot scraper | 0,974 | 0,994 | 0,908 | 0,749 | 0,285 | 2.598 |
| Rapid-fire | 0,531 | 0,571 | 0,422 | 0,297 | 0,003 | 1.578 |
| Session bomb | 0,770 | 0,777 | 0,600 | 0,563 | 0,555 | 1.022 |
| Repeated view spam | 0,261 | 0,270 | 0,196 | 0,080 | 0,104 | 716 |
| Item hoarding | 0,421 | 0,410 | 0,376 | 0,275 | 0,236 | 662 |
| Ghost buyer | 0,197 | 0,173 | 0,189 | 0,001 | 0,000 | 471 |
| Category scanning | 0,922 | 0,921 | 0,859 | 0,858 | 0,792 | 345 |
| Transaction burst | 0,809 | 0,831 | 0,618 | 0,015 | 0,012 | 339 |
| Cart abandonment | 0,710 | 0,730 | 0,418 | 0,368 | 0,169 | 320 |
| Sequence violation | 0,236 | 0,260 | 0,180 | 0,001 | 0,000 | 286 |

- Loại có **tín hiệu hành vi rõ** (Click fraud, Bot scraper, Category scanning) đạt F1 ~0,9 ngay trên safe features.
- Loại **khó tách bằng đặc trưng an toàn** (Ghost buyer, Sequence violation, Repeated view spam) F1 thấp — hợp lý vì bản chất luật của chúng nằm ở biến đếm đã bị loại khỏi safe set.

### XGBoost — validation vs test (đa nhãn)

| Tập | macro-F1 | micro-F1 | macro-PR-AUC |
|---|--:|--:|--:|
| Validation | 0,621 | 0,792 | 0,639 |
| Test | 0,628 | 0,790 | 0,646 |

Khoảng cách val–test rất nhỏ → **ổn định, chưa overfit**.

### Confusion matrix "có bất thường hay không" (test, 352.326 session)

`show_member_results` quy đa nhãn về nhị phân (một session bị xem là bất thường nếu mô hình gắn ≥ 1 loại) để vẽ confusion matrix. Với **XGBoost** (test, support bất thường = 12.701):

|  | Dự đoán Bình thường | Dự đoán Bất thường |
|---|--:|--:|
| **Thật: Bình thường** | TN ≈ 337.812 | FP ≈ 1.813 |
| **Thật: Bất thường** | FN ≈ 1.836 | TP ≈ 10.865 |

→ precision ≈ 0,857, recall ≈ 0,855 ở mức "có bất thường hay không". So sánh nhanh giữa các mô hình ở mức nhị phân này: LightGBM (prec 0,844 / rec 0,929), Decision Tree (0,777 / 0,933), Random Forest (0,306 / 0,772), Extra Trees (0,201 / 0,535) — thấy rõ RF/ET báo nhầm rất nhiều.

### Đặc trưng quan trọng nhất (feature importance, cell 37)

Trung bình qua các bộ con đa nhãn, nhóm dẫn đầu áp đảo là các đặc trưng **thời gian/nhịp thao tác**: `std_interval_sec` (độ lệch khoảng cách giữa các event), `session_duration_sec`, `mean_interval_sec`, `median_interval_sec`, `max_interval_sec`. Hợp lý vì bot/rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật (`output_group_feature_importance.png`).

### Báo cáo bất thường tổng hợp (cell 42)

Bước cuối gộp tín hiệu **luật HOẶC mô hình** (`is_anomaly_final = is_rule_anomaly | is_model_anomaly`, với `is_model_anomaly` = có ít nhất 1 trong 5 mô hình gắn cờ) rồi gán mức độ nghiêm trọng (dùng số cờ luật, số phiếu mô hình, và `extra_trees_score` ở phân vị p99). Kết quả: **308.791 / 1.761.675 session (17,5%)** bị đưa vào báo cáo cuối — cao hơn nhiều so với 3,57% của riêng luật, **chủ yếu do Random Forest và Extra Trees over-predict** (riêng "ML model-only" đã 245.897 session). Phân bố mức độ: **CRITICAL 142.509 · MEDIUM 144.237 · HIGH 22.045**. Đây là đầu ra nghiệp vụ trong `session_anomaly_report.csv`, **không phải metric đánh giá mô hình**; khi bảo vệ nên nói rõ con số 17,5% này bị thổi lên bởi hai mô hình rừng over-predict.

---

## 11. BẰNG CHỨNG KẾT QUẢ KHÔNG BỊ "ẢO"

**full_features (cell 33)** — train lại đúng 5 mô hình trên 37 đặc trưng (có biến tạo luật):

→ macro-F1 test bùng lên gần tuyệt đối với nhóm boosting/cây: **XGBoost 0,982 · LightGBM 0,983 · Decision Tree 0,978** (micro-F1 ~0,997); Random Forest 0,862; Extra Trees 0,741. Đây là minh chứng leakage (mô hình **chép lại luật** khi nhìn thấy biến sinh luật) → **không dùng làm kết luận**. Khoảng cách full (~0,98) → safe (~0,63) cho thấy việc loại biến tạo luật đã chặn phần lớn leakage.

**Shuffle-label sanity check (cell 35)** — tráo (permute) các dòng nhãn train để phá liên kết đặc trưng–nhãn:

→ XGBoost macro-F1 test **tụt còn 0,082** (micro-F1 0,145), tức gần baseline ngẫu nhiên. Đúng kỳ vọng (mô hình không thể học từ nhãn rác) → xác nhận pipeline không có rò rỉ ẩn.

---

## 12. ĐIỂM YẾU, HƯỚNG PHÁT TRIỂN, PHÂN CÔNG

**Điểm yếu & cách bảo vệ:**

1. *Nhãn là pseudo-label* → metric đo "mức khớp với luật", chưa khẳng định bắt gian lận thật. Thừa nhận thẳng + đề xuất thu thập nhãn thật.
2. *macro-F1 safe ~0,63 — hết leakage chưa?* → đã loại biến trực tiếp tạo luật; các biến thời gian còn lại vẫn tương quan gián tiếp nên còn một phần tín hiệu. Khoảng cách full (~0,98) → safe (~0,63) cho thấy đã giảm leakage đáng kể nhưng chưa triệt để.
3. *Random Forest & Extra Trees over-predict* trên safe features (precision rất thấp → micro-F1 sụp: RF 0,251; ET 0,148). Nguyên nhân: `class_weight='balanced_subsample'` kết hợp chọn ngưỡng theo F1 đẩy quá nhiều dự đoán dương. Hướng khắc phục: bỏ `class_weight`, siết `max_pos_rate_multiple`, hoặc dùng `CalibratedClassifierCV`.
4. *LightGBM cao nhất, XGBoost sát ngay sau (macro-F1 0,642 vs 0,628)* → cùng nhóm boosting, sát nhau, củng cố độ tin cậy; chọn XGBoost làm chính theo phân công dù không phải cao nhất.
5. *Chưa đánh giá theo thời gian (drift).*

**Hướng phát triển:** time-based split; calibration (Platt/Isotonic) cho RF/ET; tối ưu ngưỡng theo chi phí nghiệp vụ; phân tích độ nhạy ngưỡng luật + label model (Snorkel); và quan trọng nhất là thu thập nhãn thật.

**Phân công nhóm:**

| Thành viên | Thuật toán | Loại | Vai trò |
|---|---|---|---|
| Đình Tuấn (nhóm trưởng) | XGBoost | Có giám sát (đa nhãn) | Mô hình chính + điều phối |
| Lê Văn Anh | Decision Tree | Có giám sát (đa nhãn) | Dễ giải thích trực quan |
| Tuấn Anh | Random Forest | Có giám sát (đa nhãn) | Ensemble + feature importance |
| Thủy | LightGBM | Có giám sát (đa nhãn) | Đối chiếu với XGBoost |
| Đức Anh | Extra Trees | Có giám sát (đa nhãn) | Ensemble cây ngẫu nhiên hoá, đối chiếu Random Forest |

**Tệp đầu ra:** `group_model_metrics.csv`, `group_multilabel_type_metrics.csv`, `group_leakage_audit.csv`, `group_anomaly_type_breakdown.csv`, `session_anomaly_tags.csv`, `session_anomaly_report.csv`.
