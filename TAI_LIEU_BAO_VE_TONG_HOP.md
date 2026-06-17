# TÀI LIỆU BẢO VỆ ĐỒ ÁN (TỔNG HỢP)

## Phát hiện hành vi bất thường theo session — RetailRocket E-commerce

> File này gộp toàn bộ ý chuẩn để bảo vệ: hướng dẫn, kịch bản trình bày, ngân hàng câu hỏi. Bài toán là **phân loại ĐA NHÃN** (multi-label): dự đoán mỗi session thuộc loại bất thường nào trong 12 loại. **Mọi số liệu lấy từ `group_model_metrics.csv` và `group_multilabel_type_metrics.csv`** (đã chạy group split theo `visitorid`). Đọc file này là đủ — các file còn lại chỉ để tham khảo.
>
> Cấu trúc: Phần 1–7 là kiến thức để học; Phần 8 là kịch bản nói; Phần 9 là ngân hàng câu hỏi; Phần 10–11 là phụ lục.

---

## PHẦN 0 — Tóm tắt mở đầu (học thuộc)

"Đồ án phát hiện các phiên truy cập (session) có hành vi đáng nghi trên website thương mại điện tử, dùng dataset RetailRocket. Vì dataset **không có nhãn bất thường thật**, nhóm xây 12 luật nghiệp vụ để sinh nhãn giả (pseudo-label) **đa nhãn**, rồi huấn luyện 5 thuật toán khai phá dữ liệu để học lại và đánh giá. Quy trình: **Luật → Mô hình → Đánh giá → Xuất báo cáo.** Em là nhóm trưởng, phụ trách mô hình chính XGBoost."

---

## PHẦN 0B — MỤC TIÊU, ỨNG DỤNG & VÍ DỤ

### Mục tiêu của đồ án

**Mục tiêu chung:** xây dựng một quy trình hoàn chỉnh, có thể tin cậy được, để **tự động phát hiện những phiên truy cập có hành vi đáng nghi** trên website thương mại điện tử — dựa hoàn toàn trên log hành vi, không cần con người gắn nhãn từng phiên.

Cụ thể đồ án nhắm tới 4 mục tiêu:

1. **Biến dữ liệu thô thành đơn vị phân tích có nghĩa:** chuyển ~2,75 triệu sự kiện rời rạc thành 1,76 triệu session, mỗi session là một "chân dung hành vi" mô tả được bằng ~37 đặc trưng.
2. **Định nghĩa "bất thường" đo lường được và giải thích được:** vì không có nhãn thật, nhóm mã hóa kinh nghiệm nghiệp vụ thành 12 luật để sinh nhãn giả **đa nhãn** — mỗi cảnh báo luôn kèm một lý do cụ thể (ví dụ "bot scraper", "click fraud").
3. **So sánh và chọn được mô hình khai phá dữ liệu phù hợp** để học lại các mẫu bất thường đó — **5 thuật toán có giám sát, đa nhãn:** XGBoost, LightGBM, Random Forest, Decision Tree, Extra Trees.
4. **Đảm bảo kết quả đáng tin, không bị "ảo":** qua ba lớp kiểm soát chất lượng (safe_features, group split, shuffle-label) để chống label leakage và over-fitting.

> Lưu ý phạm vi: mục tiêu là **gắn cờ phiên đáng nghi để xem lại** và **nhận diện loại bất thường**, KHÔNG phải khẳng định chắc chắn đó là gian lận. Đây là bước sàng lọc đầu tiên trong một hệ thống chống gian lận.

### Ứng dụng thực tế

Quy trình này có thể dùng làm **lớp sàng lọc tự động (screening)** trong nhiều bài toán vận hành sàn TMĐT:

- **Chống gian lận & lạm dụng:** phát hiện phiên có dấu hiệu click fraud, đặt hàng ảo, tấn công thử thẻ (card testing) để chuyển sang đội rủi ro xem xét.
- **Chặn bot & cào dữ liệu (scraping):** nhận diện phiên quét giá/sản phẩm tốc độ cao của đối thủ hoặc bot, để giới hạn truy cập (rate-limit) hoặc thêm captcha.
- **Bảo vệ chất lượng dữ liệu phân tích:** lọc các phiên bot/bất thường ra khỏi báo cáo lượt xem, tỉ lệ chuyển đổi... để số liệu marketing không bị méo.
- **Bảo vệ tồn kho & giá:** phát hiện hành vi gom giỏ hàng bất thường (item hoarding, cart bombing) làm khóa hàng ảo.
- **Cảnh báo vận hành theo thời gian thực:** vì mô hình cho ra điểm số cho từng loại của từng phiên, có thể gắn ngưỡng để bắn cảnh báo ngay khi một phiên vượt mức đáng nghi.

Điểm mạnh để ứng dụng: kết quả **giải thích được** (mỗi cờ gắn với một luật/loại), nên đội vận hành biết *vì sao* một phiên bị gắn cờ chứ không chỉ nhận một con số.

### Ví dụ cụ thể

**Ví dụ 1 — Bot cào dữ liệu (BR01 + BR12).** Một phiên xem 80 sản phẩm thuộc 25 danh mục khác nhau trong vòng 3 phút, nhịp thao tác đều tăm tắp ~0,2 giây/lần. Người thật không bấm đều và nhanh như vậy → mô hình gắn cờ "bot scraper" + "category scanning". *Ứng dụng:* chặn IP/giới hạn tốc độ truy cập.

**Ví dụ 2 — Click fraud lúc nửa đêm (BR03 + BR05).** Một phiên lúc 3 giờ sáng xem 30 lượt sản phẩm nhưng không thêm giỏ, không mua, lặp lại nhiều ngày. → gắn cờ "click fraud" + "night crawler". *Ứng dụng:* lọc khỏi báo cáo lượt xem quảng cáo, điều tra gian lận hiển thị.

**Ví dụ 3 — Đặt hàng ảo / thử thẻ (BR02 + BR09).** Một phiên có 5 giao dịch liên tiếp mà chưa từng thêm sản phẩm vào giỏ. → gắn cờ "ghost buyer" + "transaction burst". *Ứng dụng:* chuyển sang đội rủi ro kiểm tra thẻ thanh toán.

**Ví dụ 4 — Người mua bình thường (KHÔNG bị gắn cờ).** Một phiên 15 phút: xem 8 sản phẩm với nhịp tay người (vài giây đến vài chục giây/lần), thêm 2 món vào giỏ, mua 1 món, vào lúc 8 giờ tối. → không vi phạm luật nào → mô hình xếp là bình thường. Ví dụ này cho thấy luật **không gắn cờ bừa** người dùng thật.

---

## PHẦN 1 — DỮ LIỆU NHƯ THẾ NÀO?

**Nguồn:** RetailRocket E-commerce, file gốc `events.csv` (cùng `item_properties`, `category_tree`).

**Quy mô sau tiền xử lý:**

- Sự kiện hợp lệ: **2.755.641 events**
- Visitor duy nhất: **1.407.580**
- Session sau khi tách phiên: **1.761.675**
- Loại sự kiện: `view`, `addtocart`, `transaction`

**Hai đặc điểm phải nói thẳng:**

1. **Không có nhãn bất thường thật.** Dataset chỉ ghi lại hành vi, không có cột "đây là gian lận". Đó là lý do nhóm phải tự sinh nhãn bằng luật (pseudo-label).
2. **Mất cân bằng nặng:** chỉ ~3,57% session bị gắn nhãn bất thường (62.894 / 1.761.675), và **từng loại còn hiếm hơn nhiều**. Điều này quyết định cách chọn metric (xem Phần 4–5).

**Câu trả lời mẫu "dữ liệu gì":**
"Dữ liệu là log hành vi thật của người dùng trên sàn TMĐT, khoảng 2,75 triệu sự kiện của 1,4 triệu người. Bọn em chuyển nó từ mức sự kiện rời rạc sang mức session để phân tích hành vi. Hạn chế lớn nhất là dữ liệu không có nhãn bất thường sẵn."

---

## PHẦN 2 — Ý TƯỞNG XỬ LÝ

**Ý tưởng trung tâm: chuyển từ "sự kiện" sang "phiên truy cập" (session).**

Vì sao session mà không phải visitor hay từng event? Hành vi bất thường trong TMĐT (bot quét, click ảo, rapid-fire) thường gói gọn trong **một phiên ngắn**. Phân tích theo từng event thì quá vụn; gộp theo visitor thì trộn lẫn phiên tốt và phiên xấu của cùng một người, làm mờ tín hiệu. Session là mức phân giải hợp lý nhất.

**Cách tách session:** sắp xếp sự kiện theo `visitorid` + `timestamp`, mở session mới khi (a) là event đầu của visitor, hoặc (b) cách event trước **hơn 30 phút**.

### "Bất thường" ở đây nghĩa là gì? (định nghĩa cốt lõi)

Phải nói rõ: **"bất thường" KHÔNG có nghĩa là "đã chứng minh gian lận".** Nó nghĩa là:

> Một phiên truy cập có **mẫu hành vi lệch khỏi cách một người mua sắm bình thường** — lệch đến mức đáng nghi và đáng được gắn cờ để xem lại.

Bất thường chia 3 họ chính:

1. **Dấu hiệu máy/tự động (bot):** thao tác quá nhanh, quét quá nhiều, hoạt động lệch giờ.
2. **Hành vi vô lý về logic mua hàng:** mua mà không thêm giỏ, mua trước cả khi xem, xem rất nhiều mà không bao giờ mua.
3. **Hành vi lặp/cường độ bất thường:** thêm 1 món chục lần, xem 1 món hàng chục lần, giao dịch dồn dập.

**Vì sao là bài toán đa nhãn?** Một phiên có thể dính **nhiều loại cùng lúc** (ví dụ xem 40 sản phẩm trong 2 phút lúc 3 giờ sáng không mua → dính BR01 bot + BR03 click fraud + BR05 night crawler). Vì thế mỗi mô hình học dự đoán **đồng thời 12 nhãn loại**, không phải một cờ nhị phân.

**Câu trả lời mẫu "tại sao không học máy thuần mà phải dùng luật":**
"Vì không có nhãn thật. Luật cho phép bọn em (1) tạo nhãn có thể giải thích được, và (2) gắn mỗi cảnh báo với một lý do nghiệp vụ cụ thể, thay vì một con số khó diễn giải."

---

## PHẦN 3 — HỆ THỐNG 12 LUẬT NGHIỆP VỤ

Mỗi luật là một định nghĩa đo được cho một kiểu bất thường (và là **một nhãn loại** để mô hình học). Đây là **multi-label**: một phiên có thể dính nhiều luật. **Một phiên bị coi là bất thường nếu vi phạm ít nhất một luật.** Tổng **62.894 phiên bất thường (3,57%)**.

| Mã | Loại | Hiểu nôm na | Ngưỡng cụ thể |
|---|---|---|---|
| BR01 | Bot scraper | Quét trang như máy | `total_events ≥ 12` **hoặc** `events_per_minute ≥ 4` (& đủ event) |
| BR02 | Ghost buyer | Mua mà chưa từng bỏ giỏ | `n_transaction > 0` & `n_addtocart == 0` |
| BR03 | Click fraud | Xem nhiều, không mua/thêm giỏ | `n_view ≥ 6`, `n_addtocart == 0`, `n_transaction == 0` |
| BR04 | Rapid-fire | Bấm liên tiếp nhanh bất thường | `rapid_ratio ≥ 0,20` (& đủ event) |
| BR05 | Night crawler | Hoạt động chủ yếu nửa đêm | `night_ratio ≥ 0,75` & `total_events ≥ 4` |
| BR06 | Item hoarding | Nhồi cùng 1 món vào giỏ | `max_same_item_atc > 1` |
| BR07 | Session bomb | Mở/xem hàng loạt sản phẩm khác nhau | `unique_items ≥ 12` |
| BR08 | Sequence violation | Mua trước khi xem/thêm giỏ | giao dịch xuất hiện trước view/addtocart cùng item |
| BR09 | Transaction burst | Mua dồn dập | `n_transaction ≥ 3` |
| BR10 | Cart abandonment | Chất đầy giỏ rồi bỏ | `n_addtocart ≥ 4`, `n_transaction == 0` |
| BR11 | Repeated view spam | Xem đi xem lại 1 món | `max_same_item_view ≥ 6` |
| BR12 | Category scanning | Lùng sục khắp danh mục | `unique_categories ≥ 10` |

### Cơ sở chọn ngưỡng (phần ăn điểm nếu bị hỏi "ngưỡng lấy ở đâu")

Tất cả ngưỡng cường độ được chọn dựa trên **phân vị (percentile) thực nghiệm** của toàn bộ 1,76 triệu session, không phải số tùy ý — mỗi luật nhắm vào ~top 0,1–1% hành vi cực đoan nhất nhưng vẫn đủ mẫu dương để mô hình học.

- **Ngưỡng tách session 30 phút:** chuẩn de-facto của web analytics, bắt nguồn từ **Catledge & Pitkow (1995)**.
- **Ngưỡng logic nghiệp vụ** (BR02, BR06, BR08, các điều kiện `== 0`): đúng/sai theo bản chất hành vi, không cần biện minh thống kê.
- **Ngưỡng cường độ** (BR01, BR03, BR04, BR05, BR07, BR09–BR12): đặt ở **đuôi trên của phân bố** (p98–p99,9). Cách tiếp cận dùng nhiều luật heuristic để sinh nhãn chính là **weak supervision / data programming** (Ratner et al., **Snorkel**, VLDB 2017).
- **Ba ngưỡng đã được hiệu chỉnh** (so với bản trước) vì đặt quá cao tạo "lớp chết": **BR10** `n_addtocart` 10 → **4**, **BR11** `max_same_item_view` 20 → **6**, **BR12** `unique_categories` 20 → **10** (tất cả về quanh p99,9). Nhờ vậy các lớp này tăng từ vài chục–vài trăm session lên >1.500 session, đủ để mô hình học. `events_per_minute` của BR01 cũng hạ 8 → 4 vì mốc 8 gần như không bao giờ đạt.

*Tham khảo: Catledge & Pitkow (1995); Tukey (1977); Ratner et al. (2017), Snorkel, VLDB.*

### Phân bố loại bất thường (multi-label, tính trên toàn bộ session)

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

---

## PHẦN 4 — CÁCH LÀM (PIPELINE & KIỂM SOÁT CHẤT LƯỢNG)

Trình bày theo 6 bước:

1. **Tiền xử lý:** đọc `events.csv`, bỏ trùng lặp, giữ event hợp lệ, đổi `timestamp` sang `datetime`.
2. **Tạo session profile:** gom theo `session_id`, tính ~37 đặc trưng (tần suất, độ đa dạng, thời gian/nhịp, tốc độ thao tác, hành vi lặp, tỉ lệ hành vi).
3. **Gắn nhãn đa nhãn:** áp 12 luật để sinh ma trận nhãn 12 loại (pseudo-label).
4. **Chia dữ liệu theo nhóm visitor:** **Train/Validation/Test = 60/20/20** (1.056.863 / 352.486 / 352.326 session), dùng toàn bộ ~1,76 triệu session.
5. **Huấn luyện 5 mô hình giám sát đa nhãn:** XGBoost (chính), LightGBM, Random Forest, Decision Tree, Extra Trees — tất cả trên `safe_features`.
6. **Đánh giá & xuất báo cáo:** chọn ngưỡng riêng từng nhãn trên validation, đo trên test, xuất CSV.

### BA LỚP KIỂM SOÁT CHẤT LƯỢNG (điểm khác biệt của nhóm — nhấn mạnh)

- **Lớp 1 — Tách `safe_features` và `full_features`.** `safe_features` (15 đặc trưng) **loại bỏ mọi biến trực tiếp tạo ra luật**, dùng làm kết quả chính. `full_features` (37 đặc trưng) chứa cả biến tạo luật, chỉ để minh họa "mô hình chép lại luật" (rule-mimic). → Tránh label leakage.
- **Lớp 2 — Group split theo `visitorid`** (GroupShuffleSplit). Chia sao cho **không visitor nào xuất hiện ở hai tập**. Kiểm tra tự động xác nhận **0 visitor trùng** giữa train/val/test. → Tránh rò rỉ ở mức người dùng.
- **Lớp 3 — Shuffle-label sanity check.** Train lại với nhãn xáo trộn ngẫu nhiên; nếu pipeline đúng thì điểm phải tụt mạnh. Kết quả: macro-F1 tụt còn **0,08** → xác nhận pipeline không lỗi.

### SỐ LƯỢNG ĐẶC TRƯNG & ĐẶC TRƯNG DÙNG ĐỂ TRAIN

- **37** đặc trưng số tạo ra cho mỗi session (`full_feature_cols`) — chỉ dùng cho bảng phụ minh họa leakage.
- **15** đặc trưng **dùng để TRAIN** cả 5 mô hình báo cáo chính (`safe_feature_cols`).
- **22** đặc trưng còn lại bị loại khỏi train vì là biến trực tiếp tạo luật (label leakage).

**15 đặc trưng huấn luyện (`safe_feature_cols`):** `session_duration_sec`, `active_hours`, `unique_event_types`, `event_type_entropy`, `hour_entropy`, `mean_interval_sec`, `median_interval_sec`, `std_interval_sec`, `max_interval_sec`, `peak_events`, `weekend_events`, `peak_ratio`, `weekend_ratio`, `unique_parent_categories`, `session_dayofweek`. Toàn bộ là đặc trưng về **thời gian, nhịp thao tác và bối cảnh** — không phải biến đếm cấu thành luật.

**22 đặc trưng bị loại (chỉ có trong `full_features`):** `total_events`, `unique_items`, `n_view`, `n_addtocart`, `n_transaction`, `events_per_minute`, `duration_min`, `min_interval_sec`, `rapid_fire_count`, `rapid_ratio`, `night_events`, `night_ratio`, `max_same_item_view`, `max_same_item_atc`, `unique_categories`, `view_rate`, `atc_rate`, `buy_rate`, `items_per_event`, `view_to_cart_ratio`, `cart_to_transaction_ratio`, `session_start_hour`.

### Cấu trúc mô hình đa nhãn (chung cho 5 thành viên)

XGBoost, LightGBM, Decision Tree được bọc trong `MultiOutputClassifier` (One-vs-Rest: 1 bộ con cho mỗi loại); Random Forest và Extra Trees hỗ trợ đa nhãn gốc (truyền nhãn 2D). Với **mỗi trong 12 nhãn**, chọn ngưỡng tối ưu F1 trên validation rồi cố định áp lên test, có ràng buộc chống over-predict.

### Cấu hình XGBoost (phần nhóm trưởng)

`objective='binary:logistic'`, `tree_method='hist'`; `max_depth=4` (cây nông chống overfit); `learning_rate=0.08` + `n_estimators=120` (học từ từ); `subsample=0.85`, `colsample_bytree=0.85` (lấy mẫu ngẫu nhiên tăng tổng quát). Mất cân bằng xử lý bằng **chọn ngưỡng riêng cho từng nhãn** (không dùng `scale_pos_weight` toàn cục như bản nhị phân cũ).

### XỬ LÝ MẤT CÂN BẰNG DỮ LIỆU (chỉ ~3,6% là bất thường, từng loại còn hiếm hơn)

Nhóm xử lý ở **4 tầng**, không dùng oversampling tổng hợp:

1. **Cân bằng ở mức thuật toán (trọng số lớp):** Decision Tree dùng `class_weight='balanced'`; Random Forest và Extra Trees dùng `class_weight='balanced_subsample'` — tự động tăng trọng số cho lớp hiếm theo tần suất.
2. **Lấy mẫu phân tầng / chia nhóm theo visitor** để giữ tỉ lệ anomaly giữa train/validation/test xấp xỉ nhau (~3,5–3,6%).
3. **Chọn ngưỡng quyết định RIÊNG cho từng nhãn:** không cắt mặc định 0,5 mà quét ngưỡng tối ưu F1 trên validation cho mỗi trong 12 loại, có chặn over-predict (`max_pos_rate_multiple=3.0`) → bù trực tiếp cho lệch lớp ở từng loại. Đây là cơ chế chính cho boosting.
4. **Chọn metric phù hợp lớp hiếm & đa nhãn:** bỏ accuracy, dùng **macro-F1 / micro-F1 / macro-PR-AUC** (xem Phần 5).

**Vì sao KHÔNG dùng SMOTE / oversampling tổng hợp?** (1) trọng số lớp + ngưỡng riêng từng nhãn đã đủ hiệu quả và rẻ hơn; (2) SMOTE tạo session "nhân tạo" bằng nội suy, có thể sinh mẫu hành vi phi thực tế làm méo tín hiệu thời gian/nhịp vốn rất nhạy của bài toán. Là lựa chọn có chủ đích.

---

## PHẦN 5 — ĐÁNH GIÁ KẾT QUẢ

**Vì sao KHÔNG dùng accuracy làm metric chính:** dữ liệu mất cân bằng (~3,6% bất thường), đoán "tất cả bình thường" đã đạt ~96% accuracy nhưng vô dụng. → Dùng **macro-F1, micro-F1, macro-PR-AUC**. *macro-F1* trung bình F1 của 12 loại (coi mọi loại quan trọng như nhau, làm nổi loại hiếm); *micro-F1* gộp toàn bộ cặp (session, loại) nên thiên về loại phổ biến.

### Bảng kết quả chính — tập TEST, `safe_features`

*(Số chuẩn từ `group_model_metrics.csv`.)*

| Mô hình (thành viên) | macro-F1 | micro-F1 | macro-PR-AUC |
|---|--:|--:|--:|
| Dummy baseline | 0,004 | 0,011 | 0,005 |
| **LightGBM (Thủy)** | **0,642** | **0,816** | **0,659** |
| **XGBoost (Đình Tuấn, chính)** | **0,628** | 0,790 | 0,646 |
| Decision Tree (Lê Văn Anh) | 0,536 | 0,718 | 0,550 |
| Random Forest (Tuấn Anh) | 0,361 | 0,251 | 0,417 |
| Extra Trees (Đức Anh) | 0,227 | 0,148 | 0,267 |

**Cách đọc bảng:**

- **Dummy baseline gần như bằng 0** → bài toán thực sự có ý nghĩa học máy, không phải đoán mò.
- **Nhóm gradient boosting (LightGBM, XGBoost) tốt nhất** và sát nhau (macro-F1 0,642 vs 0,628). XGBoost là mô hình chính theo phân công.
- **Random Forest & Extra Trees over-predict trên safe features:** gắn cờ quá nhiều → precision thấp → micro-F1 sụp (RF 0,251; ET 0,148). Đây là hạn chế đã ghi nhận, không phải lỗi pipeline.

### F1 theo TỪNG LOẠI bất thường — test (`group_multilabel_type_metrics.csv`)

| Loại | XGBoost | LightGBM | Decision Tree | Random Forest | Extra Trees | support |
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

Loại có tín hiệu hành vi rõ (Click fraud, Bot scraper, Category scanning) đạt ~0,9 ngay trên safe features; loại khó tách bằng đặc trưng an toàn (Ghost buyer, Sequence violation) F1 thấp — hợp lý vì bản chất luật của chúng nằm ở biến đếm đã bị loại.

### XGBoost — phần nhóm trưởng

| Tập | macro-F1 | micro-F1 | macro-PR-AUC |
|---|--:|--:|--:|
| Validation | 0,621 | 0,792 | 0,639 |
| Test | 0,628 | 0,790 | 0,646 |

Khoảng cách validation–test rất nhỏ → mô hình **ổn định, chưa overfit rõ rệt**.

**Đặc trưng quan trọng nhất** (feature importance): các biến thời gian/nhịp thao tác đứng đầu — `std_interval_sec` (độ lệch khoảng cách giữa event), `session_duration_sec`, `mean_interval_sec`. Hợp lý vì bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

### Bằng chứng kiểm soát chất lượng (rất quan trọng khi bị "vặn")

- **`full_features` (chứa biến tạo luật):** macro-F1 test bùng lên gần tuyệt đối (XGBoost 0,982, LightGBM 0,983, micro-F1 ~0,997) → đây CHÍNH LÀ minh chứng leakage; vì vậy **không** dùng bảng này làm kết luận.
- **Shuffle-label:** XGBoost macro-F1 test tụt còn **0,082** (micro-F1 0,145) → đúng kỳ vọng, pipeline lành mạnh.

---

## PHẦN 6 — ĐIỂM YẾU & CÁCH BẢO VỆ (HỌC KỸ)

**1. Hạn chế nền tảng — nhãn là pseudo-label.** Metric đo "mức khớp với luật" chứ chưa khẳng định bắt được gian lận thật. → *Trả lời:* "Đây là giới hạn khi không có nhãn thật; bọn em đã nêu rõ và đề xuất thu thập nhãn thật để đánh giá độc lập."

**2. macro-F1 trên `safe_features` ~0,63 — đã hết leakage chưa?** Câu hỏi sắc nhất. → *Trả lời trung thực:* "Bọn em đã loại các biến trực tiếp tạo luật. Tuy nhiên các đặc trưng thời gian/nhịp còn lại vẫn tương quan gián tiếp, nên một phần tín hiệu luật vẫn được học gián tiếp. Khoảng cách full (~0,98) → safe (~0,63) cho thấy đã giảm leakage đáng kể nhưng chưa loại bỏ hoàn toàn."

**3. XGBoost là "mô hình chính" nhưng LightGBM điểm cao hơn (0,642 > 0,628).** → *Trả lời:* "Cả hai cùng họ gradient boosting, kết quả sát nhau, sự tương đồng củng cố độ tin cậy. Nhóm chọn XGBoost làm mô hình chính vì phân công và độ phổ biến/ổn định, không phải vì nó cao nhất tuyệt đối."

**4. Random Forest & Extra Trees điểm thấp.** → *Trả lời:* "Hai mô hình rừng over-predict trên safe features — gắn cờ quá nhiều nên precision thấp, kéo micro-F1 xuống. Nguyên nhân là `class_weight='balanced_subsample'` cộng với chọn ngưỡng theo F1 đẩy nhiều dự đoán dương. Hướng khắc phục là calibration hoặc siết ràng buộc tỉ lệ dương."

**5. Chưa đánh giá theo thời gian (drift).** → *Hướng phát triển:* time-based split mô phỏng production.

---

## PHẦN 7 — HƯỚNG PHÁT TRIỂN

1. **Time-based split** để mô phỏng drift trong môi trường thật.
2. **Calibration score** (Platt/Isotonic), đặc biệt cho Random Forest/Extra Trees để giảm over-predict.
3. **Tối ưu ngưỡng theo chi phí nghiệp vụ** (ưu tiên recall hay precision tùy chi phí cảnh báo sai).
4. **Phân tích độ nhạy ngưỡng luật** + thay phép gộp OR bằng *label model* (Snorkel) học trọng số từng luật.
5. **Thu thập nhãn thật** để đánh giá ngoài pseudo-label — quan trọng nhất.

---

## PHẦN 8 — KỊCH BẢN TRÌNH BÀY (5–7 phút)

**1. Mở đầu (30 giây).** "Đồ án của nhóm em phát hiện các phiên truy cập có hành vi bất thường trên website TMĐT, dùng dataset RetailRocket. Em là nhóm trưởng, phụ trách mô hình chính là XGBoost."

**2. Bài toán & dữ liệu (1 phút).** "Dataset có ~2,75 triệu sự kiện của 1,4 triệu visitor. Bọn em tách thành 1,76 triệu session bằng ngưỡng nghỉ 30 phút. Vấn đề là dataset **không có nhãn bất thường thật**, nên nhóm xây 12 luật nghiệp vụ để gán nhãn giả đa nhãn, rồi dùng các thuật toán khai phá dữ liệu để học lại và so sánh."

**3. Đặc trưng & luật (1 phút).** "Mỗi session được mô tả bằng đặc trưng về tần suất, độ đa dạng, thời gian, tốc độ thao tác và tỉ lệ hành vi. 12 luật gồm bot scraper, click fraud, night crawler, rapid-fire... và một phiên có thể dính nhiều loại cùng lúc. Tỉ lệ session bất thường ~3,6%."

**4. Phương pháp & kiểm soát chất lượng (1,5 phút — phần ăn điểm).** "Điểm bọn em chú trọng là độ tin cậy của đánh giá, qua ba lớp kiểm soát: tách `safe_features` (15 đặc trưng, không chứa thứ tạo ra luật) để báo cáo chính, `full_features` (37) chỉ minh họa học lại luật; chia dữ liệu **theo nhóm visitor** để cùng một người không lọt vào cả train lẫn test; kiểm tra shuffle-label — train với nhãn xáo trộn, điểm phải tụt mạnh để xác nhận pipeline không lỗi."

**5. Kết quả & phần XGBoost (1,5 phút).** "Bài toán là đa nhãn 12 loại nên bọn em đánh giá bằng macro/micro-F1. Nhóm gradient boosting (LightGBM, XGBoost) cho kết quả tốt nhất trên `safe_features` (macro-F1 ~0,63–0,64). XGBoost em cấu hình max_depth=4, learning_rate=0.08, bọc trong MultiOutputClassifier, chọn ngưỡng tối ưu F1 riêng cho từng loại trên validation rồi mới áp lên test. Đặc trưng quan trọng nhất là độ lệch và khoảng cách thời gian giữa các thao tác — phù hợp logic phát hiện bot."

**6. Kết luận (30 giây).** "Đồ án dựng được quy trình hoàn chỉnh Rule → Model → Đánh giá → Export. Hạn chế là nhãn vẫn là pseudo-label nên metric phản ánh độ khớp với luật; hướng phát triển là time-based split, calibration ngưỡng, và thu thập nhãn thật."

---

## PHẦN 9 — NGÂN HÀNG CÂU HỎI & GỢI Ý TRẢ LỜI

### Nhóm 1 — Bản chất bài toán (hay hỏi nhất)

**Q1. Nhãn ở đâu ra? Mô hình có phải chỉ học lại đúng luật các em đặt không?**
Đúng là rủi ro lớn và nhóm đã xử lý có chủ đích. Nhãn là pseudo-label đa nhãn sinh từ 12 luật. Nếu cho mô hình học bằng chính các đặc trưng tạo ra luật thì nó sẽ "chép" lại luật và metric đẹp giả tạo. Vì vậy bọn em tách `safe_features` — bỏ các đặc trưng trực tiếp cấu thành luật — và chỉ dùng bộ này để báo cáo. Bọn em cũng để bảng `full_features` riêng để cho thấy rõ chênh lệch do leakage.

**Q2. Kết quả có ý nghĩa thực tế không khi không có nhãn thật?**
Em xin nhận đây là hạn chế nền tảng. Metric hiện tại đo mức độ mô hình tổng quát hóa được mẫu hành vi bất thường mà luật mô tả, chứ chưa khẳng định bắt được gian lận thật. Để đánh giá thật cần nhãn do con người gán hoặc dữ liệu có ground-truth — đây là hướng phát triển.

**Q3. Tại sao phân tích theo session mà không theo visitor?**
Hành vi bất thường (bot, rapid-fire, click fraud) thường xảy ra gọn trong một phiên ngắn. Gộp theo visitor sẽ trộn lẫn phiên bình thường và bất thường của cùng người, làm mờ tín hiệu. Session-level cho độ phân giải hành vi tốt hơn.

**Q4. Vì sao chọn ngưỡng nghỉ 30 phút?**
30 phút là ngưỡng phổ biến trong phân tích web (chuẩn của nhiều công cụ analytics, bắt nguồn từ Catledge & Pitkow 1995). Nghỉ quá lâu thì coi như phiên mới. Đây là tham số có thể điều chỉnh.

**Q4b. Vì sao đây là bài toán đa nhãn chứ không nhị phân?**
Vì một phiên có thể vi phạm nhiều luật cùng lúc (vd bot + click fraud + night crawler). Bọn em muốn mô hình không chỉ nói "có bất thường" mà còn nói **bất thường loại gì**, nên mỗi mô hình dự đoán đồng thời 12 nhãn loại.

### Nhóm 2 — Phương pháp đánh giá

**Q5. Vì sao không dùng accuracy làm metric chính?**
Dữ liệu mất cân bằng nặng (~3,6% bất thường), từng loại còn hiếm hơn. Đoán "tất cả bình thường" đã đạt ~96% accuracy nhưng vô dụng. Nên dùng macro-F1, micro-F1 và macro-PR-AUC.

**Q5b. Phân biệt macro-F1 và micro-F1?**
macro-F1 tính F1 cho từng loại rồi lấy trung bình — coi mọi loại quan trọng như nhau nên làm nổi các loại hiếm. micro-F1 gộp toàn bộ cặp (session, loại) rồi tính một F1 chung — bị chi phối bởi các loại phổ biến. Bọn em báo cáo cả hai.

**Q5c. Các em xử lý mất cân bằng dữ liệu như thế nào?**
Bốn tầng: (1) trọng số lớp ở mức thuật toán — `class_weight='balanced'` cho Decision Tree, `'balanced_subsample'` cho Random Forest và Extra Trees; (2) chia nhóm theo visitor giữ tỉ lệ ~3,6% ở mọi tập; (3) **chọn ngưỡng tối ưu F1 RIÊNG cho từng nhãn** trên validation (có chặn over-predict) thay vì cắt mặc định 0,5 — đây là cơ chế chính cho XGBoost/LightGBM; (4) chọn metric macro/micro-F1 + PR-AUC. Bọn em không dùng SMOTE vì có thể làm méo tín hiệu thời gian/nhịp.

**Q6. Tại sao chia dữ liệu theo nhóm visitor (group split)?**
Vì session sinh ra từ visitor. Nếu chia ngẫu nhiên theo session, các session của cùng visitor có thể nằm cả ở train lẫn test, khiến mô hình "thấy trước" người đó và metric cao giả. Group split theo `visitorid` đảm bảo không visitor nào xuất hiện ở nhiều tập — có kiểm tra tự động xác nhận 0 trùng lặp.

**Q7. Vì sao chọn ngưỡng trên validation chứ không trên test?**
Dò ngưỡng tốt nhất ngay trên test là gian lận thông tin — test không còn khách quan. Bọn em chọn ngưỡng tối ưu F1 (riêng từng nhãn) trên validation, rồi áp cố định lên test để đo trung thực.

**Q8. Shuffle-label sanity check là gì, để làm gì?**
Train lại mô hình với nhãn bị xáo trộn ngẫu nhiên. Nếu pipeline đúng, mô hình không học được gì và điểm phải tụt mạnh. Kết quả của bọn em tụt mạnh (macro-F1 còn 0,08) đúng kỳ vọng.

**Q9. Tại sao bảng full_features cho macro-F1 gần tuyệt đối?**
Đó chính là minh chứng leakage: full_features chứa các biến trực tiếp tạo ra luật, nên mô hình chỉ việc tái hiện luật. Bọn em cố tình để bảng này để so sánh, và KHÔNG dùng nó làm kết luận.

### Nhóm 3 — Phần XGBoost (nhóm trưởng)

**Q10. Vì sao chọn XGBoost làm mô hình chính?**
XGBoost xử lý tốt dữ liệu dạng bảng, quan hệ phi tuyến, có regularization chống overfit và cho feature importance để giải thích. Trên `safe_features` nó thuộc nhóm cho kết quả tốt nhất (sát LightGBM).

**Q11. Các siêu tham số chính có ý nghĩa gì?**
`max_depth=4` giữ cây nông tránh overfit; `learning_rate=0.08` với `n_estimators=120` học từ từ, ổn định; `subsample=0.85` và `colsample_bytree=0.85` lấy mẫu ngẫu nhiên dòng/cột tăng tổng quát. Bọc trong `MultiOutputClassifier` để ra một bộ XGBoost cho mỗi loại.

**Q12. Đa nhãn rồi thì xử lý mất cân bằng ở đâu, sao không thấy `scale_pos_weight`?**
Bản nhị phân cũ dùng một `scale_pos_weight` chung. Bản đa nhãn này mỗi loại có tỉ lệ dương khác nhau, nên bọn em chuyển sang **chọn ngưỡng quyết định riêng cho từng nhãn** trên validation — linh hoạt hơn và bù lệch lớp theo từng loại.

**Q13. Đặc trưng nào quan trọng nhất và vì sao hợp lý?**
Các đặc trưng thời gian/nhịp thao tác đứng đầu (độ lệch khoảng cách giữa event, độ dài session, khoảng cách trung bình). Hợp lý vì bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

**Q14. So với LightGBM thì sao, cả hai đều là boosting?**
Cả hai cùng họ gradient boosting và cho kết quả sát nhau (LightGBM macro-F1 0,642, XGBoost 0,628). LightGBM nhanh hơn nhờ leaf-wise growth, XGBoost ổn định và phổ biến hơn. Kết quả tương đồng củng cố độ tin cậy.

### Nhóm 4 — So sánh mô hình

**Q15. Vì sao Random Forest và Extra Trees điểm thấp hơn nhiều?**
Hai mô hình rừng over-predict trên safe features: chúng gắn cờ quá nhiều session nên precision rất thấp, kéo micro-F1 xuống (RF 0,251; ET 0,148). Nguyên nhân là kết hợp `class_weight='balanced_subsample'` với cơ chế chọn ngưỡng theo F1. Đây là hành vi đã ghi nhận, hướng khắc phục là calibration hoặc siết ràng buộc tỉ lệ dương — không phải lỗi pipeline.

**Q15b. Extra Trees khác Random Forest ở điểm gì?**
Extra Trees (Extremely Randomized Trees) ngẫu nhiên hoá cả việc chọn đặc trưng lẫn ngưỡng cắt tại mỗi nút, thay vì tìm ngưỡng tối ưu như Random Forest. Nhờ vậy phương sai thấp hơn và train nhanh hơn. Bọn em dùng nó (Đức Anh phụ trách) để đối chiếu với Random Forest trên dữ liệu bảng.

**Q16. Decision Tree dùng làm gì khi đã có XGBoost?**
Decision Tree dễ giải thích trực quan bằng cây quyết định, phù hợp trình bày logic phân loại cho người không chuyên. XGBoost mạnh hơn về hiệu năng nhưng khó "đọc" trực tiếp.

### Nhóm 5 — Hạn chế & mở rộng

**Q17. Hạn chế lớn nhất của đồ án?**
Nhãn là pseudo-label từ luật, nên metric đo độ khớp với luật chứ chưa phải bắt gian lận thật. Ngoài ra Random Forest/Extra Trees còn over-predict, và chưa đánh giá theo thời gian (drift).

**Q18. Nếu phát triển tiếp, các em làm gì?**
Time-based split mô phỏng production; calibration score (Platt/Isotonic) cho RF/ET; tối ưu ngưỡng theo chi phí nghiệp vụ; và quan trọng nhất là thu thập nhãn thật để đánh giá độc lập.

**Q19. Làm sao biết các luật đặt ngưỡng hợp lý?**
Ngưỡng cường độ dựa trên phân vị thực nghiệm của chính dữ liệu (p98–p99,9) chứ không tùy ý; bọn em đã hiệu chỉnh ba ngưỡng (BR10/BR11/BR12) vốn đặt quá cao tạo "lớp chết". Đây là điểm có thể tinh chỉnh tiếp; lý tưởng là hiệu chỉnh theo phản hồi chuyên gia hoặc nhãn thật.

**Q20. Đóng góp cụ thể của từng thành viên?**
Mỗi người phụ trách một thuật toán giám sát đa nhãn: em (nhóm trưởng) làm XGBoost và điều phối chung; Lê Văn Anh — Decision Tree; Tuấn Anh — Random Forest; Thủy — LightGBM; Đức Anh — Extra Trees. Tiền xử lý, xây luật và đánh giá làm chung.

### Câu "bẫy" & cách giữ bình tĩnh

- Quên số liệu cụ thể → mở `group_model_metrics.csv` hoặc bảng trong notebook, **đừng đoán bừa**.
- Bị chỉ ra điểm yếu → **thừa nhận thẳng** rồi nối sang hướng cải tiến.
- Bị hỏi "kết quả cao thế tin được không" → dẫn ngay **ba lớp kiểm soát** (safe_features, group split, shuffle-label).
- Câu chốt an toàn: *"Đây là hạn chế bọn em đã nhận thức được và nêu trong phần hướng phát triển."*

---

## PHẦN 10 — PHÂN CÔNG NHÓM

| Thành viên | Thuật toán | Loại | Vai trò |
|---|---|---|---|
| **Đình Tuấn (nhóm trưởng)** | **XGBoost** | Có giám sát (đa nhãn) | **Mô hình chính + điều phối** |
| Lê Văn Anh | Decision Tree | Có giám sát (đa nhãn) | Dễ giải thích trực quan |
| Tuấn Anh | Random Forest | Có giám sát (đa nhãn) | Ensemble + feature importance |
| Thủy | LightGBM | Có giám sát (đa nhãn) | Đối chiếu với XGBoost |
| Đức Anh | Extra Trees | Có giám sát (đa nhãn) | Ensemble cây ngẫu nhiên hoá, đối chiếu Random Forest |

*Tiền xử lý, xây luật và đánh giá: làm chung.*

---

## PHẦN 11 — TỆP ĐẦU RA KÈM THEO

- `group_model_metrics.csv` — bảng metric đa nhãn chính (nguồn số liệu chuẩn).
- `group_multilabel_type_metrics.csv` — F1 theo từng loại bất thường của 5 mô hình.
- `group_leakage_audit.csv` — kiểm toán biến nào bị loại khỏi `safe_features`.
- `group_anomaly_type_breakdown.csv` — phân bố loại bất thường.
- `session_anomaly_tags.csv` — nhãn luật đa nhãn từng session.
- `session_anomaly_report.csv` — báo cáo tổng hợp rule + model.
- Notebook: `anomaly_detection_group_project.ipynb`.
