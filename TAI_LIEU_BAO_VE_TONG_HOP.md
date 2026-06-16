# TÀI LIỆU BẢO VỆ ĐỒ ÁN (TỔNG HỢP)

## Phát hiện hành vi bất thường theo session — RetailRocket E-commerce

> File này gộp toàn bộ ý chuẩn từ 4 tài liệu cũ (hướng dẫn bảo vệ, kịch bản trình bày, báo cáo giáo viên, slide). **Mọi số liệu lấy trực tiếp từ `group_model_metrics.csv`** (đã chạy group split theo `visitorid`). Đọc file này là đủ — các file còn lại chỉ để tham khảo.
>
> Cấu trúc: Phần 1–7 là kiến thức để học; Phần 8 là kịch bản nói; Phần 9 là ngân hàng câu hỏi; Phần 10–11 là phụ lục.

---

## PHẦN 0 — Tóm tắt mở đầu (học thuộc)

"Đồ án phát hiện các phiên truy cập (session) có hành vi đáng nghi trên website thương mại điện tử, dùng dataset RetailRocket. Vì dataset **không có nhãn bất thường thật**, nhóm xây 12 luật nghiệp vụ để sinh nhãn giả (pseudo-label), rồi huấn luyện 5 thuật toán khai phá dữ liệu để học lại và đánh giá. Quy trình: **Luật → Mô hình → Đánh giá → Xuất báo cáo.** Em là nhóm trưởng, phụ trách mô hình chính XGBoost."

---

## PHẦN 0B — MỤC TIÊU, ỨNG DỤNG & VÍ DỤ

### Mục tiêu của đồ án

**Mục tiêu chung:** xây dựng một quy trình hoàn chỉnh, có thể tin cậy được, để **tự động phát hiện những phiên truy cập có hành vi đáng nghi** trên website thương mại điện tử — dựa hoàn toàn trên log hành vi, không cần con người gắn nhãn từng phiên.

Cụ thể đồ án nhắm tới 4 mục tiêu:

1. **Biến dữ liệu thô thành đơn vị phân tích có nghĩa:** chuyển ~2,75 triệu sự kiện rời rạc thành 1,76 triệu session, mỗi session là một "chân dung hành vi" mô tả được bằng ~37 đặc trưng.
2. **Định nghĩa được "bất thường" một cách đo lường được và giải thích được:** vì không có nhãn thật, nhóm mã hóa kinh nghiệm nghiệp vụ thành 12 luật để sinh nhãn giả — mỗi cảnh báo luôn kèm một lý do cụ thể (ví dụ "bot scraper", "click fraud").
3. **So sánh và chọn được mô hình khai phá dữ liệu phù hợp** để học lại các mẫu bất thường đó (5 thuật toán: XGBoost, LightGBM, Random Forest, Decision Tree, Isolation Forest).
4. **Đảm bảo kết quả đáng tin, không bị "ảo":** qua ba lớp kiểm soát chất lượng (safe_features, group split, shuffle-label) để chống label leakage và over-fitting.

> Lưu ý phạm vi: mục tiêu là **gắn cờ phiên đáng nghi để xem lại**, KHÔNG phải khẳng định chắc chắn đó là gian lận. Đây là bước sàng lọc đầu tiên trong một hệ thống chống gian lận.

### Ứng dụng thực tế

Quy trình này có thể dùng làm **lớp sàng lọc tự động (screening)** trong nhiều bài toán vận hành sàn TMĐT:

- **Chống gian lận & lạm dụng:** phát hiện tài khoản/phiên có dấu hiệu click fraud, đặt hàng ảo, tấn công thử thẻ (card testing) để chuyển sang đội rủi ro xem xét.
- **Chặn bot & cào dữ liệu (scraping):** nhận diện phiên quét giá/sản phẩm tốc độ cao của đối thủ hoặc bot, để giới hạn truy cập (rate-limit) hoặc thêm captcha.
- **Bảo vệ chất lượng dữ liệu phân tích:** lọc các phiên bot/bất thường ra khỏi báo cáo lượt xem, tỉ lệ chuyển đổi... để số liệu marketing không bị méo.
- **Bảo vệ tồn kho & giá:** phát hiện hành vi gom giỏ hàng bất thường (item hoarding, cart bombing) làm khóa hàng ảo.
- **Cảnh báo vận hành theo thời gian thực:** vì mô hình cho ra điểm số cho từng phiên, có thể gắn ngưỡng để bắn cảnh báo ngay khi một phiên vượt mức đáng nghi.

Điểm mạnh để ứng dụng: kết quả **giải thích được** (mỗi cờ gắn với một luật), nên đội vận hành biết *vì sao* một phiên bị gắn cờ chứ không chỉ nhận một con số.

### Ví dụ cụ thể

**Ví dụ 1 — Bot cào dữ liệu (BR01 + BR12).** Một phiên xem 80 sản phẩm thuộc 25 danh mục khác nhau trong vòng 3 phút, nhịp thao tác đều tăm tắp ~0,2 giây/lần. Người thật không bấm đều và nhanh như vậy → mô hình gắn cờ "bot scraper" + "category scanning". *Ứng dụng:* chặn IP/giới hạn tốc độ truy cập.

**Ví dụ 2 — Click fraud lúc nửa đêm (BR03 + BR05).** Một phiên lúc 3 giờ sáng xem 30 lượt sản phẩm nhưng không thêm giỏ, không mua, lặp lại nhiều ngày. → gắn cờ "click fraud" + "night crawler". *Ứng dụng:* lọc khỏi báo cáo lượt xem quảng cáo, điều tra gian lận hiển thị.

**Ví dụ 3 — Đặt hàng ảo / thử thẻ (BR02 + BR09).** Một phiên có 5 giao dịch liên tiếp mà chưa từng thêm sản phẩm vào giỏ. → gắn cờ "ghost buyer" + "transaction burst". *Ứng dụng:* chuyển sang đội rủi ro kiểm tra thẻ thanh toán.

**Ví dụ 4 — Người mua bình thường (KHÔNG bị gắn cờ).** Một phiên 15 phút: xem 8 sản phẩm với nhịp tay người (vài giây đến vài chục giây/lần), thêm 2 món vào giỏ, mua 1 món, vào lúc 8 giờ tối. → không vi phạm luật nào → mô hình xếp là bình thường. Ví dụ này dùng để cho thấy luật **không gắn cờ bừa** người dùng thật.

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
2. **Mất cân bằng nặng:** chỉ ~3,51% session bị gắn nhãn bất thường (61.849 / 1.761.675). Điều này quyết định cách chọn metric (xem Phần 4).

**Câu trả lời mẫu "dữ liệu gì":**
"Dữ liệu là log hành vi thật của người dùng trên sàn TMĐT, khoảng 2,75 triệu sự kiện của 1,4 triệu người. Bọn em chuyển nó từ mức sự kiện rời rạc sang mức session để phân tích hành vi. Hạn chế lớn nhất là dữ liệu không có nhãn bất thường sẵn."

---

## PHẦN 2 — Ý TƯỞNG XỬ LÝ

**Ý tưởng trung tâm: chuyển từ "sự kiện" sang "phiên truy cập" (session).**

Vì sao session mà không phải visitor hay từng event? Hành vi bất thường trong TMĐT (bot quét, click ảo, rapid-fire) thường gói gọn trong **một phiên ngắn**. Phân tích theo từng event thì quá vụn, không thấy mẫu hành vi; gộp theo visitor thì trộn lẫn phiên tốt và phiên xấu của cùng một người, làm mờ tín hiệu. Session là mức phân giải hợp lý nhất.

**Cách tách session:** sắp xếp sự kiện theo `visitorid` + `timestamp`, mở session mới khi (a) là event đầu của visitor, hoặc (b) cách event trước **hơn 30 phút**.

### "Bất thường" ở đây nghĩa là gì? (định nghĩa cốt lõi)

Phải nói rõ: **"bất thường" KHÔNG có nghĩa là "đã chứng minh gian lận".** Nó nghĩa là:

> Một phiên truy cập có **mẫu hành vi lệch khỏi cách một người mua sắm bình thường** — lệch đến mức đáng nghi và đáng được gắn cờ để xem lại.

Bất thường chia 3 họ chính:

1. **Dấu hiệu máy/tự động (bot):** thao tác quá nhanh, quét quá nhiều, hoạt động lệch giờ.
2. **Hành vi vô lý về logic mua hàng:** mua mà không thêm giỏ, mua trước cả khi xem, xem rất nhiều mà không bao giờ mua.
3. **Hành vi lặp/cường độ bất thường:** thêm 1 món chục lần, xem 1 món hàng chục lần, giao dịch dồn dập.

**Ví dụ minh họa:** một phiên xem 40 sản phẩm trong 2 phút, lúc 3 giờ sáng, không thêm giỏ, không mua → dính BR01 (bot) + BR03 (click fraud) + BR05 (night crawler) cùng lúc → bị gắn cờ bất thường.

**Câu trả lời mẫu "tại sao không học máy thuần mà phải dùng luật":**
"Vì không có nhãn thật. Luật cho phép bọn em (1) tạo nhãn có thể giải thích được, và (2) gắn mỗi cảnh báo với một lý do nghiệp vụ cụ thể, thay vì một con số khó diễn giải."

---

## PHẦN 3 — HỆ THỐNG 12 LUẬT NGHIỆP VỤ

Mỗi luật là một định nghĩa đo được cho một kiểu bất thường. Đây là **multi-label**: một phiên có thể dính nhiều luật. **Một phiên bị coi là bất thường nếu vi phạm ít nhất một luật.** Tổng 61.849 phiên bất thường (3,51%).

| Mã | Loại | Hiểu nôm na | Ngưỡng cụ thể |
|---|---|---|---|
| BR01 | Bot scraper | Quét trang như máy | `total_events ≥ 12` **hoặc** `events_per_minute ≥ 8` |
| BR02 | Ghost buyer | Mua mà chưa từng bỏ giỏ | `n_transaction > 0` & `n_addtocart == 0` |
| BR03 | Click fraud | Xem nhiều, không mua/thêm giỏ | `n_view ≥ 6`, `n_addtocart == 0`, `n_transaction == 0` |
| BR04 | Rapid-fire | Bấm liên tiếp nhanh bất thường | `rapid_ratio ≥ 0,20` (& đủ event) |
| BR05 | Night crawler | Hoạt động chủ yếu nửa đêm | `night_ratio ≥ 0,75` & `total_events ≥ 4` |
| BR06 | Item hoarding | Nhồi cùng 1 món vào giỏ | `max_same_item_atc > 1` |
| BR07 | Session bomb | Mở/xem hàng loạt sản phẩm khác nhau | `unique_items ≥ 12` |
| BR08 | Sequence violation | Mua trước khi xem/thêm giỏ | giao dịch xuất hiện trước view/addtocart cùng item |
| BR09 | Transaction burst | Mua dồn dập | `n_transaction ≥ 3` |
| BR10 | Cart abandonment | Chất đầy giỏ rồi bỏ | `n_addtocart ≥ 10`, `n_transaction == 0` |
| BR11 | Repeated view spam | Xem đi xem lại 1 món | `max_same_item_view ≥ 20` |
| BR12 | Category scanning | Lùng sục khắp danh mục | `unique_categories ≥ 20` |

### Cơ sở chọn ngưỡng (phần ăn điểm nếu bị hỏi "ngưỡng lấy ở đâu")

Cách tiếp cận tổng thể — dùng nhiều luật heuristic để sinh nhãn — chính là **weak supervision / data programming**, mỗi luật là một *labeling function* (Ratner et al., **Snorkel**, VLDB 2017).

- **Ngưỡng tách session 30 phút:** chuẩn de-facto của web analytics, bắt nguồn từ **Catledge & Pitkow (1995)** — khoảng cách trung bình giữa hai thao tác ≈ 9,3 phút, lấy trung bình + 1,5 độ lệch chuẩn ≈ 25,5 phút, làm tròn 30 phút (1.800 giây).
- **Ngưỡng logic nghiệp vụ** (BR02, BR06, BR08, các điều kiện `== 0`): đúng/sai theo bản chất hành vi, không cần biện minh thống kê.
- **Ngưỡng cường độ** (BR01, BR03, BR04, BR05, BR07, BR09–BR12): đặt ở **đuôi trên của phân bố** (p97,5–p99,99) theo nguyên tắc phát hiện outlier (tham khảo quy tắc Tukey 1,5×IQR và phương pháp phân vị). Mọi ngưỡng cường độ chỉ gắn cờ phần rất nhỏ session cực đoan nhất.

*Tham khảo: Catledge & Pitkow (1995); Tukey (1977); Ratner et al. (2017), Snorkel, VLDB.*

### Phân bố loại bất thường (multi-label, tính trên toàn bộ session)

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

---

## PHẦN 4 — CÁCH LÀM (PIPELINE & KIỂM SOÁT CHẤT LƯỢNG)

Trình bày theo 6 bước:

1. **Tiền xử lý:** đọc `events.csv`, bỏ trùng lặp, giữ event hợp lệ, đổi `timestamp` sang `datetime`.
2. **Tạo session profile:** gom theo `session_id`, tính ~37 đặc trưng (tần suất, độ đa dạng, thời gian/nhịp, tốc độ thao tác, hành vi lặp, tỉ lệ hành vi).
3. **Gắn nhãn:** áp 12 luật để sinh pseudo-label.
4. **Lấy mẫu & chia dữ liệu:** lấy 300.000 session (phân tầng), chia **Train/Validation/Test = 60/20/20** (~180k/60k/60k).
5. **Huấn luyện 5 mô hình:** XGBoost (chính), LightGBM, Random Forest, Decision Tree, Isolation Forest.
6. **Đánh giá & xuất báo cáo:** chọn ngưỡng trên validation, đo trên test, xuất CSV.

### BA LỚP KIỂM SOÁT CHẤT LƯỢNG (điểm khác biệt của nhóm — nhấn mạnh)

- **Lớp 1 — Tách `safe_features` và `full_features`.** `safe_features` (15 đặc trưng) **loại bỏ mọi biến trực tiếp tạo ra luật**, dùng làm kết quả chính. `full_features` (37 đặc trưng) chứa cả biến tạo luật, chỉ để minh họa "mô hình chép lại luật" (rule-mimic). → Tránh label leakage.
- **Lớp 2 — Group split theo `visitorid`** (GroupShuffleSplit). Chia sao cho **không visitor nào xuất hiện ở hai tập**. Kiểm tra tự động xác nhận **0 visitor trùng** giữa train/val/test. → Tránh rò rỉ ở mức người dùng.
- **Lớp 3 — Shuffle-label sanity check.** Train lại với nhãn xáo trộn ngẫu nhiên; nếu pipeline đúng thì điểm phải tụt mạnh. Kết quả: F1 tụt còn **0,21** → xác nhận pipeline không lỗi.

### Cấu hình XGBoost (phần nhóm trưởng)

`objective='binary:logistic'`, `tree_method='hist'`; `max_depth=4` (cây nông chống overfit); `learning_rate=0.08` + `n_estimators=120` (học từ từ); `subsample=0.85`, `colsample_bytree=0.85` (lấy mẫu ngẫu nhiên tăng tổng quát); `scale_pos_weight = neg/pos` (bù lớp bất thường hiếm). **Ngưỡng = 0,869**, chọn tối ưu F1 trên validation rồi cố định áp lên test.

---

## PHẦN 5 — ĐÁNH GIÁ KẾT QUẢ

**Vì sao KHÔNG dùng accuracy làm metric chính:** dữ liệu mất cân bằng (~3,5% bất thường), một mô hình đoán "tất cả bình thường" đã đạt ~96,5% accuracy nhưng vô dụng. → Dùng **Precision, Recall, F1, PR-AUC**. Với lớp hiếm, **PR-AUC** đáng tin nhất.

### Bảng kết quả chính — tập TEST, `safe_features`

*(Số chuẩn từ `group_model_metrics.csv`.)*

| Mô hình | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|--:|--:|--:|--:|--:|
| Dummy baseline | 0,041 | 0,040 | 0,040 | 0,503 | 0,035 |
| Decision Tree | 0,684 | 0,775 | 0,727 | 0,987 | 0,747 |
| Random Forest | 0,799 | 0,911 | 0,851 | 0,992 | 0,902 |
| **XGBoost (chính)** | 0,812 | 0,918 | **0,862** | 0,993 | 0,922 |
| **LightGBM** | 0,820 | 0,934 | **0,873** | 0,994 | 0,935 |
| Isolation Forest | 0,291 | 0,674 | 0,406 | 0,935 | 0,358 |

**Cách đọc bảng:**

- **Dummy baseline gần như bằng 0** → bài toán thực sự có ý nghĩa học máy, không phải đoán mò.
- **Nhóm gradient boosting (XGBoost, LightGBM) tốt nhất.** Recall ~0,92–0,93 (bắt được phần lớn session bất thường), precision ~0,81–0,82.
- **Isolation Forest thấp hơn nhiều** vì không dùng nhãn khi train — chỉ là tham chiếu unsupervised, không cạnh tranh trực tiếp.

### XGBoost — phần nhóm trưởng

| Tập | Precision | Recall | F1 |
|---|--:|--:|--:|
| Validation | 0,814 | 0,929 | 0,868 |
| Test | 0,812 | 0,918 | 0,862 |

Khoảng cách validation–test rất nhỏ (F1 0,868 vs 0,862) → mô hình **ổn định, chưa overfit rõ rệt**.

**Đặc trưng quan trọng nhất** (feature importance): các biến thời gian/nhịp thao tác đứng đầu — `std_interval_sec` (độ lệch khoảng cách giữa event), `max_interval_sec`, `session_duration_sec`. Hợp lý vì bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

### Bằng chứng kiểm soát chất lượng (rất quan trọng khi bị "vặn")

- **`full_features` (chứa biến tạo luật):** XGBoost test F1 = **0,994**, PR-AUC ≈ 1,0 → gần tuyệt đối, đây CHÍNH LÀ minh chứng leakage; vì vậy **không** dùng bảng này làm kết luận.
- **Shuffle-label:** XGBoost test F1 tụt còn **0,209**, PR-AUC 0,092 → đúng kỳ vọng, pipeline lành mạnh.

---

## PHẦN 6 — ĐIỂM YẾU & CÁCH BẢO VỆ (HỌC KỸ)

**1. Hạn chế nền tảng — nhãn là pseudo-label.** Metric đo "mức khớp với luật" chứ chưa khẳng định bắt được gian lận thật. → *Trả lời:* "Đây là giới hạn khi không có nhãn thật; bọn em đã nêu rõ và đề xuất thu thập nhãn thật để đánh giá độc lập."

**2. F1 trên `safe_features` vẫn cao (0,86) — đã hết leakage chưa?** Câu hỏi sắc nhất. → *Trả lời trung thực:* "Bọn em đã loại các biến trực tiếp tạo luật. Tuy nhiên các đặc trưng thời gian/nhịp còn lại vẫn tương quan gián tiếp với biến đếm, nên một phần tín hiệu luật vẫn được học gián tiếp. Khoảng cách full (0,99) → safe (0,86) cho thấy đã giảm leakage đáng kể nhưng chưa loại bỏ hoàn toàn. Đây là giới hạn của bài toán dựa trên luật." (Thừa nhận điểm này = ăn điểm phương pháp.)

**3. XGBoost là "mô hình chính" nhưng LightGBM điểm cao hơn (0,873 > 0,862).** → *Trả lời:* "Cả hai cùng họ gradient boosting, kết quả sát nhau, sự tương đồng củng cố độ tin cậy. Nhóm chọn XGBoost làm mô hình chính vì phân công và độ phổ biến/ổn định, không phải vì nó cao nhất tuyệt đối."

**4. Chưa đánh giá theo thời gian (drift).** → *Hướng phát triển:* time-based split mô phỏng production.

---

## PHẦN 7 — HƯỚNG PHÁT TRIỂN

1. **Time-based split** để mô phỏng drift trong môi trường thật.
2. **Calibration score** (Platt/Isotonic) trước khi đặt ngưỡng cảnh báo.
3. **Tối ưu ngưỡng theo chi phí nghiệp vụ** (ưu tiên recall hay precision tùy chi phí cảnh báo sai).
4. **Phân tích độ nhạy ngưỡng luật** + thay phép gộp OR bằng *label model* (Snorkel) học trọng số từng luật.
5. **Thu thập nhãn thật** để đánh giá ngoài pseudo-label — quan trọng nhất.

---

## PHẦN 8 — KỊCH BẢN TRÌNH BÀY (5–7 phút)

**1. Mở đầu (30 giây).** "Đồ án của nhóm em phát hiện các phiên truy cập có hành vi bất thường trên website TMĐT, dùng dataset RetailRocket. Em là nhóm trưởng, phụ trách mô hình chính là XGBoost."

**2. Bài toán & dữ liệu (1 phút).** "Dataset có ~2,75 triệu sự kiện của 1,4 triệu visitor. Bọn em tách thành 1,76 triệu session bằng ngưỡng nghỉ 30 phút. Vấn đề là dataset **không có nhãn bất thường thật**, nên nhóm xây 12 luật nghiệp vụ để gán nhãn giả, rồi dùng các thuật toán khai phá dữ liệu để học lại và so sánh."

**3. Đặc trưng & luật (1 phút).** "Mỗi session được mô tả bằng đặc trưng về tần suất, độ đa dạng, thời gian, tốc độ thao tác và tỉ lệ hành vi. 12 luật gồm bot scraper, click fraud, night crawler, rapid-fire... Tỉ lệ session bất thường ~3,5%."

**4. Phương pháp & kiểm soát chất lượng (1,5 phút — phần ăn điểm).** "Điểm bọn em chú trọng là độ tin cậy của đánh giá, qua ba lớp kiểm soát: tách `safe_features` (15 đặc trưng, không chứa thứ tạo ra luật) để báo cáo chính, `full_features` (37) chỉ minh họa học lại luật; chia dữ liệu **theo nhóm visitor** để cùng một người không lọt vào cả train lẫn test; kiểm tra shuffle-label — train với nhãn xáo trộn, điểm phải tụt mạnh để xác nhận pipeline không lỗi."

**5. Kết quả & phần XGBoost (1,5 phút).** "Nhóm gradient boosting (XGBoost, LightGBM) cho kết quả tốt nhất trên `safe_features`. XGBoost em cấu hình max_depth=4, learning_rate=0.08, dùng `scale_pos_weight` xử lý mất cân bằng, chọn ngưỡng tối ưu F1 trên validation rồi mới áp lên test. Đặc trưng quan trọng nhất là độ lệch và khoảng cách thời gian giữa các thao tác — phù hợp logic phát hiện bot."

**6. Kết luận (30 giây).** "Đồ án dựng được quy trình hoàn chỉnh Rule → Model → Đánh giá → Export. Hạn chế là nhãn vẫn là pseudo-label nên metric phản ánh độ khớp với luật; hướng phát triển là time-based split, calibration ngưỡng, và thu thập nhãn thật."

---

## PHẦN 9 — NGÂN HÀNG CÂU HỎI & GỢI Ý TRẢ LỜI

### Nhóm 1 — Bản chất bài toán (hay hỏi nhất)

**Q1. Nhãn ở đâu ra? Mô hình có phải chỉ học lại đúng luật các em đặt không?**
Đúng là rủi ro lớn và nhóm đã xử lý có chủ đích. Nhãn là pseudo-label sinh từ 12 luật. Nếu cho mô hình học bằng chính các đặc trưng tạo ra luật thì nó sẽ "chép" lại luật và metric đẹp giả tạo. Vì vậy bọn em tách `safe_features` — bỏ các đặc trưng trực tiếp cấu thành luật — và chỉ dùng bộ này để báo cáo. Bọn em cũng để bảng `full_features` riêng để cho thấy rõ chênh lệch do leakage.

**Q2. Kết quả có ý nghĩa thực tế không khi không có nhãn thật?**
Em xin nhận đây là hạn chế nền tảng. Metric hiện tại đo mức độ mô hình tổng quát hóa được mẫu hành vi bất thường mà luật mô tả, chứ chưa khẳng định bắt được gian lận thật. Để đánh giá thật cần nhãn do con người gán hoặc dữ liệu có ground-truth — đây là hướng phát triển.

**Q3. Tại sao phân tích theo session mà không theo visitor?**
Hành vi bất thường (bot, rapid-fire, click fraud) thường xảy ra gọn trong một phiên ngắn. Gộp theo visitor sẽ trộn lẫn phiên bình thường và bất thường của cùng người, làm mờ tín hiệu. Session-level cho độ phân giải hành vi tốt hơn.

**Q4. Vì sao chọn ngưỡng nghỉ 30 phút?**
30 phút là ngưỡng phổ biến trong phân tích web (chuẩn của nhiều công cụ analytics, bắt nguồn từ Catledge & Pitkow 1995). Nghỉ quá lâu thì coi như phiên mới. Đây là tham số có thể điều chỉnh.

### Nhóm 2 — Phương pháp đánh giá

**Q5. Vì sao không dùng accuracy làm metric chính?**
Dữ liệu mất cân bằng nặng (~3,5% bất thường). Đoán "tất cả bình thường" đã đạt ~96,5% accuracy nhưng vô dụng. Nên dùng precision, recall, F1 và PR-AUC.

**Q6. Tại sao chia dữ liệu theo nhóm visitor (group split)?**
Vì session sinh ra từ visitor. Nếu chia ngẫu nhiên theo session, các session của cùng visitor có thể nằm cả ở train lẫn test, khiến mô hình "thấy trước" người đó và metric cao giả. Group split theo `visitorid` đảm bảo không visitor nào xuất hiện ở nhiều tập — có kiểm tra tự động xác nhận 0 trùng lặp.

**Q7. Vì sao chọn ngưỡng trên validation chứ không trên test?**
Dò ngưỡng tốt nhất ngay trên test là gian lận thông tin — test không còn khách quan. Bọn em chọn ngưỡng tối ưu F1 trên validation, rồi áp cố định lên test để đo trung thực.

**Q8. Shuffle-label sanity check là gì, để làm gì?**
Train lại mô hình với nhãn bị xáo trộn ngẫu nhiên. Nếu pipeline đúng, mô hình không học được gì và điểm phải tụt mạnh. Nếu điểm vẫn cao thì có lỗi rò rỉ. Kết quả của bọn em tụt mạnh (F1 còn 0,21) đúng kỳ vọng.

**Q9. Tại sao bảng full_features cho F1 gần tuyệt đối?**
Đó chính là minh chứng leakage: full_features chứa các biến trực tiếp tạo ra luật, nên mô hình chỉ việc tái hiện luật. Bọn em cố tình để bảng này để so sánh, và KHÔNG dùng nó làm kết luận.

### Nhóm 3 — Phần XGBoost (nhóm trưởng)

**Q10. Vì sao chọn XGBoost làm mô hình chính?**
XGBoost xử lý tốt dữ liệu dạng bảng, mất cân bằng và quan hệ phi tuyến, có regularization chống overfit và cho feature importance để giải thích. Trên `safe_features` nó thuộc nhóm cho kết quả tốt nhất.

**Q11. Các siêu tham số chính có ý nghĩa gì?**
`max_depth=4` giữ cây nông tránh overfit; `learning_rate=0.08` với `n_estimators=120` học từ từ, ổn định; `subsample=0.85` và `colsample_bytree=0.85` lấy mẫu ngẫu nhiên dòng/cột tăng tổng quát; `scale_pos_weight = neg/pos` bù lớp bất thường hiếm.

**Q12. `scale_pos_weight` giải quyết vấn đề gì?**
Lớp bất thường rất ít, mô hình dễ thiên về lớp bình thường. `scale_pos_weight` tăng trọng số cho lỗi ở lớp dương (bất thường), giúp mô hình bắt anomaly tốt hơn — cải thiện recall.

**Q13. Đặc trưng nào quan trọng nhất và vì sao hợp lý?**
Các đặc trưng thời gian/nhịp thao tác đứng đầu (độ lệch khoảng cách giữa event, khoảng cách tối đa, độ dài session). Hợp lý vì bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

**Q14. So với LightGBM thì sao, cả hai đều là boosting?**
Cả hai cùng họ gradient boosting và cho kết quả sát nhau. LightGBM nhanh hơn nhờ leaf-wise growth, XGBoost ổn định và phổ biến hơn. Nhóm dùng cả hai để đối chiếu, kết quả tương đồng củng cố độ tin cậy.

### Nhóm 4 — Unsupervised & so sánh mô hình

**Q15. Vì sao Isolation Forest điểm thấp hơn supervised?**
Vì nó hoàn toàn không dùng nhãn khi train, chỉ tìm điểm dị biệt theo phân bố dữ liệu. Bọn em dùng nó như tham chiếu unsupervised và báo cáo mức overlap với pseudo-label, không kỳ vọng nó cạnh tranh trực tiếp.

**Q16. Decision Tree dùng làm gì khi đã có XGBoost?**
Decision Tree dễ giải thích trực quan bằng cây quyết định, phù hợp trình bày logic phân loại cho người không chuyên. XGBoost mạnh hơn về hiệu năng nhưng khó "đọc" trực tiếp.

### Nhóm 5 — Hạn chế & mở rộng

**Q17. Hạn chế lớn nhất của đồ án?**
Nhãn là pseudo-label từ luật, nên metric đo độ khớp với luật chứ chưa phải bắt gian lận thật. Ngoài ra chưa đánh giá theo thời gian (drift).

**Q18. Nếu phát triển tiếp, các em làm gì?**
Time-based split mô phỏng production; calibration score (Platt/Isotonic) trước khi đặt ngưỡng; tối ưu ngưỡng theo chi phí nghiệp vụ; và quan trọng nhất là thu thập nhãn thật để đánh giá độc lập.

**Q19. Làm sao biết các luật đặt ngưỡng hợp lý?**
Ngưỡng dựa trên phân vị thống kê của chính dữ liệu (p97,5–p99,99) và logic nghiệp vụ. Đây là điểm có thể tinh chỉnh; lý tưởng là hiệu chỉnh theo phản hồi chuyên gia hoặc nhãn thật.

**Q20. Đóng góp cụ thể của từng thành viên?**
Mỗi người phụ trách một thuật toán: em (nhóm trưởng) làm XGBoost và điều phối chung; Lê Văn Anh — Decision Tree; Tuấn Anh — Random Forest; Thủy — LightGBM; Đức Anh — Isolation Forest. Tiền xử lý, xây luật và đánh giá làm chung.

### Câu "bẫy" & cách giữ bình tĩnh

- Quên số liệu cụ thể → mở `group_model_metrics.csv` hoặc bảng trong notebook, **đừng đoán bừa**.
- Bị chỉ ra điểm yếu → **thừa nhận thẳng** rồi nối sang hướng cải tiến.
- Bị hỏi "kết quả cao thế tin được không" → dẫn ngay **ba lớp kiểm soát** (safe_features, group split, shuffle-label).
- Câu chốt an toàn: *"Đây là hạn chế bọn em đã nhận thức được và nêu trong phần hướng phát triển."*

---

## PHẦN 10 — PHÂN CÔNG NHÓM

| Thành viên | Thuật toán | Loại | Vai trò |
|---|---|---|---|
| **Đình Tuấn (nhóm trưởng)** | **XGBoost** | Có giám sát | **Mô hình chính + điều phối** |
| Lê Văn Anh | Decision Tree | Có giám sát | Dễ giải thích trực quan |
| Tuấn Anh | Random Forest | Có giám sát | Ensemble + feature importance |
| Thủy | LightGBM | Có giám sát | Đối chiếu với XGBoost |
| Đức Anh | Isolation Forest | Không giám sát | Tham chiếu unsupervised |

*Tiền xử lý, xây luật và đánh giá: làm chung.*

---

## PHẦN 11 — TỆP ĐẦU RA KÈM THEO

- `group_model_metrics.csv` — bảng metric chính (nguồn số liệu chuẩn).
- `group_leakage_audit.csv` — kiểm toán biến nào bị loại khỏi `safe_features`.
- `group_anomaly_type_breakdown.csv` — phân bố loại bất thường.
- `session_anomaly_tags.csv` — nhãn luật từng session.
- `session_anomaly_report.csv` — báo cáo tổng hợp rule + model.
- Notebook: `anomaly_detection_group_project.ipynb`.
