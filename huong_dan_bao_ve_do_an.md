# HƯỚNG DẪN BẢO VỆ ĐỒ ÁN

## Phát hiện hành vi bất thường theo session — RetailRocket E-commerce

> Mục tiêu của tài liệu: giúp bạn trả lời mạch lạc 4 câu hỏi cốt lõi giám khảo luôn hỏi — **Dữ liệu thế nào? Ý tưởng xử lý ra sao? Làm như thế nào? Kết quả đánh giá thế nào?** Số liệu trong tài liệu này lấy trực tiếp từ `group_model_metrics.csv` (đã chạy group split theo `visitorid`), là số dùng để báo cáo chính thức.

---

## PHẦN 0 — Một câu tóm tắt để mở đầu

"Đồ án phát hiện các phiên truy cập (session) có hành vi đáng nghi trên website thương mại điện tử. Vì dataset không có nhãn thật, nhóm xây 12 luật nghiệp vụ để sinh nhãn giả, rồi huấn luyện 5 thuật toán khai phá dữ liệu để học và đánh giá. Quy trình: **Luật → Mô hình → Đánh giá → Xuất báo cáo.** Em là nhóm trưởng, phụ trách mô hình chính XGBoost."

---

## PHẦN 1 — DỮ LIỆU NHƯ THẾ NÀO?

**Nguồn:** RetailRocket E-commerce, file gốc `events.csv` (cùng `item_properties`, `category_tree`).

**Quy mô sau tiền xử lý:**

- Sự kiện hợp lệ: **2.755.641 events**
- Visitor duy nhất: **1.407.580**
- Session sau khi tách phiên: **1.761.675**
- Loại sự kiện: `view`, `addtocart`, `transaction`

**Đặc điểm quan trọng cần nói thẳng:**

1. **Không có nhãn bất thường thật.** Đây là điểm cốt lõi — dataset chỉ ghi lại hành vi, không có cột "đây là gian lận". Đó là lý do nhóm phải tự sinh nhãn bằng luật (pseudo-label).
2. **Mất cân bằng nặng:** chỉ ~3,51% session bị gắn nhãn bất thường (61.849 / 1.761.675). Điều này quyết định cách chọn metric (xem Phần 4).

**Câu trả lời mẫu nếu giám khảo hỏi "dữ liệu gì":**
"Dữ liệu là log hành vi thật của người dùng trên sàn TMĐT, khoảng 2,75 triệu sự kiện của 1,4 triệu người. Bọn em chuyển nó từ mức sự kiện rời rạc sang mức session để phân tích hành vi. Hạn chế lớn nhất là dữ liệu không có nhãn bất thường sẵn."

---

## PHẦN 2 — Ý TƯỞNG XỬ LÝ

**Ý tưởng trung tâm: chuyển từ "sự kiện" sang "phiên truy cập" (session).**

Vì sao session mà không phải visitor hay từng event? Hành vi bất thường trong TMĐT (bot quét, click ảo, rapid-fire) thường gói gọn trong **một phiên ngắn**. Phân tích theo từng event thì quá vụn, không thấy mẫu hành vi; gộp theo visitor thì trộn lẫn phiên tốt và phiên xấu của cùng một người, làm mờ tín hiệu. Session là mức phân giải hợp lý nhất.

**Cách tách session:** sắp xếp sự kiện theo `visitorid` + `timestamp`, mở session mới khi (a) là event đầu của visitor, hoặc (b) cách event trước **hơn 30 phút**. Ngưỡng 30 phút là chuẩn phổ biến trong web analytics.

**Ý tưởng sinh nhãn: hệ thống 12 luật nghiệp vụ (business rules).** Mỗi luật mô tả một dạng bất thường có ý nghĩa kinh doanh:

| Mã | Loại | Điều kiện rút gọn |
|---|---|---|
| BR01 | Bot scraper | nhiều event / thao tác quá nhanh |
| BR02 | Ghost buyer | có mua nhưng không thêm giỏ |
| BR03 | Click fraud | xem nhiều, không mua/không thêm giỏ |
| BR04 | Rapid-fire | thao tác liên tiếp quá nhanh |
| BR05 | Night crawler | hoạt động chủ yếu 0h–5h |
| BR06 | Item hoarding | thêm cùng 1 item vào giỏ nhiều lần |
| BR07 | Session bomb | xem rất nhiều item khác nhau |
| BR08 | Sequence violation | mua trước cả khi xem/thêm giỏ |
| BR09 | Transaction burst | nhiều giao dịch trong 1 phiên |
| BR10 | Cart abandonment | thêm giỏ nhiều nhưng không mua |
| BR11 | Repeated view spam | xem lặp 1 item quá nhiều |
| BR12 | Category scanning | quét rất nhiều danh mục |

Đây là **multi-label**: một session có thể dính nhiều luật. Một session được coi là bất thường nếu vi phạm ít nhất một luật.

**Câu trả lời mẫu "tại sao không học máy thuần mà phải dùng luật":**
"Vì không có nhãn thật. Luật cho phép bọn em (1) tạo nhãn có thể giải thích được, và (2) gắn mỗi cảnh báo với một lý do nghiệp vụ cụ thể, thay vì một con số khó diễn giải."

---

## PHẦN 3 — CÁCH LÀM (PIPELINE & KIỂM SOÁT CHẤT LƯỢNG)

Đây là phần ăn điểm. Trình bày theo 6 bước:

1. **Tiền xử lý:** đọc `events.csv`, bỏ trùng lặp, giữ event hợp lệ, đổi `timestamp` sang `datetime`.
2. **Tạo session profile:** gom theo `session_id`, tính ~37 đặc trưng (tần suất, độ đa dạng, thời gian/nhịp, tốc độ thao tác, hành vi lặp, tỉ lệ hành vi).
3. **Gắn nhãn:** áp 12 luật để sinh pseudo-label.
4. **Lấy mẫu & chia dữ liệu:** lấy 300.000 session (phân tầng), chia **Train/Validation/Test = 60/20/20**.
5. **Huấn luyện 5 mô hình:** XGBoost (chính), LightGBM, Random Forest, Decision Tree, Isolation Forest.
6. **Đánh giá & xuất báo cáo:** chọn ngưỡng trên validation, đo trên test, xuất file CSV.

**BA LỚP KIỂM SOÁT CHẤT LƯỢNG (nhấn mạnh — đây là điểm khác biệt của nhóm):**

- **Lớp 1 — Tách `safe_features` và `full_features`.** `safe_features` (15 đặc trưng) **loại bỏ mọi biến trực tiếp tạo ra luật**, dùng làm kết quả chính. `full_features` (37 đặc trưng) chứa cả biến tạo luật, chỉ để minh họa hiện tượng "mô hình chép lại luật" (rule-mimic). → Tránh label leakage.
- **Lớp 2 — Group split theo `visitorid`.** Chia dữ liệu sao cho **không visitor nào xuất hiện ở hai tập**. Kiểm tra tự động xác nhận **0 visitor trùng** giữa train/val/test. → Tránh rò rỉ ở mức người dùng.
- **Lớp 3 — Shuffle-label sanity check.** Train lại với nhãn bị xáo trộn ngẫu nhiên; nếu pipeline đúng thì điểm phải tụt mạnh. Kết quả: F1 tụt còn **0,21** → xác nhận pipeline không lỗi.

**Cấu hình XGBoost (phần của nhóm trưởng):**
`max_depth=4` (cây nông chống overfit), `learning_rate=0.08` + `n_estimators=120` (học từ từ), `subsample=0.85`, `colsample_bytree=0.85` (lấy mẫu ngẫu nhiên tăng tổng quát), `scale_pos_weight = neg/pos` (bù lớp bất thường hiếm). **Ngưỡng = 0,869**, chọn tối ưu F1 trên validation rồi cố định áp lên test.

---

## PHẦN 4 — ĐÁNH GIÁ KẾT QUẢ ĐẦU RA

**Vì sao KHÔNG dùng accuracy làm metric chính:** dữ liệu mất cân bằng (~3,5% bất thường), một mô hình đoán "tất cả bình thường" đã đạt ~96,5% accuracy nhưng vô dụng. → Dùng **Precision, Recall, F1, PR-AUC**. Với lớp hiếm, **PR-AUC** là chỉ số đáng tin nhất.

### Bảng kết quả chính — tập TEST, `safe_features`

| Mô hình | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|--:|--:|--:|--:|--:|
| Dummy baseline | 0,041 | 0,040 | 0,040 | 0,503 | 0,035 |
| Decision Tree | 0,980 | 0,684 | 0,775 | 0,987 | 0,747 |
| Random Forest | 0,989 | 0,799 | 0,911 | 0,992 | 0,902 |
| **XGBoost (chính)** | 0,990 | 0,812 | **0,918** | 0,993 | 0,922 |
| **LightGBM** | 0,990 | 0,820 | **0,934** | 0,994 | 0,935 |
| Isolation Forest | 0,291 | 0,674 | 0,406 | 0,935 | 0,358 |

**Cách đọc bảng (nói gì khi trình bày):**

- **Dummy baseline gần như bằng 0** → bài toán thực sự có ý nghĩa học máy, không phải đoán mò.
- **Nhóm gradient boosting (XGBoost, LightGBM) tốt nhất.** Precision ~0,99 (rất ít báo động sai), recall ~0,81–0,82 (bắt được phần lớn session bất thường).
- **Isolation Forest thấp hơn nhiều** vì không dùng nhãn khi train — chỉ là tham chiếu unsupervised, không cạnh tranh trực tiếp.

### XGBoost — phần nhóm trưởng

| Tập | Precision | Recall | F1 |
|---|--:|--:|--:|
| Validation | 0,990 | 0,814 | 0,929 |
| Test | 0,990 | 0,812 | 0,918 |

Khoảng cách validation–test rất nhỏ (0,929 vs 0,918) → mô hình **ổn định, chưa overfit rõ rệt**.

### Bằng chứng kiểm soát chất lượng (rất quan trọng khi bị "vặn")

- **`full_features` (chứa biến tạo luật):** XGBoost test F1 = **0,994**, PR-AUC ≈ 1,0 → gần tuyệt đối, đây CHÍNH LÀ minh chứng leakage; vì vậy **không** dùng bảng này làm kết luận.
- **Shuffle-label:** XGBoost test F1 tụt còn **0,209**, PR-AUC 0,092 → đúng kỳ vọng, pipeline lành mạnh.

---

## PHẦN 5 — ĐIỂM YẾU & CÁCH BẢO VỆ (HỌC KỸ)

**1. Hạn chế nền tảng — nhãn là pseudo-label.** Metric đo "mức khớp với luật" chứ chưa khẳng định bắt được gian lận thật. → *Trả lời:* "Đây là giới hạn khi không có nhãn thật; bọn em đã nêu rõ và đề xuất thu thập nhãn thật để đánh giá độc lập."

**2. F1 trên `safe_features` vẫn rất cao (0,92) — có thật sự hết leakage chưa?** Đây là câu hỏi sắc nhất giám khảo có thể đặt. → *Trả lời trung thực:* "Bọn em đã loại các biến trực tiếp tạo luật. Tuy nhiên các đặc trưng thời gian/nhịp còn lại vẫn tương quan gián tiếp với biến đếm, nên một phần tín hiệu luật vẫn được học gián tiếp. Khoảng cách full (0,99) → safe (0,92) cho thấy đã giảm leakage đáng kể nhưng chưa loại bỏ hoàn toàn. Đây là giới hạn của bài toán dựa trên luật." (Thừa nhận điểm này = ăn điểm phương pháp.)

**3. XGBoost là "mô hình chính" nhưng LightGBM điểm cao hơn (0,934 > 0,918).** → *Trả lời:* "Cả hai cùng họ gradient boosting, kết quả sát nhau, sự tương đồng củng cố độ tin cậy. Nhóm chọn XGBoost làm mô hình chính vì phân công và độ phổ biến/ổn định, không phải vì nó cao nhất tuyệt đối."

**4. Chưa đánh giá theo thời gian (drift).** → *Hướng phát triển:* time-based split mô phỏng production.

---

## PHẦN 6 — DANH SÁCH HƯỚNG PHÁT TRIỂN (nếu được hỏi "làm tiếp gì")

1. **Time-based split** để mô phỏng drift trong môi trường thật.
2. **Calibration score** (Platt/Isotonic) trước khi đặt ngưỡng cảnh báo.
3. **Tối ưu ngưỡng theo chi phí nghiệp vụ** (ưu tiên recall hay precision tùy chi phí cảnh báo sai).
4. **Thu thập nhãn thật** để đánh giá ngoài pseudo-label — quan trọng nhất.

---

## PHẦN 7 — CHIẾN THUẬT GIỮ BÌNH TĨNH

- Quên số liệu cụ thể → mở `group_model_metrics.csv` hoặc bảng trong notebook, **đừng đoán bừa**.
- Bị chỉ ra điểm yếu → **thừa nhận thẳng** rồi nối sang hướng cải tiến. Giám khảo đánh giá cao thái độ này.
- Bị hỏi "kết quả cao thế tin được không" → dẫn ngay **ba lớp kiểm soát** (safe_features, group split, shuffle-label).
- Câu chốt an toàn: *"Đây là hạn chế bọn em đã nhận thức được và nêu trong phần hướng phát triển."*

---

## PHỤ LỤC — Phân công nhóm

| Thành viên | Thuật toán | Vai trò |
|---|---|---|
| Đình Tuấn (nhóm trưởng) | XGBoost | Mô hình chính + điều phối |
| Lê Văn Anh | Decision Tree | Dễ giải thích trực quan |
| Tuấn Anh | Random Forest | Ensemble + feature importance |
| Thủy | LightGBM | Đối chiếu với XGBoost |
| Đức Anh | Isolation Forest | Tham chiếu unsupervised |

*Tiền xử lý, xây luật và đánh giá: làm chung.*
