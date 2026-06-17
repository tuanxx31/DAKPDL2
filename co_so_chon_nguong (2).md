### Cơ sở chọn ngưỡng (threshold rationale)

Ngưỡng của 12 luật nghiệp vụ **không phải số tùy ý**. Chúng được đặt theo hai nguyên tắc đã được tài liệu hóa trong nghiên cứu và thực hành công nghiệp, áp dụng trên phân phối thực nghiệm của toàn bộ 1.76 triệu session RetailRocket:

**Nguyên tắc 1 — Độ hiếm thống kê (percentile cao, không giả định phân phối).**
Bất thường về bản chất là *sự kiện hiếm nằm ở vùng xác suất thấp* của phân phối hành vi bình thường. Vì các đặc trưng session (số event, số item, số lần xem lại…) là dữ liệu **đếm, lệch phải mạnh**, ta không dùng ngưỡng theo trung bình ± k·độ-lệch-chuẩn (giả định chuẩn) mà dùng **phân vị thực nghiệm (quantile)** — một cách tiếp cận phi tham số, không giả định phân phối, đúng theo tinh thần luật IQR của Tukey và các phương pháp ngưỡng-theo-percentile cho điểm bất thường. Mỗi luật vì vậy nhắm vào nhóm **top ~0.1–1%** (≈ p99–p99.9): đủ "lệch chuẩn" để coi là đáng nghi, nhưng vẫn là một tiêu chí khách quan đọc thẳng từ dữ liệu.

**Nguyên tắc 2 — Tính học được của lớp (đủ mẫu dương tối thiểu).**
Đây là phần lý giải vì sao BR10/BR11/BR12 *được hạ xuống* khỏi đuôi cực hiếm. Trong học máy mất cân bằng lớp, khi lớp thiểu số quá nhỏ, mô hình suy biến thành "luôn đoán lớp đa số": độ chính xác tổng thể cao giả tạo nhưng recall của lớp hiếm gần như bằng 0 (vấn đề *learning from imbalanced data*). Ngưỡng đặt quá cao (cũ: BR11 ≥ 20 chỉ trúng 49 session, BR10 ≥ 10 còn ~224 session) tạo ra **lớp chết** — không đủ mẫu để bất kỳ thuật toán nào học được, F1 ≈ 0.03. Do đó ba ngưỡng này được hiệu chỉnh về quanh p99.9 sao cho **mỗi lớp có >1.500 session dương**, đủ tín hiệu để mô hình học và F1 cải thiện rõ.

Khi hai nguyên tắc xung đột (đuôi quá hiếm ⇒ lớp chết), Nguyên tắc 2 được ưu tiên có chủ đích, và việc đó được ghi rõ ở cột "Ghi chú".

#### Cơ sở theo nhóm hành vi

- **Đơn vị session = khoảng nghỉ 30 phút** dựa trên heuristic kinh điển của Catledge & Pitkow (1995): khoảng cách trung bình giữa hai thao tác ≈ 9.3 phút, lấy 1.5 độ lệch chuẩn ≈ 25.5 phút rồi làm tròn lên 30 phút — đây cũng là mặc định ngành (Google Analytics).
- **BR01/BR03/BR04/BR05 (bot, click-fraud, rapid-fire, night-crawler)**: lưu lượng tự động chiếm tỉ trọng lớn và phần lớn là độc hại (Imperva *Bad Bot Report* 2025: ~51% lưu lượng Internet là tự động, ~72% trong số đó là bot xấu). Các tín hiệu cấp session như tần suất event, khoảng-cách-thời-gian-giữa-event cực ngắn, và tỉ lệ hoạt động ban đêm là đặc trưng phát hiện bot tiêu chuẩn — ngưỡng đặt ở đuôi trên (p98–p99.9) của chính các đặc trưng này.
- **BR10 (cart abandonment)**: bỏ giỏ là hiện tượng rất phổ biến (Baymard Institute: trung bình **70.22%** giỏ hàng bị bỏ, tổng hợp từ 50 nghiên cứu). Vì phổ biến nên *bản thân việc bỏ giỏ không bất thường*; luật chỉ gắn cờ khi **số lần add-to-cart cao bất thường mà tuyệt nhiên không có giao dịch** (n_addtocart ≥ p99.9, n_transaction = 0) — tức hành vi tích trữ/abuse, không phải khách do dự.

| Rule | Đặc trưng | Ngưỡng | Vị trí phân vị | % session trúng | Nguyên tắc | Ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| BR01 | total_events | ≥ 12 | ≈ p99.5 | ~0.7% | 1 | giữ nguyên |
| BR03 | n_view | ≥ 6 | ≈ p98 | ~1.8% | 1 | lớp lớn nhất |
| BR04 | rapid_fire_count / ratio | ≥1 / ≥0.20 | top ~0.5% | ~0.5% | 1 | giữ nguyên |
| BR05 | night_ratio (&events≥4) | ≥ 0.75 | — | ~1.7% | 1 | giữ nguyên |
| BR06 | max_same_item_atc | > 1 | ≈ p99.8 | ~0.2% | 1 | giữ nguyên |
| BR07 | unique_items | ≥ 12 | ≈ p99.7 | ~0.3% | 1 | giữ nguyên |
| BR09 | n_transaction | ≥ 3 | > p99.9 | ~0.09% | 1 | giữ nguyên |
| **BR10** | n_addtocart (&tx=0) | **≥ 4** (cũ 10) | ≈ p99.9 | ~0.09% | 1+2 | **hạ: cũ 10 ≫ p99.9 → lớp gần rỗng (n=224)** |
| **BR11** | max_same_item_view | **≥ 6** (cũ 20) | ≈ p99.9 (=7) | ~0.19% | 1+2 | **hạ: cũ 20 chỉ trúng 49 session → lớp chết** |
| **BR12** | unique_categories | **≥ 10** (cũ 20) | ≈ p99.9 (=9) | ~0.09% | 1+2 | **hạ: cũ 20 ≫ p99.9 → quá hiếm** |

#### Giới hạn cần nêu rõ
Phân vị định nghĩa **độ hiếm thống kê**, là *proxy* cho bất thường hành vi chứ không phải bằng chứng gian lận. Vì vậy nhãn sinh từ các luật này là **nhãn giả (pseudo-label)**, dùng để huấn luyện/đối chứng, không phải ground-truth đã kiểm chứng thủ công. Đây cũng là lý do notebook tách `safe_feature_cols` và kiểm tra rò rỉ nhãn ở mục 5.

#### Tài liệu tham khảo
1. Catledge, L.D., & Pitkow, J.E. (1995). *Characterizing Browsing Strategies in the World-Wide Web*. Computer Networks and ISDN Systems, 27(6), 1065–1073. — cơ sở session 30 phút. https://www.semanticscholar.org/paper/0078d9ec8a3cecf9a30cb85f2b5d235605d42a85
2. Google Analytics Help — *[GA4] Session* (mặc định hết hạn sau 30 phút không hoạt động). — chuẩn ngành. https://support.google.com/analytics/answer/12798876
3. Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley. — luật outlier theo IQR/quantile, phi tham số. (sách in; xem WorldCat: https://search.worldcat.org/title/3058187)
4. Quantile/percentile-based anomaly thresholding — vd. *Quantile-Based Statistical Techniques for Anomaly Detection* (CEUR-WS, Vol-3746): https://ceur-ws.org/Vol-3746/Paper_7.pdf; ngưỡng = percentile(scores, τ) cho điểm bất thường (SPINEX, arXiv:2407.04760): https://arxiv.org/abs/2407.04760
5. He, H., & Garcia, E.A. (2009). *Learning from Imbalanced Data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284. — vì sao lớp thiểu số quá nhỏ thì không học được; cơ sở cho việc bảo đảm mẫu dương tối thiểu. https://doi.org/10.1109/TKDE.2008.239
6. Baymard Institute. *Cart Abandonment Rate Statistics* (trung bình 70.22%, meta-analysis 50 nghiên cứu). — cơ sở BR10. https://baymard.com/lists/cart-abandonment-rate
7. Imperva. *Bad Bot Report 2025* (≈51% lưu lượng là tự động; ~72% bot là độc hại). — cơ sở nhóm luật bot BR01/BR03/BR04/BR05. https://www.imperva.com/resources/resource-library/reports/2025-bad-bot-report/
