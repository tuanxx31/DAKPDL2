# KỊCH BẢN TRÌNH BÀY & CÂU HỎI GIÁO VIÊN CÓ THỂ HỎI

### Đồ án: Phát hiện hành vi bất thường theo session (RetailRocket)

> Mẹo: đọc lướt mục A để biết trình tự nói, học kỹ mục B vì đây là nơi giáo viên hay "vặn". Khi nói số liệu, dùng bảng kết quả mới nhất sau khi chạy lại group split.

---

## A. KỊCH BẢN TRÌNH BÀY (khoảng 5–7 phút)

**1. Mở đầu (30 giây).**
"Đồ án của nhóm em phát hiện các phiên truy cập (session) có hành vi bất thường trên website thương mại điện tử, dùng dataset RetailRocket. Em là nhóm trưởng, phụ trách mô hình chính là XGBoost."

**2. Bài toán & dữ liệu (1 phút).**
"Dataset có khoảng 2,75 triệu sự kiện của 1,4 triệu visitor. Bọn em tách thành 1,76 triệu session bằng ngưỡng nghỉ 30 phút. Vấn đề là dataset **không có nhãn bất thường thật**, nên nhóm xây 12 luật nghiệp vụ để gán nhãn giả (pseudo-label), rồi dùng các thuật toán khai phá dữ liệu để học lại và so sánh."

**3. Đặc trưng & luật (1 phút).**
"Mỗi session được mô tả bằng các đặc trưng về tần suất, độ đa dạng, thời gian, tốc độ thao tác và tỉ lệ hành vi. 12 luật gồm bot scraper, click fraud, night crawler, rapid-fire... Tỉ lệ session bất thường khoảng 3,5%."

**4. Phương pháp & kiểm soát chất lượng (1,5 phút — phần ăn điểm).**
"Điểm bọn em chú trọng là độ tin cậy của đánh giá, qua ba lớp kiểm soát:

- Tách `safe_features` (15 đặc trưng, không chứa thứ tạo ra luật) để báo cáo chính, và `full_features` (37) chỉ để minh họa hiện tượng học lại luật.
- Chia dữ liệu **theo nhóm visitor** để cùng một người không lọt vào cả train lẫn test, tránh rò rỉ.
- Kiểm tra shuffle-label: train với nhãn xáo trộn, điểm phải tụt mạnh — xác nhận pipeline không lỗi."

**5. Kết quả & phần XGBoost của em (1,5 phút).**
"Trong các mô hình, nhóm gradient boosting (XGBoost, LightGBM) cho kết quả tốt nhất trên `safe_features`. XGBoost em cấu hình max_depth=4, learning_rate=0.08, dùng `scale_pos_weight` để xử lý mất cân bằng, và chọn ngưỡng tối ưu F1 trên validation rồi mới áp lên test. Đặc trưng quan trọng nhất là độ lệch và khoảng cách thời gian giữa các thao tác — phù hợp logic phát hiện bot."

**6. Kết luận (30 giây).**
"Đồ án dựng được quy trình hoàn chỉnh Rule → Model → Đánh giá → Export. Hạn chế là nhãn vẫn là pseudo-label nên metric phản ánh độ khớp với luật; hướng phát triển là time-based split, calibration ngưỡng, và thu thập nhãn thật."

---

## B. NGÂN HÀNG CÂU HỎI & GỢI Ý TRẢ LỜI

### Nhóm 1 — Về bản chất bài toán (hay hỏi nhất)

**Q1. Nhãn ở đâu ra? Mô hình có phải chỉ học lại đúng cái luật các em đặt ra không?**
Đúng là một rủi ro lớn và nhóm em đã xử lý có chủ đích. Nhãn là pseudo-label sinh từ 12 luật. Nếu cho mô hình học bằng chính các đặc trưng tạo ra luật thì nó sẽ "chép" lại luật và metric đẹp giả tạo. Vì vậy bọn em tách `safe_features` — bỏ các đặc trưng trực tiếp cấu thành luật — và chỉ dùng bộ này để báo cáo. Khi đó mô hình phải học các mẫu hành vi gián tiếp chứ không chép luật. Bọn em cũng để bảng `full_features` riêng để cho thấy rõ chênh lệch do leakage.

**Q2. Vậy kết quả này có ý nghĩa thực tế không, khi không có nhãn thật?**
Em xin nhận đây là hạn chế nền tảng. Metric hiện tại đo mức độ mô hình tổng quát hóa được mẫu hành vi bất thường mà luật mô tả, chứ chưa khẳng định bắt được gian lận thật. Để đánh giá thật cần nhãn do con người gán hoặc dữ liệu có ground-truth — đây là hướng phát triển bọn em đề xuất.

**Q3. Tại sao phân tích theo session mà không theo visitor?**
Hành vi bất thường trong TMĐT (bot, rapid-fire, click fraud) thường xảy ra gọn trong một phiên ngắn. Gộp theo visitor sẽ trộn lẫn phiên bình thường và bất thường của cùng người, làm mờ tín hiệu. Session-level cho độ phân giải hành vi tốt hơn.

**Q4. Vì sao chọn ngưỡng nghỉ 30 phút để tách session?**
30 phút là ngưỡng phổ biến trong phân tích web (chuẩn của nhiều công cụ analytics). Nếu nghỉ quá lâu thì coi như phiên mới. Đây là tham số có thể điều chỉnh nếu cần.

### Nhóm 2 — Về phương pháp đánh giá

**Q5. Vì sao không dùng accuracy làm metric chính?**
Dữ liệu mất cân bằng nặng (chỉ ~3,5% bất thường). Một mô hình đoán "tất cả bình thường" đã đạt ~96,5% accuracy nhưng vô dụng. Nên bọn em dùng precision, recall, F1 và PR-AUC — phản ánh đúng khả năng bắt lớp hiếm.

**Q6. Tại sao chia dữ liệu theo nhóm visitor (group split)?**
Vì session sinh ra từ visitor. Nếu chia ngẫu nhiên theo session, các session của cùng một visitor có thể nằm cả ở train lẫn test, khiến mô hình "thấy trước" người dùng đó và metric cao giả. Group split theo `visitorid` đảm bảo không visitor nào xuất hiện ở nhiều tập — bọn em có kiểm tra tự động xác nhận 0 trùng lặp.

**Q7. Vì sao chọn ngưỡng trên validation chứ không trên test?**
Nếu dò ngưỡng tốt nhất ngay trên test thì là gian lận thông tin — test không còn khách quan. Bọn em chọn ngưỡng tối ưu F1 trên validation, rồi áp cố định lên test để đo trung thực.

**Q8. Shuffle-label sanity check là gì, để làm gì?**
Là train lại mô hình với nhãn bị xáo trộn ngẫu nhiên. Nếu pipeline đúng, mô hình không thể học được gì và điểm phải tụt mạnh (về gần mức ngẫu nhiên). Nếu điểm vẫn cao thì chứng tỏ có lỗi rò rỉ ở đâu đó. Kết quả của bọn em tụt mạnh đúng kỳ vọng.

**Q9. Tại sao bảng full_features lại cho F1 gần như tuyệt đối?**
Đó chính là minh chứng cho leakage: full_features chứa các biến trực tiếp tạo ra luật, nên mô hình chỉ việc tái hiện luật. Bọn em cố tình để bảng này để so sánh, và KHÔNG dùng nó làm kết luận.

### Nhóm 3 — Về phần XGBoost (phần của em, nhóm trưởng)

**Q10. Vì sao chọn XGBoost làm mô hình chính?**
XGBoost xử lý tốt dữ liệu dạng bảng, mất cân bằng và quan hệ phi tuyến, lại có cơ chế regularization chống overfit và cho feature importance để giải thích. Trên `safe_features` nó thuộc nhóm cho kết quả tốt nhất.

**Q11. Các siêu tham số chính có ý nghĩa gì?**
`max_depth=4` giữ cây nông để tránh overfit; `learning_rate=0.08` với `n_estimators=120` để học từ từ, ổn định; `subsample=0.85` và `colsample_bytree=0.85` lấy mẫu ngẫu nhiên dòng/cột tăng tính tổng quát; `scale_pos_weight = neg/pos` để bù cho lớp bất thường hiếm.

**Q12. `scale_pos_weight` giải quyết vấn đề gì?**
Lớp bất thường rất ít, mô hình dễ thiên về lớp bình thường. `scale_pos_weight` tăng trọng số cho lỗi ở lớp dương (bất thường), giúp mô hình quan tâm hơn đến việc bắt đúng anomaly — cải thiện recall.

**Q13. Đặc trưng nào quan trọng nhất và vì sao hợp lý?**
Các đặc trưng về thời gian/nhịp thao tác đứng đầu (độ lệch khoảng cách giữa event, khoảng cách tối đa, độ dài session). Điều này hợp lý vì bot và rapid-fire có nhịp thao tác đều/nhanh bất thường so với người thật.

**Q14. So với LightGBM thì sao, cả hai đều là boosting?**
Cả hai cùng họ gradient boosting và cho kết quả sát nhau. LightGBM nhanh hơn nhờ leaf-wise growth, XGBoost ổn định và phổ biến hơn. Nhóm dùng cả hai để đối chiếu, kết quả tương đồng củng cố độ tin cậy.

### Nhóm 4 — Về unsupervised & so sánh mô hình

**Q15. Vì sao Isolation Forest điểm thấp hơn các mô hình có giám sát?**
Vì nó hoàn toàn không dùng nhãn khi train, chỉ tìm điểm dị biệt theo phân bố dữ liệu. Bọn em dùng nó như tham chiếu unsupervised và báo cáo mức overlap với pseudo-label, không kỳ vọng nó cạnh tranh trực tiếp với supervised.

**Q16. Decision Tree dùng để làm gì khi đã có XGBoost?**
Decision Tree dễ giải thích trực quan bằng cây quyết định, phù hợp để trình bày logic phân loại cho người không chuyên. XGBoost mạnh hơn về hiệu năng nhưng khó "đọc" trực tiếp.

### Nhóm 5 — Hạn chế & mở rộng

**Q17. Hạn chế lớn nhất của đồ án?**
Nhãn là pseudo-label từ luật, nên metric đo độ khớp với luật chứ chưa phải bắt gian lận thật. Ngoài ra chưa đánh giá theo thời gian (drift).

**Q18. Nếu phát triển tiếp, các em làm gì?**
Time-based split để mô phỏng production; calibration score (Platt/Isotonic) trước khi đặt ngưỡng; tối ưu ngưỡng theo chi phí nghiệp vụ (recall vs precision); và quan trọng nhất là thu thập nhãn thật để đánh giá độc lập.

**Q19. Làm sao biết các luật nghiệp vụ đặt ngưỡng hợp lý?**
Các ngưỡng dựa trên phân vị thống kê của chính dữ liệu và logic nghiệp vụ. Đây là điểm có thể tinh chỉnh; lý tưởng nhất là hiệu chỉnh dựa trên phản hồi của chuyên gia hoặc nhãn thật.

**Q20. Đóng góp cụ thể của từng thành viên?**
Mỗi người phụ trách một thuật toán: em (nhóm trưởng) làm XGBoost và điều phối chung; Lê Văn Anh — Decision Tree; Tuấn Anh — Random Forest; Thủy — LightGBM; Đức Anh — Isolation Forest. Phần tiền xử lý, xây luật và đánh giá là làm chung.

---

## C. CÂU "BẪY" & CÁCH GIỮ BÌNH TĨNH

- Nếu bị hỏi số liệu cụ thể chưa nhớ: mở bảng kết quả trong notebook/báo cáo, đừng đoán bừa.
- Nếu bị chỉ ra điểm yếu: thừa nhận thẳng thắn rồi nối sang hướng cải tiến — giáo viên đánh giá cao thái độ này hơn là cãi.
- Nếu hỏi "kết quả cao thế có tin được không": dẫn ngay tới ba lớp kiểm soát (safe_features, group split, shuffle-label) để chứng minh độ tin cậy.
- Câu chốt an toàn: "Đây là hạn chế bọn em nhận thức được và đã nêu trong phần hướng phát triển."
