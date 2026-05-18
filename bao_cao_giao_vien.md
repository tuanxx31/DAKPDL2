# BAO CAO DO AN NHOM
## Phat hien bat thuong theo luat nghiep vu tren Session trong web thuong mai dien tu

- Mon hoc: Phan tich du lieu / Machine Learning ung dung
- Dataset: RetailRocket E-commerce events
- File notebook: `anomaly_detection_group_project.ipynb`
- Ngay lap bao cao: 18/05/2026

## 1. Muc tieu bai toan
Muc tieu cua do an la phat hien cac session co hanh vi dang nghi tren website thuong mai dien tu. Do dataset khong co nhan anomaly that, nhom xay dung he thong business rules de gan pseudo-label, sau do dung cac thuat toan ML de hoc va danh gia muc do phu hop voi pseudo-label.

## 2. Du lieu va pham vi phan tich
- Tong so su kien sau tien xu ly: 2,755,641
- So visitor duy nhat: 1,407,580
- So session profile: 1,761,675
- Don vi phan tich chinh: `session_id` (tach session bang khoang nghi 30 phut)

Session-level anomaly theo luat:
- So session bat thuong theo business rules: 61,849 (3.51%)
- So session bat thuong sau tong hop Rule + Model: 170,842 (9.70%)

## 3. Huong xu ly cua bai
Do an di theo 4 huong xu ly chinh:

1. Rule-based anomaly tagging
- Xay dung 12 business rules (BR01 -> BR12) de gan nhan `is_anomaly_rule`.
- Cho phep 1 session mang nhieu anomaly type (multi-label) de phan anh hanh vi phuc hop.

2. Supervised learning voi pseudo-label
- Train cac mo hinh: XGBoost, Decision Tree, Random Forest, LightGBM.
- Danh gia tren `safe_features` de giam label leakage.

3. Unsupervised tham chieu
- Dung Isolation Forest khong can nhan khi train.
- Bao cao overlap voi pseudo-label de doi chieu.

4. Kiem soat chat luong mo hinh
- Tach `safe_features` (bao cao chinh) va `full_features` (chi de minh hoa rule-mimic/leakage).
- Co `shuffle-label sanity check` de kiem tra pipeline.

## 4. Chuan bi du lieu cho mo hinh
- Mau supervised duoc lay: 300,000 session
- Chia du lieu: Train 180,000, Validation 60,000, Test 60,000
- Ty le anomaly toan bo tap session: 3.51%
- So luong feature:
- `safe_features`: 15
- `full_features`: 37

## 5. Ket qua dinh luong (tap test)
Bang duoi day dung cho bao cao chinh (`safe_features`):

| Mo hinh | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy baseline | 0.0454 | 0.0451 | 0.0453 | 0.5053 | 0.0356 |
| Decision Tree | 0.6363 | 0.8082 | 0.7120 | 0.9836 | 0.7228 |
| Random Forest | 0.8049 | 0.9031 | 0.8512 | 0.9902 | 0.8986 |
| XGBoost | 0.8122 | 0.9055 | 0.8563 | 0.9912 | 0.9216 |
| LightGBM | 0.8276 | 0.9255 | 0.8738 | 0.9917 | 0.9343 |
| Isolation Forest (overlap) | 0.2998 | 0.6695 | 0.4142 | 0.9373 | 0.3557 |

Nhan xet:
- Nhom Gradient Boosting (LightGBM, XGBoost) cho hieu qua tot nhat.
- Dummy baseline rat thap, xac nhan bai toan co y nghia hoc may.
- Isolation Forest giu vai tro tham chieu unsupervised, khong canh tranh truc tiep voi supervised.

## 6. Phan tich rieng ket qua XGBoost
Cau hinh chinh:
- `objective='binary:logistic'`, `tree_method='hist'`
- `n_estimators=120`, `max_depth=4`, `learning_rate=0.08`
- `subsample=0.85`, `colsample_bytree=0.85`
- `scale_pos_weight = neg/pos` de xu ly mat can bang lop

Nguong du doan:
- Chon tren validation theo toi uu F1: `threshold = 0.8834`

Ket qua:
- Validation (`safe_features`): F1 = 0.8602, Precision = 0.8075, Recall = 0.9202
- Test (`safe_features`): F1 = 0.8563, Precision = 0.8122, Recall = 0.9055

Dien giai:
- Recall cao (~90.55%) cho thay mo hinh bat duoc phan lon session anomaly theo pseudo-label.
- Precision ~81.22% nghia la canh bao sai con ton tai, nhung da o muc kha tot voi bai toan anomaly mat can bang.
- Khoang cach nho giua validation va test cho thay do on dinh tot, chua thay overfit ro rang.

Top feature importance cua XGBoost (`safe_features`):
- `std_interval_sec` (0.659989)
- `max_interval_sec` (0.116186)
- `session_duration_sec` (0.070828)
- `peak_ratio` (0.038097)
- `unique_event_types` (0.028831)

Y nghia nghiep vu:
- Mo hinh hoc manh tu mau thoi gian va nhip hanh vi trong session (do lech khoang cach event, khoang cach toi da, do dai session), phu hop logic phat hien bot/rapid-fire/session bat thuong.

## 7. Kiem soat leakage va sanity check
Ket qua `full_features` (co chua feature tao rule) rat cao:
- XGBoost test: F1 = 0.996, ROC-AUC = 1.0, PR-AUC = 1.0

Dien giai:
- Day la ket qua rule-mimic do leakage, khong dung de ket luan hieu nang trien khai that.

Sanity check (train voi nhan bi tron ngau nhien):
- XGBoost shuffled-label test: F1 = 0.1691, PR-AUC = 0.0918

Dien giai:
- Diem giam manh dung ky vong, xac nhan pipeline danh gia hop ly va khong bi loi nghiem trong.

## 8. Co cau anomaly type theo business rules
Top cac anomaly type xuat hien nhieu:
- Click fraud: 31,213 session (1.7718%)
- Night crawler: 30,130 session (1.7103%)
- Bot scraper: 10,884 session (0.6178%)
- Rapid-fire: 8,052 session (0.4571%)
- Session bomb: 5,112 session (0.2902%)

Nhom anomaly nay cho thay hanh vi dang nghi tap trung vao:
- Luong thao tac nhieu/nhanh bat thuong
- Mau hanh vi xem nhieu, it chuyen doi
- Hoat dong lech khung gio binh thuong

## 9. Ket luan
- Do an da xay dung duoc quy trinh anomaly detection session-level hoan chinh: Rule -> Model -> Danh gia -> Export report.
- Huong `safe_features + threshold tuning + sanity check` la diem manh, giup ket qua tin cay hon.
- XGBoost la mo hinh manh trong bai toan nay (F1 test = 0.8563), nhung LightGBM dang nhuong tot hon nhe tren bo metric chinh.

## 10. De xuat cai tien
1. Bo sung danh gia theo thoi gian (time-based split) de mo phong production drift.
2. Dung calibration cho score (Platt/Isotonic) truoc khi dat nguong canh bao.
3. Toi uu nguong theo muc tieu nghiep vu (uu tien recall hay precision theo chi phi false alert).
4. Neu co the, thu thap nhan anomaly that de danh gia ngoai pseudo-label.

## 11. Tep dau ra kem theo
- `group_model_metrics.csv`
- `group_leakage_audit.csv`
- `group_anomaly_type_breakdown.csv`
- `session_anomaly_tags.csv`
- `session_anomaly_report.csv`

