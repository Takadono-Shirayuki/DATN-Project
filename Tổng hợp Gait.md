**Hướng dẫn quyết định nhanh (key considerations)**

Mục tiêu: nhận dạng cá nhân trong giám sát từ xa; real-time hay batch; single-view hay multi-view.

Dữ liệu: chọn CASIA-B hoặc OU-MVLP làm benchmark ban đầu để so sánh chuẩn hóa.

Biểu diễn: bắt đầu với GEI (Gait Energy Image) để có baseline nhẹ, sau đó mở rộng sang chuỗi silhouette + transformer để nắm bắt tính động. [Tham khảo](https://www.cse.msu.edu/~rossarun/BiometricsTextBook/Papers/OtherBiometrics/Han_GaitEnergy_PAMI06.pdf)

Đánh giá: protocol cross-view, covariate (quần áo, túi, tốc độ) và open-set.

**Cơ sở khoa học**

Nhận dạng dáng đi dựa trên giả thuyết rằng mẫu hình dạng và chuyển động khi đi chứa đặc trưng cá nhân ổn định; deep learning hiện là hướng chủ đạo để học biểu diễn discriminative từ silhouette hoặc GEI [(tham khảo)](https://arxiv.org/pdf/2102.09546). Các dataset chuẩn như CASIA-B và OU-MVLP cung cấp protocol đa góc và quy mô lớn để đánh giá cross-view và population-scale performanceGithub+1.

**Hướng tiếp cận — phân tích silhouette (chi tiết)**

**Tại sao chọn silhouette**: Giảm nhiễu nền, tập trung hình dạng/chuyển động, tiết kiệm bộ nhớ; phù hợp cho GEI và các mạng CNN/3D-CNN nhẹ. Tuy nhiên mất texture/color và phụ thuộc chất lượng segmentation (background subtraction) là nhược điểm chính.

**Các lớp phương pháp:**

Biểu diễn tĩnh: GEI tổng hợp một chu kỳ đi thành ảnh đặc trưng, thuận tiện cho CNN-based extractor và nhanh cho baseline.

Biểu diễn động: 3D-CNN, CNN+RNN, hoặc Temporal Transformer để nắm bắt mối quan hệ dài hạn giữa khung hình; cải thiện robustness với biến đổi thời gian và tốc độ bước.

Cross-view handling: view-adaptive layers, view-normalization hoặc domain-adversarial training để giảm sai lệch góc nhìn; OpenGait cung cấp khung benchmark và thực nghiệm hướng thực tế để so sánh các thiết kế này. [(tham khảo)](https://openaccess.thecvf.com/content/CVPR2023/papers/Fan_OpenGait_Revisiting_Gait_Recognition_Towards_Better_Practicality_CVPR_2023_paper.pdf)



**Cấu trúc mô hình đề xuất (gợi ý)**

Preprocessing: silhouette extraction → gait cycle detection hoặc sliding-window; augmentation (view, clothing, speed).

Backbone: Option A: GEI → 2D-CNN (lightweight ResNet) cho baseline; Option B: silhouette sequence → 3D-CNN hoặc Temporal Transformer cho SOTA.

Aggregation & Head: spatial pooling + temporal pooling; embedding 256D với ArcFace hoặc triplet loss để tối ưu phân biệt.

Cross-view module: view classifier + adversarial branch hoặc learnable view embedding.

Deployment: pruning/quantization cho edge inference; pipeline segmentation robust hóa bằng background modeling.

**Datasets, protocol và công cụ**

CASIA-B (multi-view, chuẩn nghiên cứu phổ biến) và mã nguồn/README trong OpenGait giúp chuẩn hóa tiền xử lý và protocol.

OU-MVLP là dataset quy mô lớn (10k+ subjects, nhiều góc) phù hợp cho đánh giá cross-view ở quy mô dân số lớn.

OpenGait cung cấp framework, baseline và hướng thực nghiệm để so sánh tính thực tế của các phương pháp.

**Rủi ro, hạn chế và khuyến nghị**

Kỹ thuật: silhouette-based phụ thuộc segmentation; cross-view generalization vẫn là thách thức—khuyến nghị dùng multi-dataset training và self-supervised pretraining để cải thiện tính khái quát.

Thực nghiệm: bắt đầu với GEI baseline (nhanh, ít tài nguyên) rồi tiến tới sequence-based transformer cho hiệu năng cao hơn.

Nguồn tham khảo chọn lọc: [CASIA-B dataset & OpenGait README](https://github.com/ShiqiYu/OpenGait/blob/master/datasets/CASIA-B/README.md); [OpenGait (CVPR 2023) benchmark và frameworkCVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Fan_OpenGait_Revisiting_Gait_Recognition_Towards_Better_Practicality_CVPR_2023_paper.pdf); [OU-MVLP multi-view large population dataset](https://link.springer.com/content/pdf/10.1186/s41074-018-0039-6.pdf); [survey Deep Gait Recognition (comprehensive review)arXiv.org](https://arxiv.org/pdf/2102.09546); [Gait Energy Image (GEI) — Han & Bhanu (gốc).](https://www.cse.msu.edu/~rossarun/BiometricsTextBook/Papers/OtherBiometrics/Han_GaitEnergy_PAMI06.pdf)

**📌 Tóm tắt quy trình xây dựng mô hình**

**1. Tiền xử lý dữ liệu**

- **Input:** video dáng đi của người dùng.
- **Bước xử lý:**
  - Tách silhouette (ảnh bóng dáng).
  - Tạo **GEI (Gait Energy Image)** hoặc chuỗi ảnh silhouette.
- **Mục tiêu:** chuẩn hóa dữ liệu đầu vào để đưa vào CNN.
- **Hướng phụ:** Cải tiến GEI để không mất đặc trưng thời gian

**2. Trích xuất đặc trưng bằng CNN**

- **CNN pipeline:**
  - Conv + Pooling → học đặc trưng cục bộ (biên, hướng).
  - Nhiều tầng sâu hơn → học đặc trưng trừu tượng (cấu trúc cơ thể, dáng đi).
  - Global pooling / Fully Connected → tạo **embedding vector** (ví dụ 128–512 chiều).
- **Kết quả:** mỗi GEI hoặc chuỗi silhouette được ánh xạ thành một vector đặc trưng gọn nhẹ, giàu thông tin.

**3. Lưu trữ và quản lý nhãn**

- **Centroid (vector trung bình):**
  - Với mỗi người đã đăng ký, tính trung bình embedding của các mẫu → centroid đại diện.
- **Multi-prototype (nhiều nhãn cho một người):**
  - Nếu dáng đi biến động mạnh (trang phục, phụ kiện, trạng thái), lưu nhiều centroid cho cùng một người.
  - Ví dụ: centroid 1 (dáng đi bình thường), centroid 2 (mặc áo khoác), centroid 3 (mang cặp).
- **Quản lý:** tất cả centroid đều gắn cùng một ID người.

**4. Phân loại với open-set**

- **So sánh probe:** embedding của probe được so với tất cả centroid.
- **Ngưỡng tùy chỉnh theo nhãn:**
  - Mỗi centroid có ngưỡng riêng, dựa trên phân bố nội-lớp (mean + k·σ).
  - Người ổn định → ngưỡng nhỏ. Người biến động → ngưỡng lớn hơn.
- **Kết quả:**
  - Nếu probe nằm trong ngưỡng của bất kỳ centroid nào → nhận dạng đúng người đó.
  - Nếu probe nằm ngoài ngưỡng của tất cả centroid → gán “unknown”.

**5. Xử lý người mới (incremental enrollment)**

- Khi probe bị gán “unknown”:
  - Tạo lớp mới bằng embedding của probe.
  - Lưu centroid mới cho người đó.
  - Khi có thêm dữ liệu, cập nhật centroid để ổn định hơn.
- **Định kỳ:** huấn luyện lại CNN/metric learning toàn cục để duy trì độ chính xác cao.

**6. Chiến lược tổng thể**

- **Nhanh:** thêm người mới bằng cách lưu centroid, không cần huấn luyện lại ngay.
- **Chính xác:** huấn luyện lại CNN/metric learning định kỳ để tái tối ưu không gian đặc trưng.
- **Linh hoạt:** multi-prototype + ngưỡng riêng giúp giảm chồng lấn và tăng khả năng nhận dạng trong thực tế.

**🎯 Tóm gọn**

- **Dữ liệu:** Video → Silhouette → GEI.
- **Đặc trưng:** CNN embedding.
- **Quản lý nhãn:** centroid + multi-prototype.
- **Phân loại:** nearest centroid + ngưỡng open-set tùy chỉnh.
- **Người mới:** gán “unknown” → tạo lớp mới → cập nhật định kỳ.

👉 Đây là mô hình **open-set gait recognition** hiện đại: vừa linh hoạt (incremental), vừa mạnh mẽ (CNN embedding), vừa thực tế (multi-prototype + adaptive threshold).

**1. Thu thập phân bố khoảng cách nội-lớp**

- Với mỗi người (nhãn), ta có nhiều vector đặc trưng (embedding từ CNN).
- Tính khoảng cách từ từng vector đến **centroid** của người đó.
- Tập hợp các khoảng cách này tạo thành phân bố nội-lớp.

**2. Xác định ngưỡng riêng**

Có nhiều cách để đặt ngưỡng từ phân bố này:

**🔹 Cách 1: Mean + k·σ (phổ biến)**

- Tính trung bình μivà độ lệch chuẩn σicủa khoảng cách nội-lớp cho nhãn i.
- Ngưỡng cho nhãn i:

τi=μi+k⋅σi

- klà hệ số điều chỉnh (ví dụ 2 hoặc 3).
- Người ổn định → σinhỏ → ngưỡng hẹp.
- Người biến động → σilớn → ngưỡng rộng hơn.

**🔹 Cách 2: Percentile (phân vị)**

- Chọn ngưỡng bằng **phân vị 95% hoặc 99%** của khoảng cách nội-lớp.
- Nghĩa là 95% mẫu của người đó nằm trong ngưỡng, 5% có thể bị coi là “unknown”.
- Cách này trực quan, không cần giả định phân bố chuẩn.

**🔹 Cách 3: Adaptive margin**

- So sánh phân bố nội-lớp với phân bố liên-lớp (khoảng cách đến centroid của người khác).
- Chọn ngưỡng sao cho **tối đa hóa khoảng cách giữa nội-lớp và liên-lớp**.
- Đây là cách “margin-based”, thường dùng trong metric learning.

**📌 Chiến lược thuật toán để tách một người thành nhiều nhãn**

**1. Gom cụm (Clustering) trên dữ liệu của từng người**

- **KMeans:**
  - Cho embedding của tất cả mẫu thuộc một người.
  - Chạy KMeans để chia thành Kcụm nhỏ.
  - Mỗi cụm → một centroid → một nhãn phụ (prototype).
  - Ưu điểm: đơn giản, nhanh. Nhược điểm: phải chọn trước K.
- **Gaussian Mixture Model (GMM):**
  - Giả định dữ liệu của một người gồm nhiều phân bố Gaussian.
  - GMM tự ước lượng số cụm bằng tiêu chí BIC/AIC.
  - Ưu điểm: linh hoạt, có thể tự chọn số cụm.
- **Hierarchical clustering:**
  - Gom cụm theo cây, cắt cây ở mức phù hợp để tạo nhiều prototype.
  - Ưu điểm: không cần chọn số cụm trước, dễ trực quan hóa.

**2. Xác định số cụm (số nhãn phụ)**

- **Elbow method:** chạy KMeans với nhiều giá trị K, chọn Ktại điểm “gãy” của biểu đồ SSE.
- **Silhouette score:** chọn Ktối ưu dựa trên độ tách biệt giữa cụm.
- **BIC/AIC (với GMM):** chọn số cụm sao cho mô hình cân bằng giữa độ khớp và độ phức tạp.

**3. Gán nhãn phụ (multiprototype)**

- Sau khi gom cụm, mỗi cụm được coi là một **prototype** của người đó.
- Lưu centroid và ngưỡng riêng cho từng cụm.
- Khi phân loại: probe chỉ cần khớp với **một prototype** trong ngưỡng để được nhận dạng đúng người.

**4. Ưu điểm của cách này**

- **Giảm chồng lấn:** mỗi prototype bao phủ vùng nhỏ hơn → ít nhầm lẫn với người khác.
- **Xử lý biến động:** dáng đi khác nhau (trang phục, phụ kiện, trạng thái) được gom thành cụm riêng.
- **Tự động:** không cần tạo centroid bằng tay, hệ thống tự học từ dữ liệu.

**🎯 Tóm lại**

- Để chia một người thành nhiều nhãn, ta dùng **clustering nộilớp** (KMeans, GMM, hierarchical).
- Mỗi cụm → một centroid → một nhãn phụ.
- Số cụm có thể chọn bằng **Elbow, Silhouette, BIC/AIC**.
- Kết quả: một người có nhiều prototype, mỗi prototype có ngưỡng riêng, tất cả cùng trỏ về cùng ID.

**1. Gaussian Mixture Model (GMM)**

- **Ý tưởng:** dữ liệu của một người không chỉ nằm trong một cụm duy nhất mà có thể được tạo ra từ nhiều phân bố Gaussian khác nhau (ví dụ dáng đi bình thường, dáng đi khi mang cặp, dáng đi khi mặc áo khoác).
- **Cách làm:**
  - Giả định dữ liệu embedding của một người được sinh ra từ KGaussian (mỗi Gaussian có trung bình và ma trận hiệp phương sai riêng).
  - Dùng thuật toán **Expectation–Maximization (EM)** để ước lượng tham số của các Gaussian.
  - Mỗi mẫu được gán xác suất thuộc về từng Gaussian → từ đó chia thành các cụm.
- **Ưu điểm:**
  - Cho phép cụm có hình dạng **elliptical**, không chỉ tròn như KMeans.
  - Có thể tự chọn số cụm bằng tiêu chí **BIC (Bayesian Information Criterion)** hoặc **AIC (Akaike Information Criterion)**.
  - Phù hợp khi dữ liệu có nhiều kiểu biến động phức tạp.
- **Nhược điểm:**
  - Tính toán phức tạp hơn KMeans.
  - Cần dữ liệu đủ nhiều để ước lượng phân bố Gaussian chính xác.

**2. Hierarchical Clustering**

- **Ý tưởng:** thay vì chọn số cụm trước, ta xây dựng một **cây phân cấp** (dendrogram) để biểu diễn mối quan hệ giữa các mẫu.
- **Cách làm:**
  - Bắt đầu với mỗi mẫu là một cụm riêng.
  - Tính khoảng cách giữa các cụm (ví dụ khoảng cách trung bình, khoảng cách gần nhất).
  - Lần lượt gộp cụm gần nhau nhất → tạo thành cây phân cấp.
  - Cắt cây ở mức phù hợp để tạo ra số cụm mong muốn.
- **Ưu điểm:**
  - Không cần chọn số cụm trước.
  - Dễ trực quan hóa bằng dendrogram để thấy cấu trúc dữ liệu.
  - Có thể phát hiện các cụm nhỏ hiếm gặp (ví dụ dáng đi đặc biệt khi mang vật nặng).
- **Nhược điểm:**
  - Tốn thời gian nếu dữ liệu lớn.
  - Kết quả phụ thuộc vào cách định nghĩa khoảng cách giữa cụm (singlelink, completelink, averagelink).

**🎯 Tóm lại**

- **GMM:** mô hình xác suất, cho phép cụm có hình dạng phức tạp, chọn số cụm bằng BIC/AIC.
- **Hierarchical clustering:** xây dựng cây phân cấp, không cần chọn số cụm trước, dễ trực quan hóa.
- Cả hai đều phù hợp để tách một người thành nhiều prototype, giúp hệ thống nhận dạng openset chính xác hơn khi dáng đi biến động mạnh.

**🎯 Mục tiêu của hàm khoảng cách**

- **Phân biệt rõ ràng:** cùng một người thì khoảng cách nhỏ, khác người thì khoảng cách lớn.
- **Ổn định:** khoảng cách có thể so sánh được giữa các nhãn khác nhau để đặt ngưỡng openset.
- **Thích ứng:** hỗ trợ nhiều prototype cho một người và ngưỡng riêng cho từng prototype.

**1. Các lựa chọn cơ bản**

- **Khoảng cách Euclid (L2):**

d(x,c)=∥x-c∥2

Đơn giản, dễ dùng, nhưng nhạy với độ lớn vector.

- **Khoảng cách Cosine:**

d(x,c)=1-x⊤c∥x∥⋅∥c∥

Tốt khi embedding đã chuẩn hóa, phản ánh góc giữa vector.

- **Khoảng cách Mahalanobis:**

d(x,c)=x-c)⊤Σ-1(x-c

Tính đến tương quan giữa các chiều, nhưng cần dữ liệu đủ để ước lượng ma trận hiệp phương sai.

**2. Thực hành tốt**

- **Chuẩn hóa embedding (L2norm):**

x=x∥x∥,c=c∥c∥

Giúp ổn định và làm cosine ≈ Euclid.

- **Dùng cosine hoặc Euclid sau chuẩn hóa:** thường là lựa chọn an toàn và phổ biến.
- **Khoảng cách Mahalanobis cục bộ:** với mỗi prototype, ước lượng hiệp phương sai từ dữ liệu nộilớp để có khoảng cách thích ứng.

**3. Metric learning (tùy chọn)**

- Học một phép biến đổi tuyến tính Ađể tối ưu khoảng cách:

dA(x,c)=∥Ax-Ac∥2

- Giúp embedding cùng người gần nhau, khác người xa nhau → tăng độ chính xác.

**4. Multiprototype**

- Một người có nhiều centroid (prototype).
- **Khoảng cách tổng hợp:**

dmin(x,Ci)=min⁡p∈Pid(x,ci,p)

Probe được gán cho người inếu gần ít nhất một prototype trong ngưỡng.

**5. Ngưỡng openset**

- Với mỗi prototype, tính phân bố khoảng cách nộilớp.
- Ngưỡng có thể đặt theo:
  - **Mean + k·σ** (trung bình + hệ số nhân độ lệch chuẩn).
  - **Phân vị (percentile)**, ví dụ 95%.
- Mỗi prototype có ngưỡng riêng → thích ứng với mức biến động khác nhau.

**✅ Tóm lại**

- **Embedding chuẩn hóa** → dùng cosine hoặc Euclid.
- **Multiprototype:** mỗi người có nhiều centroid, probe so với tất cả.
- **Ngưỡng riêng:** đặt theo phân bố nộilớp của từng prototype.
- **Metric learning:** tăng cường độ phân biệt nếu có dữ liệu đủ lớn.


