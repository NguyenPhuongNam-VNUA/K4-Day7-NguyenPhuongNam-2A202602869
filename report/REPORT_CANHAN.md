# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Phương Nam
**Nhóm:** G17
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là góc giữa hai vector biểu diễn văn bản (embeddings) trong không gian vector rất nhỏ (hướng gần như trùng nhau). Điều này thể hiện hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa hoặc ngữ cảnh sử dụng, bất kể độ dài ký tự của hai văn bản đó chênh lệch nhau như thế nào.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên có thể nộp đơn xin xét cấp học bổng khuyến khích học tập tại phòng công tác sinh viên.
- Câu B: Thủ tục đăng ký nhận học bổng học tập của sinh viên được thực hiện tại văn phòng quản lý học sinh sinh viên.
- Tại sao tương đồng: Cả hai câu đều cùng hướng tới một hành động thực tế là quy trình sinh viên đăng ký xét duyệt học bổng tại bộ phận chuyên trách, sử dụng các từ ngữ đồng nghĩa và mang chung trường nghĩa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Sinh viên thuộc diện chính sách được miễn một trăm phần trăm học phí theo quy định của nhà nước.
- Câu B: Dự báo thời tiết ngày mai khu vực đồng bằng Bắc Bộ trời nhiều mây và có mưa dông rải rác.
- Tại sao khác: Hai câu thuộc hai lĩnh vực hoàn toàn độc lập (chế độ học phí giáo dục đại học đối lập với thông tin khí tượng thủy văn), không chia sẻ ngữ cảnh hay mối liên hệ ngữ nghĩa nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị chi phối nặng bởi độ lớn (magnitude) của vector, khiến một câu ngắn và một đoạn văn dài dù cùng một chủ đề vẫn có khoảng cách Euclid rất xa. Ngược lại, Cosine similarity chuẩn hóa độ dài và chỉ đo góc định hướng giữa các vector, giúp đánh giá chính xác độ tương đồng ngữ nghĩa mà không bị sai lệch bởi số lượng từ hay độ dài văn bản.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`
> `số lượng chunk = ceil((10,000 - 50) / (500 - 50)) = ceil(9,950 / 450) = ceil(22.111...) = 23`
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số lượng chunk sẽ tăng lên thành `ceil((10,000 - 100) / (500 - 100)) = ceil(9,900 / 400) = ceil(24.75) = 25` chunks (tăng 2 chunks). Chúng ta muốn tăng độ chồng chéo để bảo toàn mạch ngữ cảnh liền mạch giữa các ranh giới cắt, đảm bảo những thông tin hay câu văn nằm ở đoạn tiếp giáp không bị ngắt đôi gây mất nghĩa khi mô hình tìm kiếm và truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi sử dụng biểu thức chính quy lookbehind `(?<=[.!?])\s+` để phân tách văn bản ngay sau các dấu chấm câu (`.`, `!`, `?`) mà vẫn giữ nguyên được dấu câu trong nội dung. Thuật toán kiểm tra và xử lý ngoại lệ văn bản rỗng hoặc chỉ toàn khoảng trắng (`text.strip() == ""`), sau đó gom các câu đơn lẻ thành từng nhóm tối đa `max_sentences_per_chunk` câu và dùng `join` để tạo chunk hoàn chỉnh.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Tôi xây dựng hàm đệ quy `_split` duyệt qua danh sách dấu phân cách ưu tiên từ thô đến mịn (`["\n\n", "\n", ". ", " ", ""]`). Base case là khi đoạn văn bản có độ dài `<= chunk_size` hoặc khi danh sách separator đã duyệt hết (lúc này cắt trực tiếp theo kích thước). Nếu một đoạn nhỏ sau khi tách vẫn dài hơn `chunk_size`, thuật toán sẽ đệ quy tiếp với separator kế tiếp; các đoạn nhỏ hơn được tích lũy liên tục cho đến khi chạm ngưỡng kích thước tối đa.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Tôi sử dụng cấu trúc danh sách trong bộ nhớ (in-memory list of dicts) với cấu trúc chuẩn gồm `id`, `content`, `metadata`, và `embedding`. Khi thêm tài liệu (`add_documents`), từng document được tạo vector qua `_embedding_fn` và chuẩn hóa trường `metadata['doc_id']`. Khi tìm kiếm (`search`), truy vấn được nhúng thành vector, tính điểm độ tương đồng cosine thông qua tích vô hướng `_dot`, sau đó sắp xếp giảm dần theo điểm và trích xuất top-k kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Với `search_with_filter`, tôi áp dụng chiến lược lọc trước (pre-filtering): duyệt lọc các bản ghi thỏa mãn toàn bộ các cặp khóa-giá trị trong `metadata_filter`, rồi mới chạy hàm tìm kiếm tính điểm trên tập bản ghi đã lọc. Với `delete_document`, tôi dùng cơ chế lọc danh sách để loại bỏ tất cả các chunk có `id` hoặc `metadata['doc_id']` trùng với `doc_id` được chỉ định, trả về `True` nếu số lượng chunk thực tế bị giảm đi.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Tác tử thực thi mô hình RAG 3 bước: đầu tiên gọi `store.search(question, top_k)` để thu thập các đoạn ngữ cảnh liên quan nhất, sau đó định dạng các chunk thành danh sách gạch đầu dòng và đưa vào template prompt `Context:\n...\n\nQuestion:...\nPlease answer...`. Cuối cùng, prompt hoàn chỉnh được chuyển tới hàm `llm_fn` để tạo ra câu trả lời được neo chắc chắn vào tài liệu.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0 -- /Users/namdev/Documents/Code/VinAI/K4-L3A-Data-Foundations/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/namdev/Documents/Code/VinAI/K4-L3A-Data-Foundations
configfile: pyproject.toml
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.03s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | Học viện xét cấp học bổng cho tân sinh viên | Chính sách học bổng dành cho sinh viên mới nhập học | cao | 0.1507 | Đúng |
| 2 | Học phí chương trình đào tạo kỹ sư công nghệ thông tin | Mức học phí các ngành kỹ thuật và phần mềm | cao | 0.0536 | Đúng |
| 3 | Quy định mượn sách thư viện trường đại học | Điều kiện đăng ký học phần tín chỉ trực tuyến | thấp | -0.0879 | Đúng |
| 4 | Sinh viên được miễn 100% học phí toàn khóa | Thủ tục xin cấp giấy chứng nhận kết quả tốt nghiệp | thấp | -0.0160 | Đúng |
| 5 | Viện Toán học cấp học bổng nghiên cứu sinh | Học bổng Sigma Gold cho sinh viên ngành Toán | cao | -0.1808 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất là cặp số 5 ("Viện Toán học cấp học bổng..." và "Học bổng Sigma Gold..."): dù có cùng ngữ cảnh về ngành Toán và học bổng, điểm thực tế lại là số âm (-0.1808). Điều này giải thích rõ cơ chế của `_mock_embed` (sử dụng hàm băm ký tự giản đơn) chưa thể nắm bắt được ngữ nghĩa học sâu như các mô hình ngôn ngữ thực thụ (như BERT/Sentence-Transformers); trong thực tế, các mô hình nhúng ngữ nghĩa chuyên dụng sẽ dựa vào không gian ẩn (latent space) được huấn luyện trên khối dữ liệu lớn để liên kết các khái niệm đồng nghĩa ngay cả khi không trùng lặp bề mặt ký tự.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|:---:|:---|:---|:---:|:---:|:---|
| 1 | Hạn chót nộp hồ sơ xét học bổng tân sinh viên K71 của VNUA là khi nào? | Quyết định xét cấp học bổng tân sinh viên K71 Học viện Nông nghiệp Việt Nam | 0.0515 | Có | Hạn nộp hồ sơ từ 30/05/2026 đến hết ngày 30/08/2026 tính theo dấu bưu điện. |
| 2 | Mức học bổng Sigma Gold của Viện Nghiên cứu cao cấp về Toán (VIASM) là bao nhiêu một tháng? | Học bổng Sigma Gold năm học 2026-2027 cho sinh viên ngành Toán - VIASM | 0.1204 | Có | Mức học bổng là 15.000.000 đồng/tháng, cấp 10 tháng/năm học (tối đa 10 suất). |
| 3 | Điều kiện để nhận học bổng CMC Khai Phóng miễn 100% học phí tại Đại học CMC? | Chính sách học bổng Đại học CMC: Học bổng Khai phóng 100% học phí toàn khóa | 0.0842 | Có | Giải HSG/KHKT quốc gia, quốc tế hoặc điểm thi THPT >= 35/40, IELTS >= 7.5 hoặc Toán HK1/cả năm lớp 12 >= 8.0. |
| 4 | Mức học bổng hỗ trợ hàng tháng cho sinh viên ngành Vi mạch bán dẫn tại HaUI là bao nhiêu? | Học bổng các ngành kỹ thuật then chốt và công nghệ chiến lược - ĐH Công nghiệp Hà Nội | 0.0763 | Có | Sinh viên ngành Vi mạch bán dẫn được nhận 4.200.000 đồng/tháng, cấp trong 10 tháng/năm học. |
| 5 | Sinh viên diện chính sách nào được miễn 100% học phí theo quy định tại HaUI? *(Lọc: audience=student)* | Chính sách miễn giảm học phí theo Nghị định số 238/2025/NĐ-CP tại HaUI | 0.0918 | Có | Người có công/thân nhân người có công, sinh viên mồ côi cả cha lẫn mẹ, DTTS thuộc hộ nghèo/cận nghèo hoặc DTTS rất ít người. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua quá trình thử nghiệm và trao đổi nhóm, tôi nhận thấy việc chia nhỏ văn bản theo cấu trúc mục/tiêu đề (heading/section chunking) giữ trọn vẹn được bảng điều kiện và mốc thời gian tốt hơn nhiều so với việc chỉ cắt theo số lượng ký tự cố định (fixed size). Ngoài ra, kết hợp lọc theo metadata (`audience`, `department`) giúp loại bỏ hoàn toàn các văn bản gây nhiễu trước khi tính điểm vector, gia tăng độ chính xác của câu trả lời từ tác tử RAG.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|:---|:---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
