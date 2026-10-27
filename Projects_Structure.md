# Cấu trúc dự án — Trợ lý Pháp luật đại cương (PTIT)

Dự án gồm hai tính năng chính:

1. **Hỏi đáp (RAG):** trả lời câu hỏi dựa trên giáo trình và slide môn Pháp luật đại cương của PTIT, có dẫn nguồn (mục, số trang, số slide).
2. **Luyện trắc nghiệm:** 676 câu có đáp án, AI giải thích vì sao đáp án đúng và các đáp án khác sai, có chế độ ôn lại câu làm sai.

Luồng dữ liệu tổng quát:

```
data/*.pdf ──src/parsers──▶ data/processed/*.jsonl ──chunking + embedding──▶ data/index/
                                                                                 │
   ui/ (NiceGUI) hoặc app/ (Streamlit) ──▶ pldc.chat / pldc.quiz ──▶ pldc.search (BM25 + vector, RRF)
                                                    │                     └─▶ pldc.rerank (tùy chọn)
                                                    └──────────────▶ pldc.llm (Gemini, nhiều model)
```

---

## Cây thư mục

```
RAG_PLDC/
├── .env.example          # Mẫu biến môi trường
├── .gitignore
├── .streamlit/           # Cấu hình giao diện Streamlit
├── README.md             # Hướng dẫn cài đặt, chạy và kiến trúc
├── pyproject.toml        # Khai báo package và dependency
├── app/                  # Giao diện cũ (Streamlit)
├── ui/                   # Giao diện mới (NiceGUI)
├── src/
│   ├── parsers/          # Đọc PDF thành dữ liệu có cấu trúc
│   ├── pldc/             # Thư viện lõi: tìm kiếm, LLM, quiz, lưu trữ...
│   └── *.py              # Script dòng lệnh: parse, build index, đánh giá
├── data/                 # PDF gốc, dữ liệu đã xử lý, index, golden cases
├── models/               # Model ONNX chạy trên máy (không commit)
└── tests/                # Test bằng pytest
```

---

## File ở thư mục gốc

| File / thư mục | Mục đích |
|---|---|
| `README.md` | Giới thiệu, hướng dẫn cài đặt và chạy, bảng kiến trúc |
| `pyproject.toml` | Tên package `rag-pldc`, dependency chính (`google-genai`, `nicegui`, `streamlit`, `numpy`, `pdfminer.six`, `pypdfium2`...), nhóm tùy chọn `dev` (pytest) và `local` (onnxruntime, tokenizers); cấu hình pytest với marker `live` và `slow` |
| `.env.example` | Mẫu `.env`: `GOOGLE_API_KEY`, `GEMINI_MODELS`, `LLM_TIMEOUT`, `EMBEDDER`, `RERANKER`, `RERANK_CANDIDATES`... |
| `.env` | Khóa API thật, **không commit** |
| `.gitignore` | Bỏ qua secrets, venv, dữ liệu runtime (`data/app.db`, `data/.ui_secret`), `models/`, `data/eval/`, `data/index-bge-m3/` |
| `.streamlit/config.toml` | Theme Streamlit (font Be Vietnam Pro, màu đỏ chủ đạo cho sáng/tối) |
| `.venv/`, `.pytest_cache/`, `.nicegui/` | Môi trường ảo, cache pytest, dữ liệu runtime của NiceGUI (không commit) |

---

## `src/` — Mã nguồn

### `src/*.py` — Script dòng lệnh

| File | Mục đích |
|---|---|
| `parse_documents.py` | Đọc mọi PDF trong `data/`, chọn parser theo tên file, ghi ra `data/processed/*.jsonl` và báo các bản ghi có vấn đề |
| `build_index.py` | Chia đoạn và embed dữ liệu đã xử lý, ghi index. `--embedder gemini` → `data/index/`, `--embedder local` (bge-m3) → `data/index-bge-m3/` |
| `evaluate.py` | Đo độ chính xác trả lời trắc nghiệm theo 3 chế độ (`none`, `bm25`, `hybrid`), kết quả ở `data/eval/<timestamp>/` |
| `analyze_eval.py` | Phân tích sau một lượt đánh giá: `keys` tìm đáp án nghi sai, `support` đo ngữ cảnh tìm được có đủ căn cứ không |
| `eval_retrieval.py` | So sánh các cấu hình tìm kiếm (cách chia đoạn × BM25/hybrid × reranker) bằng hit@k, MRR; nhãn do LLM gán, cache ở `data/eval/retrieval_labels.jsonl` |

### `src/parsers/` — Parser PDF

| File | Mục đích |
|---|---|
| `common.py` | Hàm dùng chung: chuẩn hóa Unicode, dataclass bản ghi, `write_jsonl` |
| `textbook.py` | Giáo trình → các mục theo số hiệu ("1.1", "1.1.2"...). Lồng theo độ sâu thay vì giá trị vì giáo trình đánh số sai ở vài chỗ |
| `slides.py` | Slide bài giảng → mỗi slide một bản ghi, bỏ các dòng thừa của template (tiêu đề môn khác, "Trang N"...) |
| `mcq_color.py` | Ngân hàng câu hỏi có đáp án đúng **in màu đỏ**: đọc màu từng ký tự để xác định đáp án |
| `mcq_eduquiz.py` | Bản xuất EduQuiz (430 câu, đáp án đúng có dấu `*`): dựa vào bố cục dòng để tách câu hỏi và phương án |

### `src/pldc/` — Thư viện lõi (dùng chung cho mọi giao diện)

| File | Mục đích |
|---|---|
| `config.py` | Đọc `.env`, định nghĩa đường dẫn (`ROOT`, `DATA_DIR`, `PROCESSED_DIR`, `INDEX_DIR`...) và `get_settings()` |
| `llm.py` | Gọi Gemini theo chuỗi model trong `GEMINI_MODELS`; tự chuyển sang model kế tiếp khi lỗi kết nối, hết hạn mức, 429, 5xx, timeout. Khi stream chỉ fallback trước token đầu tiên |
| `embeddings.py` | Hai bộ embedding chung giao diện: `GeminiEmbedder` (API) và `LocalEmbedder` (bge-m3 qua onnxruntime, GPU DirectML hoặc CPU) |
| `chunking.py` | Chia mục giáo trình và slide thành đoạn ~1000 ký tự, cắt theo câu, gối 1–2 câu; mỗi đoạn mang đường dẫn mục/chủ đề |
| `search.py` | Tìm kiếm lai: BM25 (âm tiết + cặp âm tiết) kết hợp vector bằng Reciprocal Rank Fusion; mất embedding thì chỉ dùng BM25 |
| `rerank.py` | Xếp hạng lại bằng cross-encoder bge-reranker-v2-m3 (ONNX); `RERANKER=auto` chỉ bật khi có GPU |
| `prompts.py` | Prompt tiếng Việt: bắt buộc dẫn nguồn `[n]`, đưa tài liệu vào dạng dữ liệu tách biệt (chống prompt injection), nói rõ khi tài liệu không đề cập |
| `chat.py` | Hỏi đáp: tìm đoạn liên quan, ghép lịch sử hội thoại (tối đa 6 lượt), stream câu trả lời |
| `quiz.py` | Ngân hàng câu hỏi (`QuestionBank`, `BANKS`), áp dụng `data/answer_overrides.json`, sinh lời giải thích AI |
| `practice.py` | Logic phiên luyện tập độc lập giao diện: thứ tự câu, ghi đáp án, tổng kết, làm lại, chế độ ôn |
| `review.py` | Ôn câu sai theo hộp Leitner: sai → ôn ngay → 1 → 3 → 7 → 16 ngày → đã thuộc |
| `store.py` | Lưu SQLite (`data/app.db`): lịch sử chat, tiến độ trắc nghiệm, hàng ôn, đánh giá 👍/👎, theo mã phiên |
| `pages.py` | Tìm trang PDF chứa đoạn được trích và render PNG để người dùng đối chiếu nguồn gốc |
| `evaluation.py` | Logic đo độ chính xác trên ngân hàng câu hỏi (dùng bởi `src/evaluate.py`) |
| `judging.py` | LLM làm giám khảo: rà soát đáp án nghi sai (giải độc lập, không xem đáp án gốc) và đánh giá ngữ cảnh tìm được |
| `answer_checks.py` | Kiểm tra câu trả lời sinh ra: có đủ ý, trích dẫn hợp lệ và đúng mục, không lộ prompt; tùy chọn LLM chấm độ trung thực |

---

## `ui/` — Giao diện mới (NiceGUI): "Vở bài tập và lời phê"

Chạy: `python ui/main.py` → http://localhost:8080

| File | Mục đích |
|---|---|
| `main.py` | Điểm khởi chạy, đăng ký các trang, tạo khóa ký cookie `data/.ui_secret` |
| `theme.py` | Hệ thống thiết kế: màu (mực tím, mực đỏ, giấy kẻ lề đỏ), font, CSS, nét khoanh tròn và dấu chấm bài (SVG) |
| `shell.py` | Khung trang chung: thanh bên điều hướng (Luyện tập / Ôn câu sai / Hỏi đáp) và thanh menu trên điện thoại |
| `quiz_page.py` | Màn luyện tập (`/`, `/?mode=review`): bộ lọc, tờ bài, chấm, lời giải AI, tổng kết, phím tắt |
| `chat_page.py` | Màn hỏi đáp (`/hoi-dap`): bố cục Hỏi/Đáp, stream câu trả lời, nguồn, đánh giá, thử lại |
| `sources.py` | Biến `[n]` trong câu trả lời thành chú thích bấm được và bảng xem trang PDF gốc |
| `services.py` | Tài nguyên dùng chung toàn tiến trình (LLM, retriever, store, ngân hàng câu), mã người dùng qua cookie, thông báo lỗi dễ hiểu |
| `fonts/` | Font tự host (Be Vietnam Pro, Noto Serif) chia theo bộ ký tự latin / latin-ext / vietnamese, khai báo trong `fonts.css`, để luôn hiển thị đúng dấu tiếng Việt |

## `app/` — Giao diện cũ (Streamlit)

Chạy: `streamlit run app/main.py`

| File | Mục đích |
|---|---|
| `main.py` | Điểm khởi chạy, điều hướng giữa hai trang |
| `chat_page.py` | Trang hỏi đáp, có câu hỏi gợi ý |
| `quiz_page.py` | Trang luyện trắc nghiệm: thẻ đáp án, "Hỏi AI", ôn câu sai, biểu đồ tổng kết (Altair) |
| `services.py` | Tài nguyên dùng chung và hàm hỗ trợ giao diện cho Streamlit (mã phiên `?sid=`, hiển thị nguồn, lỗi) |

---

## `data/` — Dữ liệu

| File / thư mục | Mục đích |
|---|---|
| `giaotrinh_pldc_baigiang2019.pdf` | Giáo trình bài giảng → `textbook_sections.jsonl` |
| `giaotrinh_pldc_slide_keywords.pdf` | Slide bài giảng → `slides.jsonl` |
| `cauhoi_tno_abcd.pdf` | Ngân hàng trắc nghiệm (đáp án in đỏ) → `questions_nganhang_tn.jsonl` |
| `916930112-Phap-Luật-Đại-Cương-PTIT-Mới-Nhất-2025.pdf` | Bộ câu hỏi EduQuiz → `questions_eduquiz.jsonl` |
| `answer_overrides.json` | Sửa đáp án/phương án thủ công theo mã câu, có `source` (đã đối chiếu văn bản luật) và `confidence` |
| `processed/` | Kết quả parse dạng JSONL. Được commit để không phải parse lại |
| `index/` | Index tìm kiếm dùng Gemini: `chunks.jsonl`, `embeddings.npy`, `manifest.json` (hash dữ liệu, model embedding, số đoạn). Được commit vì build lại tốn gần hết hạn mức miễn phí trong ngày |
| `index-bge-m3/` | Index tương tự dùng bge-m3 chạy máy (không commit) |
| `golden/chat_cases.jsonl` | Câu hỏi mẫu cho test hỏi đáp: loại `grounded` (mục cần trích, ý bắt buộc), `out_of_scope`, `injection` |
| `golden/explain_cases.jsonl` | Câu mẫu cho test lời giải thích trắc nghiệm |
| `eval/` | Kết quả các lượt đánh giá (không commit) |
| `app.db`, `.ui_secret` | Cơ sở dữ liệu người dùng và khóa cookie lúc chạy (không commit) |

## `models/` — Model chạy trên máy (không commit, tùy chọn)

| Thư mục | Mục đích |
|---|---|
| `bge-m3/` | Bản ONNX của BAAI/bge-m3, dùng khi `EMBEDDER=local` |
| `bge-reranker-v2-m3/` | Reranker ONNX (`model_fp16.onnx` cho GPU, `model_int8.onnx` cho CPU) kèm tokenizer |

---

## `tests/` — Kiểm thử

Chạy `pytest` cho test đơn vị; `pytest -m live` gọi API thật; `pytest -m slow` nạp model local.

| File | Kiểm tra |
|---|---|
| `golden.py` | Hàm nạp golden cases từ `data/golden/` (dùng chung) |
| `test_core.py` | Tách từ, BM25, chia đoạn, retriever, ngân hàng câu hỏi |
| `test_store.py` | Ghi/đọc SQLite: chat, tiến độ, đánh giá |
| `test_review.py` | Lịch ôn Leitner |
| `test_practice.py` | Luồng một phiên luyện tập |
| `test_llm_fallback.py` | Thứ tự fallback giữa các model, chỉ fallback với lỗi "không khả dụng" |
| `test_citations.py` | Đánh số chú thích theo thứ tự xuất hiện và ánh xạ về nguồn |
| `test_answer_checks.py` | Bản thân các hàm kiểm tra câu trả lời phải đúng |
| `test_evaluation.py` | Đọc chữ cái đáp án từ câu trả lời của model, lấy mẫu, tổng hợp |
| `test_judging.py` | Phân loại kết luận khi rà đáp án (`KEY_WRONG`, `KEY_RIGHT`, `UNCLEAR`) |
| `test_golden_retrieval.py` | Chất lượng tìm kiếm trên golden cases (offline, BM25); biến thể bge-m3 chạy với `-m slow` |
| `test_golden_answers.py` | Chất lượng trả lời đầu-cuối với Gemini thật (`-m live`) |
| `test_local_models.py` | Model ONNX dùng đồng thời từ nhiều luồng (`-m slow`) |
