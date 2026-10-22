# Trợ lý Pháp luật đại cương (PTIT)

Hai tính năng, viết hoàn toàn bằng Python (giao diện Streamlit):

1. **Hỏi đáp (RAG):** trả lời câu hỏi dựa trên giáo trình và slide PTIT, có dẫn nguồn (mục, số trang, số slide).
2. **Luyện trắc nghiệm:** 676 câu có đáp án; sau khi nộp bài có nút **🤖 Hỏi AI vì sao** để giải thích đáp án đúng và các đáp án sai.

## Cài đặt và chạy

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
cp .env.example .env                      # điền GOOGLE_API_KEY

python src/parse_documents.py             # PDF -> data/processed/*.jsonl
python src/build_index.py                 # embed -> data/index/ (chỉ chạy lại khi dữ liệu đổi)

python ui/main.py                         # giao diện mới (NiceGUI) -> http://localhost:8080
streamlit run app/main.py                 # giao diện cũ (Streamlit)
```

### Giao diện mới (`ui/`, NiceGUI): "Vở bài tập và lời phê"
Người học làm bài bằng **mực tím**, máy chấm bằng **mực đỏ** trên trang vở có lề đỏ: dấu ✓/✗ ở lề,
khoanh tròn đáp án đúng (hiệu ứng duy nhất, tắt khi máy bật giảm chuyển động), lời giải như lời phê.
Chữ: Noto Serif (nội dung) + Be Vietnam Pro (giao diện), tự host trong `ui/fonts/` để luôn hiển thị đúng dấu tiếng Việt. Phím tắt: `A–D`/`1–4` chọn, `Enter` nộp,
`←/→` chuyển câu, `H` hỏi AI. Bấm số nguồn trong lời giải để mở trang PDF gốc ở bảng bên phải.
Màn **Hỏi đáp** (`/hoi-dap`) trình bày như sách hỏi đáp pháp luật: "Hỏi" mực tím, "Đáp" lề đỏ, số nguồn
bấm được; lịch sử lưu theo trình duyệt, câu hỏi gợi ý khi chưa có gì, nút thử lại khi AI lỗi.
Màn **Luyện tập** (`/`, `/?mode=review` cho Ôn câu sai) có tổng kết và làm lại câu sai.

| File | Vai trò |
|---|---|
| `ui/main.py` | Điểm khởi chạy, khóa ký cookie (`data/.ui_secret`, không commit) |
| `ui/theme.py` | Hệ thống thiết kế: màu, chữ, CSS, nét khoanh và dấu chấm bài (SVG) |
| `ui/shell.py` | Khung trang dùng chung: thanh bên (tên app, điều hướng), thanh menu trên điện thoại |
| `ui/quiz_page.py` | Màn luyện tập: bộ lọc, tờ bài, chấm, lời giải AI, tổng kết, phím tắt |
| `ui/chat_page.py` | Màn hỏi đáp: Hỏi/Đáp, stream câu trả lời, nguồn, đánh giá, thử lại |
| `ui/sources.py` | Số trích dẫn bấm được và bảng xem trang PDF gốc |
| `ui/services.py` | Tài nguyên dùng chung, mã người dùng (cookie), thông báo lỗi dễ hiểu |
| `src/pldc/practice.py` | Logic phiên luyện tập dùng chung cho mọi giao diện |

Test: `pytest` (test đơn vị) · `pytest -m live` (gọi API thật, kiểm tra chuỗi model Gemini).

## Kiến trúc

```
data/*.pdf ──parsers──▶ data/processed/*.jsonl ──chunking + Gemini embedding──▶ data/index/
                                                                                      │
                 Streamlit app/ ──▶ pldc.chat / pldc.quiz ──▶ pldc.search (BM25 + vector, RRF)
                                                 └──────────▶ pldc.llm (Gemini, nhiều model)
```

| Thành phần | File | Ghi chú |
|---|---|---|
| Parser PDF | `src/parsers/` | giáo trình theo mục, slide, 2 ngân hàng câu hỏi |
| Cấu hình | `src/pldc/config.py` | đọc `.env` |
| LLM | `src/pldc/llm.py` | lần lượt các model Gemini trong `GEMINI_MODELS`; lỗi kết nối / hết hạn mức / 429 / 5xx thì chuyển sang model kế tiếp |
| Embedding | `src/pldc/embeddings.py` | Gemini `gemini-embedding-001` |
| Tìm kiếm | `src/pldc/search.py` | BM25 (âm tiết + cặp âm tiết) kết hợp vector; không gọi được embedding thì chỉ dùng BM25 |
| Prompt | `src/pldc/prompts.py` | bắt buộc dẫn nguồn, nói rõ khi tài liệu không đề cập hoặc đáp án lỗi thời |
| Trắc nghiệm | `src/pldc/quiz.py` | áp dụng `data/answer_overrides.json`, bỏ câu không có đáp án |
| Lưu trữ | `src/pldc/store.py` | SQLite `data/app.db`: lịch sử chat, tiến độ trắc nghiệm, hàng ôn, đánh giá 👍/👎; khóa theo `?sid=` trên URL |
| Ôn câu sai | `src/pldc/review.py` | Hộp Leitner: câu sai ôn ngay → 1 → 3 → 7 → 16 ngày → đã thuộc |
| Trang PDF gốc | `src/pldc/pages.py` | Xác định trang chứa đoạn trích, render PNG cho cửa sổ xem nguồn |
| Giao diện | `app/`, `.streamlit/config.toml` | `main.py`, `chat_page.py`, `quiz_page.py`; theme đỏ PTIT, font Be Vietnam Pro |

## Lộ trình

- [x] Parser cho 4 tài liệu PDF
- [x] Khung RAG + chuyển lần lượt giữa các model Gemini
- [x] Giao diện hỏi đáp và luyện trắc nghiệm
- [ ] Script đánh giá: cho bot làm 676 câu trắc nghiệm, so sánh độ chính xác khi có và không có RAG
- [ ] Gán chương cho câu EduQuiz và slide (hiện `chapter: null`) để lọc theo chương
- [ ] Lưu lịch sử làm bài, thống kê chương hay sai
- [ ] Bổ sung văn bản luật hiện hành để cảnh báo phần giáo trình 2019 đã lỗi thời
- [ ] Xếp hạng lại kết quả tìm kiếm (reranker) nếu kết quả đánh giá cho thấy cần
