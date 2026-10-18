# Trợ lý Pháp luật đại cương (PTIT)

Hai tính năng, viết hoàn toàn bằng Python (giao diện Streamlit):

1. **Hỏi đáp (RAG):** trả lời câu hỏi dựa trên giáo trình và slide PTIT, có dẫn nguồn (mục, số trang, số slide).
2. **Luyện trắc nghiệm:** 676 câu có đáp án; sau khi nộp bài có nút **🤖 Hỏi AI vì sao** để giải thích đáp án đúng và các đáp án sai.

## Cài đặt và chạy

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
cp .env.example .env                      # điền DEEPSEEK_API_KEY, GOOGLE_API_KEY

python src/parse_documents.py             # PDF -> data/processed/*.jsonl
python src/build_index.py                 # embed -> data/index/ (chỉ chạy lại khi dữ liệu đổi)
streamlit run app/main.py
```

Test: `pytest` (test đơn vị) · `pytest -m live` (gọi API thật, kiểm tra cơ chế chuyển DeepSeek → Gemini).

## Kiến trúc

```
data/*.pdf ──parsers──▶ data/processed/*.jsonl ──chunking + Gemini embedding──▶ data/index/
                                                                                      │
                 Streamlit app/ ──▶ pldc.chat / pldc.quiz ──▶ pldc.search (BM25 + vector, RRF)
                                                 └──────────▶ pldc.llm (DeepSeek → Gemini)
```

| Thành phần | File | Ghi chú |
|---|---|---|
| Parser PDF | `src/parsers/` | giáo trình theo mục, slide, 2 ngân hàng câu hỏi |
| Cấu hình | `src/pldc/config.py` | đọc `.env` |
| LLM | `src/pldc/llm.py` | ưu tiên DeepSeek, lỗi kết nối / hết hạn mức / 429 / 5xx thì chuyển sang lần lượt các model Gemini |
| Embedding | `src/pldc/embeddings.py` | Gemini `gemini-embedding-001` (DeepSeek không có API embedding) |
| Tìm kiếm | `src/pldc/search.py` | BM25 (âm tiết + cặp âm tiết) kết hợp vector; không gọi được embedding thì chỉ dùng BM25 |
| Prompt | `src/pldc/prompts.py` | bắt buộc dẫn nguồn, nói rõ khi tài liệu không đề cập hoặc đáp án lỗi thời |
| Trắc nghiệm | `src/pldc/quiz.py` | áp dụng `data/answer_overrides.json`, bỏ câu không có đáp án |
| Lưu trữ | `src/pldc/store.py` | SQLite `data/app.db`: lịch sử chat, tiến độ trắc nghiệm, hàng ôn, đánh giá 👍/👎; khóa theo `?sid=` trên URL |
| Ôn câu sai | `src/pldc/review.py` | Hộp Leitner: câu sai ôn ngay → 1 → 3 → 7 → 16 ngày → đã thuộc |
| Trang PDF gốc | `src/pldc/pages.py` | Xác định trang chứa đoạn trích, render PNG cho cửa sổ xem nguồn |
| Giao diện | `app/`, `.streamlit/config.toml` | `main.py`, `chat_page.py`, `quiz_page.py`; theme đỏ PTIT, font Be Vietnam Pro |

## Lộ trình

- [x] Parser cho 4 tài liệu PDF
- [x] Khung RAG + chuyển nhà cung cấp DeepSeek → Gemini
- [x] Giao diện hỏi đáp và luyện trắc nghiệm
- [ ] Script đánh giá: cho bot làm 676 câu trắc nghiệm, so sánh độ chính xác khi có và không có RAG
- [ ] Gán chương cho câu EduQuiz và slide (hiện `chapter: null`) để lọc theo chương
- [ ] Lưu lịch sử làm bài, thống kê chương hay sai
- [ ] Bổ sung văn bản luật hiện hành để cảnh báo phần giáo trình 2019 đã lỗi thời
- [ ] Xếp hạng lại kết quả tìm kiếm (reranker) nếu kết quả đánh giá cho thấy cần
