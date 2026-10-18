from pldc.store import Store


def test_chat_quiz_feedback_roundtrip(tmp_path):
    s = Store(tmp_path / "t.db")
    s.add_chat_message("a" * 32, 0, {"role": "user", "content": "xin chào"})
    s.add_chat_message("a" * 32, 1, {"role": "assistant", "content": "chào bạn",
                                      "sources": [{"citation": "c", "body": "b"}], "provider": "P"})
    chat = s.load_chat("a" * 32)
    assert [m["role"] for m in chat] == ["user", "assistant"]
    assert chat[1]["sources"][0]["citation"] == "c"
    assert s.load_chat("b" * 32) == []

    state = {"filters": [None, None, True], "order": ["q1"], "pos": 0,
             "submitted": {"q1": ["A"]}, "explanations": {"q1|A": {"text": "t"}}}
    s.save_quiz("a" * 32, state)
    assert s.load_quiz("a" * 32) == state

    s.save_feedback("a" * 32, "chat-1", "chat", 1, "q", "r", "P")
    s.save_feedback("a" * 32, "chat-1", "chat", 0, "q", "r", "P")  # changed mind
    assert s.get_rating("a" * 32, "chat-1") == 0

    s.clear_chat("a" * 32)
    assert s.load_chat("a" * 32) == []
