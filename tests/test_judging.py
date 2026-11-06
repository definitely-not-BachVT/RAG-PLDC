from pldc.judging import classify


def test_classify():
    key = ["C"]
    models = [["B"], ["B"], ["B"]]
    assert classify(key, models, ["B"], "cao") == "KEY_WRONG"  # judge sides with the models
    assert classify(key, models, ["C"], "cao") == "KEY_RIGHT"  # judge sides with the key
    assert classify(key, models, ["D"], "cao") == "UNCLEAR"  # a third answer
    assert classify(key, models, ["B"], "thap") == "UNCLEAR"  # low confidence
    assert classify(key, models, [], "cao") == "UNCLEAR"
