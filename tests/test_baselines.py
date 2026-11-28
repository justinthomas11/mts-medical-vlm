import pandas as pd

from src.pipeline.run_baselines import baseline_records, most_frequent_report


def test_most_frequent_report_is_deterministic_on_ties():
    assert most_frequent_report(["b", "a", "b ", "c"]) == "b"
    assert most_frequent_report(["z", "y"]) == "y"


def test_baseline_records_use_template_and_top1_train_report():
    retrieval = pd.DataFrame({"uid": [1, 2], "retrieved_uid_1": [10, 11], "retrieved_uid_2": [11, 10],
                              "retrieved_uid_3": [12, 12]})
    reports = {10: "report ten ", 11: "report eleven", 12: "x"}
    out = baseline_records("test", [1, 2], reports, retrieval, "normal", k=3, template_any="other")
    assert [r["generated_report"] for r in out["template"]] == ["normal", "normal"]
    assert [r["generated_report"] for r in out["template_any"]] == ["other", "other"]
    assert [r["generated_report"] for r in out["retrieval_top1"]] == ["report ten", "report eleven"]
    assert out["retrieval_top1"][0]["retrieved_uids"] == [10, 11, 12]
    assert out["template"][0]["retrieved_uids"] == [] and out["template"][0]["split"] == "test"
