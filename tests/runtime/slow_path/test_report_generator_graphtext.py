from graphslm_ids.runtime.slow_path.report_generator import (
    ReportGenerator, build_repair_prompt, build_user_prompt_from_graphtext,
)

GRAPH_TEXT = "## ALERT A_001 — HGT decision\npred=SqlInjection conf=0.88 [E_ALERT]\n## NODES\n..."


def test_user_prompt_embeds_graph_text():
    prompt = build_user_prompt_from_graphtext(GRAPH_TEXT, alert_id="A_001")
    assert GRAPH_TEXT in prompt
    assert "A_001" in prompt
    assert "[E_" in prompt  # instructs handle citation


def test_repair_prompt_lists_failing_claims():
    prompt = build_repair_prompt(GRAPH_TEXT, ["claim X is wrong", "claim Y unverified"])
    assert "claim X is wrong" in prompt and "claim Y unverified" in prompt
    assert GRAPH_TEXT in prompt


def test_generate_calls_injected_callable_with_graph_text():
    seen = {}

    def fake_slm(system, user):
        seen["system"] = system
        seen["user"] = user
        return "# XAI Report - A_001\nok [E_ALERT]"

    gen = ReportGenerator(slm_callable=fake_slm)
    out = gen.generate_from_graphtext(GRAPH_TEXT, alert_id="A_001")
    assert out.startswith("# XAI Report")
    assert GRAPH_TEXT in seen["user"]
    assert "HGT" in seen["system"]
