from contract_heldout_ipi.eval.trace import AgentStep, EpisodeRun, ToolCallRecord


def test_episode_run_exposes_tool_calls_in_step_order():
    first = ToolCallRecord(tool="list_emails", arguments={}, result=[])
    second = ToolCallRecord(
        tool="read_email",
        arguments={"email_id": "missing"},
        error="email not found: missing",
    )
    run = EpisodeRun(
        episode_id="episode-1",
        final_answer="No matching email was found.",
        steps=[
            AgentStep(index=0, tool_call=first),
            AgentStep(index=1, content="Trying the selected message.", tool_call=second),
            AgentStep(index=2, content="Done."),
        ],
    )

    assert run.tool_calls == [first, second]
    assert run.tool_call_count == 2
    assert first.succeeded
    assert not second.succeeded


def test_episode_run_defaults_to_completed_without_steps():
    run = EpisodeRun(episode_id="episode-1", final_answer="Done")

    assert run.tool_calls == []
    assert run.tool_call_count == 0
    assert run.terminated_reason == "completed"
