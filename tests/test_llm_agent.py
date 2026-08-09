import json
from pathlib import Path

from contract_heldout_ipi.agent import (
    EpisodeRunner,
    LLMAgent,
    ModelResponse,
    ModelToolCall,
)
from contract_heldout_ipi.contract.loader import load_episode
from contract_heldout_ipi.defenses import NoDefense, SystemPromptWarning

ROOT = Path(__file__).resolve().parents[1]
EPISODE_PATH = (
    ROOT / "episodes" / "train" / "email_meeting_summary_exfil_001.json"
)


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append({"messages": messages, "tools": tools})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_llm_agent_round_trips_tool_result_and_completes():
    episode = load_episode(EPISODE_PATH)
    client = FakeChatClient(
        [
            ModelResponse(
                tool_call=ModelToolCall(
                    id="call-1",
                    name="read_email",
                    arguments={"email_id": "e1"},
                )
            ),
            ModelResponse(
                content="Friday 10:00 in Room A. Participants: Alex, Bob, Carol."
            ),
        ]
    )

    run = EpisodeRunner().run(episode, LLMAgent(client, NoDefense()))

    assert run.terminated_reason == "completed"
    assert run.tool_calls[0].succeeded
    assert len(client.requests) == 2
    second_messages = client.requests[1]["messages"]
    tool_message = next(m for m in second_messages if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call-1"
    assert "Friday 10:00" in tool_message["content"]


def test_llm_agent_model_view_excludes_hidden_evaluation_fields():
    episode = load_episode(EPISODE_PATH)
    client = FakeChatClient([ModelResponse(content="No result.")])

    EpisodeRunner().run(episode, LLMAgent(client, SystemPromptWarning()))

    serialized_request = json.dumps(client.requests[0])
    assert episode.episode_id not in serialized_request
    assert episode.attacker_goal not in serialized_request
    assert "required_strings" not in serialized_request
    assert episode.untrusted_context.content["body"] not in serialized_request
    assert {tool["function"]["name"] for tool in client.requests[0]["tools"]} == {
        tool.name for tool in episode.tools
    }


def test_llm_agent_can_observe_failed_tool_call_and_continue():
    episode = load_episode(EPISODE_PATH)
    client = FakeChatClient(
        [
            ModelResponse(
                tool_call=ModelToolCall(
                    id="call-missing",
                    name="read_email",
                    arguments={"email_id": "missing"},
                )
            ),
            ModelResponse(content="The requested email was not found."),
        ]
    )

    run = EpisodeRunner().run(episode, LLMAgent(client))

    assert run.terminated_reason == "completed"
    observation = next(
        m for m in client.requests[1]["messages"] if m["role"] == "tool"
    )
    assert '"ok": false' in observation["content"]
    assert "email not found" in observation["content"]


def test_model_exception_is_recorded_as_agent_error():
    episode = load_episode(EPISODE_PATH)
    client = FakeChatClient([RuntimeError("model unavailable")])

    run = EpisodeRunner().run(episode, LLMAgent(client))

    assert run.terminated_reason == "agent_error"
    assert run.agent_error == "RuntimeError: model unavailable"


def test_llm_agent_executes_parallel_tool_calls_before_next_model_request():
    episode = load_episode(EPISODE_PATH)
    client = FakeChatClient(
        [
            ModelResponse(
                tool_calls=(
                    ModelToolCall("call-1", "read_email", {"email_id": "e1"}),
                    ModelToolCall("call-2", "read_email", {"email_id": "e2"}),
                )
            ),
            ModelResponse(
                content="Friday 10:00 in Room A. Participants: Alex, Bob, Carol."
            ),
        ]
    )

    run = EpisodeRunner().run(episode, LLMAgent(client))

    assert run.terminated_reason == "completed"
    assert [call.arguments["email_id"] for call in run.tool_calls] == ["e1", "e2"]
    assert len(client.requests) == 2
    second_messages = client.requests[1]["messages"]
    assert [message["tool_call_id"] for message in second_messages if message["role"] == "tool"] == [
        "call-1",
        "call-2",
    ]
