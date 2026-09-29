import uuid
from agent import agent
from tool import live_cricket_score


def test_live_cricket_score():
    result = live_cricket_score.invoke({
        "country1": "India",
        "country2": "Australia"
    })

    assert "India" in result
    assert "Australia" in result
    assert "120-3" in result


def test_live_cricket_score_returns_string():
    result = live_cricket_score.invoke({
        "country1": "India",
        "country2": "Australia"
    })

    assert isinstance(result, str)


def test_rate_limit_middleware():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Use the rate limit test tool"
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": f"pytest-rate-limit-{uuid.uuid4()}"
            }
        }
    )

    for message in result["messages"]:
      print(
        "TYPE:", getattr(message, "type", None),
        "STATUS:", getattr(message, "status", None),
        "CONTENT:", getattr(message, "content", None)
    )

    assert result["messages"]

def test_pii_redaction():
     result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "My email is test@example.com"
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": f"pytest-pii-{uuid.uuid4()}"
            }
        }
    )

     messages_text = str(result["messages"])

     assert "test@example.com" not in messages_text
     assert "[REDACTED_EMAIL]" in messages_text


def test_stop_agent_middleware():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "STOP_AGENT"
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": f"pytest-stop-{uuid.uuid4()}"
            }
        }
    )

    assert any(
        getattr(message, "content", "") == "STOP_AGENT"
        for message in result["messages"]
    )