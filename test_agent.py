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

    response = result["messages"][-1].content

    assert "rate-limited" in response.lower()
    assert "try again later" in response.lower()