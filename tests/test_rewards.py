from self_summarization_agent.rewards import answer_reward, incorrect_reward


def test_answer_reward_is_terminal_only() -> None:
    assert answer_reward("correct_answer") == 1.0
    assert answer_reward("wrong_answer") == -1.0
    assert incorrect_reward() == -1.0
