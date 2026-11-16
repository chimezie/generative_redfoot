import pytest

from generative_redfoot import prompts


@pytest.fixture(autouse=True)
def fresh_prompt_cache():
    # Loom files are rewritten between tests; never let a cached parse leak across them
    prompts.clear_cache()
    yield
    prompts.clear_cache()
