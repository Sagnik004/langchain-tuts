from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ) -> Any:
        """Run when LLM starts running."""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("************************")

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs) -> Any:
        """Run when LLM ends running."""
        print(f"***LLM response:***\n{response.generations[0][0].text}")
        print("************************")
