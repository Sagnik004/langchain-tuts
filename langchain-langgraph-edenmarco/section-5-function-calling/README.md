# Section-5: Function Calling aka Tool Calling

## Important Learnings

- **Function Calling in LLM**: Function calling allows an LLM to identify when a user's intent requires an external tool or API call and then generate a structured output (often in JSON format) containing the function name and its arguments. This output is then passed to an external application for execution. The LLM's role is to understand the user's request and determine the appropriate function and its parameters, but it does not execute the function itself.
- Function Calling may not be available for all LLM's, but all major vendors like OpenAI, Google, Anthropic, Mistral does provide it in their most models (usually the state of the art models). Nowadays this is quite the standard for state of the art models.
- Function Calling was introduced in 2023 by OpenAI.

## Course Useful Packages, Tools Used

- The package manager used here is [uv](https://docs.astral.sh/uv/), this is similar to `pip` but `uv` is much faster since it is built using Rust.

## Other Useful Resources

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [The first function calling](https://openai.com/index/function-calling-and-other-api-updates/)

### Function Calling Sample Execution Flow
![Function Calling Sample Execution Flow](/langchain-langgraph-edenmarco/section-5-function-calling/images/function_calling_sample_execution_flow.png)

### Function Calling Advantages
![Function Calling Advantages](/langchain-langgraph-edenmarco/section-5-function-calling/images/function_calling_advantages.png)

### Function Calling Drawbacks
![Function Calling Drawbacks](/langchain-langgraph-edenmarco/section-5-function-calling/images/function_calling_drawback.png)

### LangChain Function Calling Interface
![LangChain Function Calling Interface](/langchain-langgraph-edenmarco/section-5-function-calling/images/langchain_function_calling_interface.png)
