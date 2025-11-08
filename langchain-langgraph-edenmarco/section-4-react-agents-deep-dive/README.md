# Section-4: ReAct Agents Deep Dive

## Important Learnings

- **AgentExecutor in LangChain**: it serves as the orchestrator/runtime for agents, enabling them to leverage external tools and perform multi-step reasoning to achieve a goal. In other words, **it is just a fancy while loop**.

## Course Useful Packages, Tools Used

- The package manager used here is [uv](https://docs.astral.sh/uv/), this is similar to `pip` but `uv` is much faster since it is built using Rust.

## Other Useful Resources

- [Tools](https://docs.langchain.com/oss/python/langchain/tools)

### ReAct Agent Framework under the hood
![ReAct Agent Framework under the hood](/images/react-agent-under-the-hood.png)
