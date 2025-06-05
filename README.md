# Ravana

**Ravana** is an experimental agentic system built using the [FastMCP](https://github.com/epfml/fastmcp) framework. It is designed to explore tabular datasets, uncover trends, and iteratively improve its insights using gradient boosting models like XGBoost or LightGBM.

Inspired by multi-perspective reasoning and recursive self-critique, Ravana employs multiple agents to simulate a collaborative analysis process—recommending, evaluating, and refining machine learning configurations in pursuit of optimal predictive performance.

**UNDER CONSTRUCTION**

---

## ✨ Project Goals

- Implement a simple AutoML-style loop with analyst and critic agents
- Use XGBoost or LightGBM as configurable, tunable tools
- Focus on explainability and iterative improvement based on agent dialogue

---

## 🧠 Core Design

- **Agent: Data Analyst**  
  Analyzes the uploaded CSV file, summarizes the structure, and recommends initial model parameters.

- **Tool: Boosting Model Runner**  
  Wraps a LightGBM or XGBoost call using a configurable schema; returns performance metrics and model artifacts.

- **Agent: Critic**  
  Reviews model performance and recommends changes. If improvements are likely, the system loops and re-trains.

- **MCP Server**  
  Orchestrates the flow between agents and tools using standardized message schemas via FastMCP.

---

