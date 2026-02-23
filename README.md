# Leave Request Assistant — MCP
---

## A conversational AI assistant for submitting employee leave requests, built with FastMCP, OpenAI GPT-4.1, and Oracle Cloud Cohere. Employees chat naturally to check balances, prepare, and confirm leave requests — with a two-step confirmation flow, multi-model support, and full LangSmith observability. System prompts are managed directly on LangSmith Prompt Hub, making the entire pipeline fully automated — no code changes needed to update AI behavior. The architecture is designed for easy extensibility: to add any new feature, clients simply register a new tool in the MCP server and it becomes instantly available to the AI without touching the client application.


## Project Structure

```
leave-request-mcp/
├── server.py          # FastMCP server — tools & in-memory DB
├── app.py             # Streamlit client — chat UI + AI routing
├── .env               # Environment variables (see setup below)
└── requirements.txt   # Python dependencies
```
---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/leave-request-mcp.git
cd leave-request-mcp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root directory:

```env
# MCP Server
MCP_SERVER_URL=http://localhost:8001/sse

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Oracle Cloud Infrastructure (OCI) — Cohere
OCI_CONFIG_PATH="your-oci-config-path"
OCI_CONFIG_PROFILE=DEFAULT
OCI_COMPARTMENT_ID="your_compartment_id"
OCI_SERVICE_ENDPOINT="your-oci-service-endpoint"
OCI_COHERE_MODEL_ID= "cohere.command-a-03-2025"

# LangSmith Tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=leave-request-mcp
```

### 4. Run the MCP Server

```bash
python server.py
```

The server starts on `http://0.0.0.0:8001` using SSE transport.

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

---

##  MCP Tools

| Tool | Description |
|------|-------------|
| `get_current_date` | Returns today's date — used for relative date parsing (e.g. "tomorrow", "next week") |
| `check_leave_balance` | Returns annual and sick leave balances for an employee |
| `prepare_leave_request` | Validates and stages a leave request for review before submission |
| `confirm_leave_request` | Finalizes and submits a staged request after user confirmation |

---

## Test Employee IDs

| ID | Name | Department |
|----|------|------------|
| EMP001 | Abdullah Elgabry | Engineering |
| EMP002 | Laila Zaki | HR |
| EMP003 | Amr Mohamed | Finance |

---

## Supported AI Models

You can switch between models from the sidebar in the UI:

- **OpenAI** — GPT-4.1 via the OpenAI API
- **Cohere** — Command R+ via Oracle Cloud Generative AI (OCI)

Switching models automatically clears the conversation history.

---

## Observability with LangSmith

All interactions are traced via [LangSmith](https://smith.langchain.com/), including:

- `load_prompt` — fetching system prompts from LangSmith Prompt Hub
- `get_mcp_tools` — tool discovery from the MCP server
- `execute_mcp_tool` — individual tool executions
- `chat_openai` / `chat_cohere` — full chat chains
- `chat_router` — model routing logic

Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in your `.env` to enable tracing.

---

## 💬 Example Conversation

```
User:   What's my leave balance?
Bot:    You have 21 annual leave days and 10 sick leave days remaining.

User:   I'd like to request 3 days of annual leave starting tomorrow.
Bot:    Here's your leave request summary:
        - Type: Annual
        - From: 2025-02-24 to 2025-02-26
        - Days: 3
        - Remaining after approval: 18 days
        Type 'confirm' to submit.

User:   confirm
Bot:    Leave request REQ1001 submitted successfully!
```

---