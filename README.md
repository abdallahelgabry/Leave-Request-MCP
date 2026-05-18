# Leave Request Assistant MCP


## Project Structure

```
leave-request-mcp/
├── server.py          # FastMCP server — tools & in-memory DB
├── app.py             # Streamlit client — chat UI + AI routing
├── .env               # Environment variables (see setup below)
└── requirements.txt   # Python dependencies
```


## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/abdallahelgabry/Leave-Request-MCP.git
cd leave-request-mcp
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the MCP Server

```bash
python server.py
```

The server starts on `http://0.0.0.0:8001` using SSE transport.

### 4. Run the Streamlit App

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



## Test Employee IDs

| ID | Name | Department |
|----|------|------------|
| EMP001 | Abdullah Elgabry | Engineering |
| EMP002 | Laila Zaki | HR |
| EMP003 | Amr Mohamed | Finance |



## Observability with LangSmith

All interactions are traced via [LangSmith](https://smith.langchain.com/), including:

- `load_prompt` — fetching system prompts from LangSmith Prompt Hub
- `get_mcp_tools` — tool discovery from the MCP server
- `execute_mcp_tool` — individual tool executions
- `chat_openai` / `chat_cohere` — full chat chains
- `chat_router` — model routing logic

Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in your `.env` to enable tracing.


## Example Conversation

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

