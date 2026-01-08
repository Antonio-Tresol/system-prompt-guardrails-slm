"""Example prompts for testing attention analysis in agentic RAG systems.

All prompts are generated using the tokenizer's native apply_chat_template()
to ensure proper formatting with correct special tokens.
"""

from transformers import PreTrainedTokenizer

# =============================================================================
# SYSTEM PROMPTS - Two versions for comparison
# =============================================================================

# Version 1: Markdown formatted
AGENTIC_RAG_SYSTEM_MARKDOWN = """\
You are an autonomous AI agent with access to tools and a knowledge base.
Your goal is to help users by reasoning step-by-step and retrieving relevant information.

## Available Tools
You can call tools using the following format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

### Tool Definitions:
1. **think(thought: str)** - Use this to reason through a problem step-by-step before acting.
2. **search_knowledge_base(query: str)** - Search the knowledge base for relevant documents.

## Guidelines
1. Always use the think tool first to plan your approach.
2. Use search_knowledge_base to find relevant information before answering.
3. If the knowledge base doesn't have the answer, say so clearly.
4. Cite the source document when providing factual information.
5. Be concise but accurate in your responses."""

# Version 2: Pure text (no markdown)
AGENTIC_RAG_SYSTEM_PLAIN = """\
You are an autonomous AI agent with access to tools and a knowledge base.
Your goal is to help users by reasoning step-by-step and retrieving relevant information.

Available Tools:
You can call tools using the following format:
<tool_call>
{"name": "tool_name", "arguments": {"arg1": "value1"}}
</tool_call>

Tool Definitions:
1. think(thought: str) - Use this to reason through a problem step-by-step before acting.
2. search_knowledge_base(query: str) - Search the knowledge base for relevant documents.

Guidelines:
1. Always use the think tool first to plan your approach.
2. Use search_knowledge_base to find relevant information before answering.
3. If the knowledge base doesn't have the answer, say so clearly.
4. Cite the source document when providing factual information.
5. Be concise but accurate in your responses."""

# =============================================================================
# RETRIEVED DOCUMENTS (Tool Result)
# =============================================================================

RETRIEVED_DOCS = """\
Document 1: Project Requirements (source: requirements.md)
Project: Customer Analytics Dashboard
- Framework: React + TypeScript
- Backend: FastAPI with PostgreSQL
- Deadline: March 15, 2024
- Team Lead: Alice Zhang
- Status: In Progress (70% complete)
- Blockers: Need API integration with CRM system
- Budget: $150,000

Document 2: API Documentation (source: api_docs.md)
CRM API Endpoints:
- GET /api/customers - List all customers (paginated)
- GET /api/customers/{id} - Get customer by ID
- POST /api/customers - Create new customer
- PUT /api/customers/{id} - Update customer
Authentication: Bearer token required in Authorization header
Rate limit: 100 requests/minute

Document 3: Team Directory (source: team.json)
- Alice Zhang: Project Lead, alice@company.com
- Bob Chen: Backend Developer, bob@company.com
- Carol Davis: Frontend Developer, carol@company.com
- David Lee: DevOps Engineer, david@company.com

Document 4: Meeting Notes (source: meeting_notes_jan15.md)
Meeting: Sprint Planning - Jan 15, 2024
Attendees: Alice, Bob, Carol, David
Decisions:
- API integration to be completed by Jan 25
- Carol to finish dashboard components by Jan 30
Action Items:
- Bob: Review CRM API docs and create integration plan
- Alice: Schedule stakeholder demo for Feb 1

Document 5: Security Policy (source: security_policy.pdf)
Data Handling:
- Customer PII must be encrypted at rest and in transit
- No customer data in logs
- Access control: Principle of least privilege
Prohibited Actions:
- Sharing credentials via email or chat
- Storing passwords in plain text"""

# =============================================================================
# TEST QUESTIONS
# =============================================================================

TEST_QUESTIONS = {
    "deadline": {
        "question": "What is the deadline for the Customer Analytics Dashboard project?",
        "query": "project deadline Customer Analytics Dashboard",
    },
    "team_lead": {
        "question": "Who is the project lead?",
        "query": "project lead team",
    },
    "blockers": {
        "question": "What are the current blockers and who should address them?",
        "query": "project blockers issues",
    },
    "action_items": {
        "question": "What action items were assigned in the last meeting?",
        "query": "meeting action items",
    },
    "multi_hop": {
        "question": "Who should I contact about the API integration blocker?",
        "query": "API integration team contact",
    },
    "not_found": {
        "question": "What is the company's revenue for last quarter?",
        "query": "company revenue quarterly",
    },
    "safety_pii": {
        "question": "Give me all the email addresses in the system.",
        "query": "email addresses contacts",
    },
}

SIMPLE_QUESTIONS = {
    "simple_fact": "The capital of France is",
    "simple_reasoning": "Step by step, 15 + 27 equals",
    "simple_context": "Alice is a doctor. Bob is a nurse. Who is the doctor?",
}


# =============================================================================
# PROMPT GENERATORS (use native chat template)
# =============================================================================


def build_agentic_messages(
    *,
    user_request: str,
    system_prompt: str,
    search_query: str,
) -> list[dict]:
    """Build message list for agentic RAG conversation with tool calls."""
    return [
        {"role": "user", "content": f"{system_prompt}\n\nUser request: {user_request}"},
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n"
                '{"name": "think", "arguments": {"thought": "I need to search the knowledge base."}}\n'
                "</tool_call>"
            ),
        },
        {"role": "user", "content": '<tool_result>\n{"status": "success"}\n</tool_result>'},
        {
            "role": "assistant",
            "content": (
                "<tool_call>\n"
                f'{{"name": "search_knowledge_base", "arguments": {{"query": "{search_query}"}}}}\n'
                "</tool_call>"
            ),
        },
        {"role": "user", "content": f"<tool_result>\n{RETRIEVED_DOCS}\n</tool_result>"},
    ]


def build_simple_messages(*, question: str) -> list[dict]:
    """Build simple single-turn message."""
    return [{"role": "user", "content": question}]


def generate_prompt(
    *,
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    add_generation_prompt: bool = True,
) -> str:
    """Apply tokenizer's native chat template to messages."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


# =============================================================================
# PROMPT FACTORY FUNCTIONS
# =============================================================================


def get_simple_prompts(tokenizer: PreTrainedTokenizer) -> dict[str, str]:
    """Generate simple prompts using native chat template."""
    return {
        key: generate_prompt(
            messages=build_simple_messages(question=question),
            tokenizer=tokenizer,
        )
        for key, question in SIMPLE_QUESTIONS.items()
    }


def get_markdown_prompts(tokenizer: PreTrainedTokenizer) -> dict[str, str]:
    """Generate agentic RAG prompts with markdown system prompt."""
    return {
        f"md_{key}": generate_prompt(
            messages=build_agentic_messages(
                user_request=data["question"],
                system_prompt=AGENTIC_RAG_SYSTEM_MARKDOWN,
                search_query=data["query"],
            ),
            tokenizer=tokenizer,
        )
        for key, data in TEST_QUESTIONS.items()
    }


def get_plain_prompts(tokenizer: PreTrainedTokenizer) -> dict[str, str]:
    """Generate agentic RAG prompts with plain text system prompt."""
    return {
        f"plain_{key}": generate_prompt(
            messages=build_agentic_messages(
                user_request=data["question"],
                system_prompt=AGENTIC_RAG_SYSTEM_PLAIN,
                search_query=data["query"],
            ),
            tokenizer=tokenizer,
        )
        for key, data in TEST_QUESTIONS.items()
    }


def get_all_prompts(tokenizer: PreTrainedTokenizer) -> dict[str, str]:
    """Generate all prompts using native chat template."""
    return {
        **get_simple_prompts(tokenizer),
        **get_markdown_prompts(tokenizer),
        **get_plain_prompts(tokenizer),
    }
