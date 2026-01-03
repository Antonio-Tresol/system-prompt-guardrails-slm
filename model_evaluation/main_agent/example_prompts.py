"""Example prompts for testing attention analysis, including agentic RAG contexts."""

# Simple prompts for quick testing
SIMPLE_PROMPTS = {
    "simple_fact": "# Question: The capital of France is? please provide the answer ",
    "reasoning": "Step by step, the solution to 2+2 is",
    "context_retrieval": "Mary is a doctor. John is a nurse. Who is the doctor?",
}

# A realistic RAG agent context with retrieved documents and system instructions
RAG_AGENT_SYSTEM = """You are an AI assistant that answers questions based on retrieved documents.
Follow these rules:
1. Only answer based on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Cite the source document when possible
4. Be concise but accurate"""

RAG_RETRIEVED_DOCS = """## Retrieved Documents

### Document 1: Company Overview (source: about_us.pdf)
TechFlow Inc. was founded in 2018 by Sarah Chen and Marcus Rodriguez in San Francisco, California. 
The company specializes in building AI-powered workflow automation tools for enterprise customers.
As of 2024, TechFlow has over 500 employees across offices in San Francisco, New York, London, and Singapore.
The company raised $150 million in Series C funding in March 2023, led by Sequoia Capital.
Current CEO is Sarah Chen, with Marcus Rodriguez serving as CTO.

### Document 2: Product Information (source: products.pdf)
TechFlow's flagship product is "FlowAI", an intelligent automation platform that uses machine learning
to optimize business processes. Key features include:
- Natural language workflow creation
- Automated document processing
- Integration with 200+ enterprise applications
- Real-time analytics dashboard
- Custom AI model training
Pricing starts at $500/month for teams up to 10 users, with enterprise plans available for larger organizations.

### Document 3: Recent News (source: press_release_2024.pdf)
In January 2024, TechFlow announced a strategic partnership with Microsoft Azure to provide
native cloud integration for FlowAI customers. This partnership enables seamless deployment
of TechFlow solutions within Azure environments and provides access to Azure's AI services.
The partnership is expected to expand TechFlow's enterprise customer base by 40% over the next year.

### Document 4: Technical Specifications (source: technical_docs.pdf)
FlowAI is built on a microservices architecture using:
- Backend: Python (FastAPI), Go for high-performance components
- Frontend: React with TypeScript
- Database: PostgreSQL for transactional data, MongoDB for document storage
- ML Infrastructure: PyTorch, deployed on NVIDIA GPUs
- Message Queue: Apache Kafka for event streaming
System requirements: 8GB RAM minimum, 16GB recommended for production workloads.
API rate limits: 1000 requests/minute for standard plans, unlimited for enterprise.

### Document 5: Customer Testimonials (source: case_studies.pdf)
"FlowAI reduced our document processing time by 75% and saved us approximately $2M annually."
- James Wilson, VP Operations at GlobalBank

"The natural language workflow creation is game-changing. Non-technical team members can now
build complex automations without any coding knowledge."
- Maria Garcia, Director of Digital Transformation at RetailMax

### Document 6: Support and SLA (source: support_policy.pdf)
TechFlow provides 24/7 customer support for enterprise customers. Standard SLA guarantees:
- 99.9% uptime for the FlowAI platform
- 4-hour response time for critical issues
- 24-hour response time for standard support tickets
- Dedicated success manager for accounts over $50k ARR
Support channels: Email, phone, Slack integration, and in-app chat."""


def build_rag_prompt(question: str) -> str:
    """Build a full RAG agent prompt with system, context, and user question."""
    return f"""<system>
{RAG_AGENT_SYSTEM}
</system>

<context>
{RAG_RETRIEVED_DOCS}
</context>

<user>
{question}
</user>

<assistant>
"""


# Pre-built RAG prompts for testing
RAG_PROMPTS = {
    "rag_founder": build_rag_prompt("Who founded TechFlow and when?"),
    "rag_pricing": build_rag_prompt("How much does FlowAI cost per month?"),
    "rag_tech_stack": build_rag_prompt("What programming languages does FlowAI use?"),
    "rag_partnership": build_rag_prompt("Tell me about TechFlow's partnership with Microsoft."),
    "rag_testimonial": build_rag_prompt("What did GlobalBank say about FlowAI?"),
}

# Combined prompts dictionary for easy access
ALL_PROMPTS = {**SIMPLE_PROMPTS, **RAG_PROMPTS}
