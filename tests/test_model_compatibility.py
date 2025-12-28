import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from pathlib import Path
from knowledge_base.config.settings import Settings

# Setup
PROJECT_ROOT = Path.cwd()
CONFIG_PATH = PROJECT_ROOT / "knowledge_base" / "config" / "config.yaml"
settings = Settings.load_from_yaml(config_path=str(CONFIG_PATH), project_root=PROJECT_ROOT)


class TestResult(BaseModel):
    success: bool
    message: str


@tool
def dummy_tool(query: str) -> str:
    """A dummy tool for testing."""
    return f"Processed: {query}"


def test_model():
    print("Testing moonshotai/kimi-k2-thinking on OpenRouter...")
    model = ChatOpenAI(
        model="moonshotai/kimi-k2-thinking",
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
    )

    agent = create_agent(
        model=model,
        tools=[dummy_tool],
        response_format=ToolStrategy(TestResult),
        system_prompt="You are a test agent. Verify the user input and return a TestResult.",
    )

    try:
        response = agent.invoke({"messages": [{"role": "user", "content": "Test this agent."}]})
        print("\n--- Response ---")
        print(response)
        if "structured_response" in response:
            print(f"\n✅ Success! Structured response: {response['structured_response']}")
        else:
            print("\n❌ Failed to get structured response.")
    except Exception as e:
        print(f"\n❌ Error during invocation: {e}")


if __name__ == "__main__":
    test_model()
