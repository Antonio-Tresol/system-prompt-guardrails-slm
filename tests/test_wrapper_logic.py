"""Unit tests for GemmaWithSAE logic.

These tests mock the underlying HuggingFace model and tokenizer to verify
message formatting, tool binding, and SAE capture logic without requiring
heavy model weights or GPUs.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, ANY
from dotenv import load_dotenv

# Ensure project root is in path
load_dotenv()

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from model_evaluation.main_agent.gemma_wrapper import GemmaWithSAE
from model_evaluation.main_agent.gemma_scope_sae import SAEConfig

class TestGemmaWrapper(unittest.TestCase):
    def setUp(self):
        # Mocks
        self.mock_model = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_sae = MagicMock()
        self.mock_sae_config = MagicMock(spec=SAEConfig)
        
        # Setup tokenizer mock behavior
        self.mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        
        # Initialize wrapper
        self.wrapper = GemmaWithSAE(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            sae=self.mock_sae,
            sae_config=self.mock_sae_config,
            max_tokens=64
        )

    def test_message_formatting_logic(self):
        """Test that LangChain messages are correctly mapped to HF dicts."""
        
        # 1. Prepare input messages including tool usage
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="user input"),
            AIMessage(content="", tool_calls=[{"name": "tool1", "args": {}, "id": "call_1"}]),
            ToolMessage(content="result", tool_call_id="call_1", name="tool1"),
        ]
        
        # 2. Trigger generation (which calls extract_sae_features)
        # We need to mock extract_sae_features since it is imported in the module
        # Easier to check what was passed to tokenizer.apply_chat_template
        
        # We also need to mock the extract_sae_features logic inside the wrapper
        # or mock the module function. 
        # Since _generate calls self.tokenizer.apply_chat_template BEFORE calling extract_sae_features,
        # we can verify the formatting logic by checking tokenizer usage logic.
        
        # Mocking the extract_sae_features function patch
        with unittest.mock.patch("model_evaluation.main_agent.gemma_wrapper.extract_sae_features") as mock_extract:
            mock_extract.return_value = MagicMock(answer="response")
            
            self.wrapper._generate(messages)
            
            # 3. Verify tokenizer call arguments
            call_args = self.mock_tokenizer.apply_chat_template.call_args
            conversation = call_args[0][0] # First arg is conversation list
            kwargs = call_args[1] # Keyword args
            
            # Check Tools passed to tokenizer
            if self.wrapper._bound_tools:
                 self.assertIn("tools", kwargs)
                 self.assertEqual(kwargs["tools"], self.wrapper._bound_tools)

            # Check System
            self.assertEqual(conversation[0]["role"], "system")
            self.assertEqual(conversation[0]["content"], "sys")
            
            # Check User
            self.assertEqual(conversation[1]["role"], "user")
            self.assertEqual(conversation[1]["content"], "user input")
            
            # Check AI with Tool Call
            self.assertEqual(conversation[2]["role"], "assistant") # Should be assistant now
            self.assertEqual(conversation[2]["tool_calls"][0]["id"], "call_1")
            
            # Check Tool Result
            self.assertEqual(conversation[3]["role"], "tool")
            self.assertEqual(conversation[3]["tool_call_id"], "call_1")
            # name might be optional depending on implementation but we added it
            self.assertEqual(conversation[3]["name"], "tool1")

    def test_bind_tools(self):
        """Test tool binding storage and conversion."""
        # Create a real tool-like object or mock that convert_to_openai_tool can handle
        # For simplicity, we can mock convert_to_openai_tool usage in the wrapper or just ensure
        # that we pass something that doesn't crash. 
        # But wait, we modified the code to import convert_to_openai_tool.
        # We should test that it ACTUALLY converts.
        
        from langchain_core.tools import Tool
        
        def dummy_func(x): return x
        tool = Tool(name="dummy", func=dummy_func, description="desc")
        
        self.wrapper.bind_tools([tool])
        
        # Check internal storage 
        self.assertEqual(len(self.wrapper._bound_tools), 1)
        self.assertEqual(self.wrapper._bound_tools[0]["type"], "function")
        self.assertEqual(self.wrapper._bound_tools[0]["function"]["name"], "dummy")

if __name__ == "__main__":
    unittest.main()
