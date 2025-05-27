import requests
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models import LLM, BaseLanguageModel, LangSmithParams
from langchain_core.outputs import Generation, LLMResult
from pydantic import Field, ConfigDict, model_validator
from typing_extensions import Self
from langchain_core.messages.tool import ToolCall, tool_call




# This code is part of the LangFlow project, which provides a framework for building and deploying language model applications.

from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.inputs import MessageTextInput, SecretStrInput
from langflow.io import DictInput, DropdownInput


class FuelIXComponent(LCModelComponent):
    display_name: str = "FuelIX"
    description: str = "Generate text using FuelIX LLMs."
    icon = "FuelIX"
    name = "FuelIXModel"

    inputs = [
        *LCModelComponent._base_inputs,
        DropdownInput(
            name="model_id",
            display_name="Model ID",
            options=[
                "ca-claude-3-haiku",
                "ca-claude-3-haiku-20240307",
                "ca-claude-3-sonnet",
                "ca-claude-3-sonnet-20240229",
                "claude-3-5-sonnet",
                "claude-3-5-sonnet-20240620",
                "claude-3-7-sonnet",
                "claude-3-haiku",
                "claude-3-haiku-20240307",
                "claude-3-sonnet",
                "claude-3-sonnet-20240229",
                "claude-sonnet-4",
                "dall-e-3",
                "gemini-1.5-flash",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro",
                "gemini-1.5-pro-001",
                "gemini-2.0-flash",
                "gemini-2.0-flash-exp",
                "gpt-4-0125-preview",
                "gpt-4.1-2025-04-14",
                "gpt-4o",
                "gpt-4o-2024-05-13",
                "gpt-4o-2024-08-06",
                "gpt-4o-mini",
                "gpt-4o-mini-2024-07-18",
                "imagen-3",
                "llama-3.2-90b",
                "llama-4-maverick-17b-128e-instruct",
            ],
            value="ca-claude-3-haiku",
            info="List of available FuelIX model IDs.",
        ),
        SecretStrInput(
            name="api_key",
            display_name="FuelIX API Key",
            info="Your FuelIX API key (Bearer token).",
            value="FUELIX_API_KEY",
            required=True,
        ),
        DictInput(
            name="model_kwargs",
            display_name="Model Kwargs",
            advanced=True,
            is_list=True,
            info="Additional keyword arguments to pass to the model.",
        ),
        MessageTextInput(
            name="endpoint_url",
            display_name="Endpoint URL",
            advanced=True,
            info="Override the FuelIX API endpoint URL (optional).",
        ),
        MessageTextInput(
            name="temperature",
            display_name="Temperature",
            advanced=True,
            info="Sampling temperature to use (optional).",
        ),
        MessageTextInput(
            name="max_tokens",
            display_name="Max Tokens",
            advanced=True,
            info="Maximum number of tokens to generate (optional).",
        ),
        DropdownInput(
            name="streaming",
            display_name="Streaming",
            options=["False", "True"],  # <-- Use strings
            value="False",              # <-- Default value as string
            info="Enable streaming responses.",
        ),
    ]

    def build_model(self) -> LanguageModel:  # type: ignore[type-var]


        endpoint_url = self.endpoint_url or "https://api-beta.fuelix.ai/chat/completions"
        model_kwargs = self.model_kwargs or {}

        # Optional fields
        temperature = None
        if self.temperature:
            try:
                temperature = float(self.temperature)
            except Exception:
                pass

        max_tokens = None
        if self.max_tokens:
            try:
                max_tokens = int(self.max_tokens)
            except Exception:
                pass

        output = FuelIXLLM(
            api_key=self.api_key,
            model_id=self.model_id,
            model_kwargs=model_kwargs,
            streaming=self.streaming,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Optionally override endpoint if supported in your FuelIXLLM
        if hasattr(output, "endpoint_url"):
            output.endpoint_url = endpoint_url
        return output


FUELIX_API_URL = "https://api-beta.fuelix.ai/chat/completions"

def _tool_to_dict(tool):
    # Try to convert tool to dict if it's a Pydantic model or has a .dict()/.to_dict() method
    if hasattr(tool, "dict"):
        return tool.dict()
    if hasattr(tool, "to_dict"):
        return tool.to_dict()
    if hasattr(tool, "schema"):
        return tool.schema()
    if isinstance(tool, dict):
        return tool
    # Fallback: try __dict__ (may not always work)
    if hasattr(tool, "__dict__"):
        return dict(tool.__dict__)
    raise TypeError(f"Tool of type {type(tool)} is not JSON serializable")

class FuelIXBase(BaseLanguageModel):
    """Base class for FuelIX models."""

    api_key: str = Field(..., description="FuelIX API key (Bearer token)")
    model_id: str = Field(..., alias="model")
    model_kwargs: Optional[Dict[str, Any]] = None
    streaming: bool = False

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        _model_kwargs = self.model_kwargs or {}
        return {
            "model_id": self.model_id,
            "stream": self.streaming,
            **_model_kwargs,
        }

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if not self.api_key:
            raise ValueError("FuelIX API key must be provided.")
        return self

    def _prepare_input(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Ensure every message has an 'id'
        for msg in messages:
            if "id" not in msg:
                msg["id"] = str(uuid.uuid4())
        body = {
            "model": self.model_id,
            "messages": messages,
        }
        if self.streaming:
            body["stream"] = True
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.max_tokens is not None:
            body["max_tokens"] = self.max_tokens
        if self.model_kwargs:
            body.update(self.model_kwargs)
        # Convert tools to dicts for JSON serialization
        if self._tools:
            body["tools"] = [_tool_to_dict(tool) for tool in self._tools]
        if self._tool_choice:
            body["tool_choice"] = self._tool_choice
        body.update(kwargs)
        return body

    def _parse_response(self, response: requests.Response) -> Tuple[str, Dict[str, Any]]:
        data = response.json()
        # If FuelIX supports tool-calling, handle tool_calls in the response
        if "tool_calls" in data:
            # Ensure each tool_call has an 'id'
            for call in data["tool_calls"]:
                if "id" not in call:
                    call["id"] = call.get("name", "tool_call")
            usage = data.get("usage", {})
            return "", {"tool_calls": data["tool_calls"], "usage": usage}
        # FuelIX returns choices[0].message.content
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return text, usage

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = self._prepare_input(messages, stop, **kwargs)
        response = requests.post(FUELIX_API_URL, headers=headers, json=body)
        if not response.ok:
            logging.error(f"FuelIX API error: {response.status_code} {response.text}")
            raise ValueError(f"FuelIX API error: {response.status_code} {response.text}")
        return self._parse_response(response)

    def get_num_tokens(self, text: str) -> int:
        # Not implemented for FuelIX, return length as a fallback
        return len(text.split())

    def get_token_ids(self, text: str) -> List[int]:
        # Not implemented for FuelIX
        return []

class FuelIXLLM(LLM, FuelIXBase):
    """LangChain LLM for FuelIX chat completions."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    _tools: Optional[List[Any]] = None
    _tool_choice: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "fuelix"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "llms", "fuelix"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        return {}

    def bind_tools(
        self,
        tools: List[Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "FuelIXLLM":
        """
        Bind tool-like objects to this LLM, simulating tool calling if not natively supported.
        """
        self._tools = tools
        self._tool_choice = tool_choice
        return self

    def _simulate_tool_prompt(self, prompt: str) -> str:
        """Inject tool descriptions into the system prompt to simulate tool calling."""
        if not self._tools:
            return prompt
        tool_descriptions = []
        for tool in self._tools:
            # Try to extract name, description, and parameters
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None) or "tool"
            description = getattr(tool, "description", None) or getattr(tool, "__doc__", None) or ""
            parameters = getattr(tool, "args_schema", None)
            if parameters:
                try:
                    params = parameters.schema() if hasattr(parameters, "schema") else str(parameters)
                except Exception:
                    params = str(parameters)
            else:
                params = ""
            tool_descriptions.append(
                f"Tool name: {name}\nDescription: {description}\nParameters: {params}"
            )
        tools_block = "\n\n".join(tool_descriptions)
        system_prompt = (
            "You can call tools by responding with a JSON object in the following format:\n"
            '{"tool_call": {"name": "<tool_name>", "args": {...}}}\n'
            "Available tools:\n"
            f"{tools_block}\n\n"
            "If you want to call a tool, respond ONLY with the JSON object."
        )
        # Prepend system prompt to user prompt
        return f"{system_prompt}\n\nUser: {prompt}"

    def _parse_tool_response(self, text: str) -> Tuple[Optional[ToolCall], str]:
        """
        Try to parse a tool call from the LLM response.
        Returns (tool_call, remaining_text).
        """
        import json
        import re

        # Look for a JSON object in the response
        match = re.search(r'(\{.*"tool_call".*\})', text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
                tc = obj.get("tool_call")
                if tc and "name" in tc and "args" in tc:
                    return tool_call(name=tc["name"], args=tc["args"], id=None), ""
            except Exception:
                pass
        return None, text

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Simulate tool calling if tools are bound
        if self._tools:
            prompt = self._simulate_tool_prompt(prompt)
        messages = [{"role": "user", "content": prompt}]
        text, usage = self._call_api(messages, stop, **kwargs)
        tool_call_obj, remaining = self._parse_tool_response(text) if self._tools else (None, text)
        if run_manager is not None:
            run_manager.on_llm_end(
                LLMResult(generations=[[Generation(text=text)]], llm_output={"usage": usage})
            )
        # If a tool call was detected, return it as a string (or handle as needed)
        if tool_call_obj:
            return f"[TOOL_CALL]{tool_call_obj}"
        return remaining

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self._tools:
            prompt = self._simulate_tool_prompt(prompt)
        import asyncio
        loop = asyncio.get_running_loop()
        messages = [{"role": "user", "content": prompt}]
        text, usage = await loop.run_in_executor(
            None, lambda: self._call_api(messages, stop, **kwargs)
        )
        tool_call_obj, remaining = self._parse_tool_response(text) if self._tools else (None, text)
        if run_manager is not None:
            await run_manager.on_llm_end(
                LLMResult(generations=[[Generation(text=text)]], llm_output={"usage": usage})
            )
        if tool_call_obj:
            return f"[TOOL_CALL]{tool_call_obj}"
        return remaining
