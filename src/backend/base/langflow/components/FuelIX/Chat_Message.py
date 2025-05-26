from typing import Any, Dict, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langflow.base.models.model import LCModelComponent
from langflow.field_typing import LanguageModel
from langflow.field_typing.range_spec import RangeSpec
from langflow.inputs import (
    BoolInput,
    DictInput,
    DropdownInput,
    IntInput,
    SecretStrInput,
    SliderInput,
    StrInput,
    MultilineInput
)
from pydantic.v1 import SecretStr
import requests
import logging

class FuelixChatLLM(LLM):
    api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.5
    max_tokens: int = 4096
    system_message: str = "You are a helpful assistant."
    timeout: int = 30
    additional_params: Dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "fuelix"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        messages = [
            {
                'role': 'system',
                'content': self.system_message,
            },
            {
                'role': 'user',
                'content': prompt,
            },
        ]

        json_data = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            **self.additional_params,
        }

        try:
            response = requests.post(
                'https://api-beta.fuelix.ai/v1/chat/completions',
                headers=headers,
                json=json_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error(f"FuelIX API request failed: {str(e)}")
            raise ValueError(f"API request failed: {str(e)}")

class FuelixChatRequest(LCModelComponent):
    display_name = "FuelIX Chat Request [BETA]"
    description = "Component to make requests to FuelIX API with advanced options"
    documentation = "https://docs.fuelix.ai"
    icon = "chat"
    name = "FuelixChatRequest"

    AVAILABLE_MODELS = [
        "ca-claude-3-haiku",
        "ca-claude-3-sonnet",
        "gpt-4o-mini",
        "claude-3-5-sonnet",
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet",
        "claude-3-haiku",
        "claude-3-haiku-20240307",
        "claude-3-sonnet",
        "claude-3-sonnet-20240229",
        "dall-e-3",
        "gemini-1.5-flash",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro",
        "gemini-1.5-pro-001",
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gpt-4-0125-preview",
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "imagen-3",
        "llama-3.1-405b",
        "llama-3.1-70b",
        "llama-3.1-8b",
        "llama-3.2-90b",
    ]

    inputs = [
        *LCModelComponent._base_inputs,
        SecretStrInput(
            name="api_key",
            display_name="FuelIX API Key",
            info="The FuelIX API Key",
            advanced=False,
            required=True,
        ),
        DropdownInput(
            name="model",
            display_name="Model Name",
            options=AVAILABLE_MODELS,
            value="gpt-4o-mini",
            advanced=False,
        ),
        
        MultilineInput(
            name="system_message",
            display_name="System Message",
            value="You are a helpful assistant.",
            advanced=False,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.5,
            range_spec=RangeSpec(min=0, max=2, step=0.1),
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            value=4096,
            range_spec=RangeSpec(min=1, max=32768),
            advanced=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            value=30,
            advanced=True,
        ),
        DictInput(
            name="additional_params",
            display_name="Additional Parameters",
            advanced=True,
        ),
    ]

    def build_model(self) -> LanguageModel:
        api_key = SecretStr(self.api_key).get_secret_value() if self.api_key else None
        
        if not api_key:
            raise ValueError("API key is required")

        return FuelixChatLLM(
            api_key=api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_message=self.system_message,
            timeout=self.timeout,
            additional_params=self.additional_params or {},
        )

    def _get_exception_message(self, e: Exception) -> str:
        if isinstance(e, requests.exceptions.RequestException):
            return f"FuelIX API Error: {str(e)}"
        return str(e)