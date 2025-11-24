from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from enum import Enum

# --- Token Usage Models ---
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# OpenAI Chat Completion pydantic Models (From pydantic-openai repo)
class ChatMessageRole(str, Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"

class ChatCompletionMessage(BaseModel):
    role: ChatMessageRole
    content: str
    name: Optional[str] = Field(None, alias="name")

class ChatCompletionRequest(BaseModel):
    model: str # Model Name
    messages: List[ChatCompletionMessage] # List of messages in chat format
    max_tokens: Optional[int] = Field(None, alias="max_tokens") # Max tokens to generate
    temperature: Optional[float] = Field(None, alias="temperature") # Sampling temperature
    top_p: Optional[float] = Field(None, alias="top_p") # Nucleus sampling parameter
    n: Optional[int] = Field(None, alias="n") # Number of completions to generate
    stream: Optional[bool] = Field(None, alias="stream") # Whether to stream responses
    stop: Optional[List[str]] = Field(None, alias="stop") # Stop sequences
    presence_penalty: Optional[float] = Field(None, alias="presence_penalty") # Presence penalty
    frequency_penalty: Optional[float] = Field(None, alias="frequency_penalty") # Frequency penalty
    logit_bias: Optional[Dict[str, int]] = Field(None, alias="logit_bias") # Logit bias
    user: Optional[str] = Field(None, alias="user") # User identifier

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: str = Field(..., alias="finish_reason")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., alias="id")
    object: str = Field(..., alias="object")
    created: int = Field(..., alias="created")
    model: str = Field(..., alias="model")
    choices: List[ChatCompletionChoice]
    usage: Usage