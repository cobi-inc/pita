from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional
from enum import Enum

# --- Token Usage Models ---
class Usage(BaseModel):
    """
    Token usage statistics for API requests and responses.

    Attributes:
        prompt_tokens: Number of tokens in the prompt/input.
        completion_tokens: Number of tokens in the completion/output.
        total_tokens: Total number of tokens (prompt + completion).
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# OpenAI Chat Completion pydantic Models (From pydantic-openai repo)
class ChatMessageRole(str, Enum):
    """
    Enumeration of possible roles for chat messages.

    Values:
        System: System message role for instructions.
        User: User message role for user inputs.
        Assistant: Assistant message role for AI responses.
    """
    System = "system"
    User = "user"
    Assistant = "assistant"

class ChatCompletionMessage(BaseModel):
    """
    A single message in a chat conversation.

    Attributes:
        role: The role of the message sender (system, user, or assistant).
        content: The text content of the message.
        name: Optional name of the message author.
    """
    role: ChatMessageRole
    content: str
    name: Optional[str] = Field(None, alias="name")

class ChatCompletionRequest(BaseModel):
    """
    Request model for chat completion API endpoint.

    This model defines all parameters that can be sent when requesting a chat completion,
    following the OpenAI-compatible API format.

    Attributes:
        model: Name of the model to use for completion.
        messages: List of messages in chat format (conversation history).
        max_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. Higher values make output more random.
        top_p: Nucleus sampling parameter.
        n: Number of completions to generate for each prompt.
        stream: Whether to stream responses incrementally.
        stop: List of sequences where the API will stop generating.
        presence_penalty: Penalty for tokens based on their presence in the text so far.
        frequency_penalty: Penalty for tokens based on their frequency in the text so far.
        logit_bias: Map of token IDs to bias values.
        user: Optional unique identifier for the end-user.
    """
    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: Optional[int] = Field(None, alias="max_tokens")
    temperature: Optional[float] = Field(None, alias="temperature")
    top_p: Optional[float] = Field(None, alias="top_p")
    n: Optional[int] = Field(None, alias="n")
    stream: Optional[bool] = Field(None, alias="stream")
    stop: Optional[List[str]] = Field(None, alias="stop")
    presence_penalty: Optional[float] = Field(None, alias="presence_penalty")
    frequency_penalty: Optional[float] = Field(None, alias="frequency_penalty")
    logit_bias: Optional[Dict[str, int]] = Field(None, alias="logit_bias")
    user: Optional[str] = Field(None, alias="user")

class ChatCompletionChoice(BaseModel):
    """
    A single completion choice from the model.

    Attributes:
        index: The index of this choice in the list of choices.
        message: The generated message for this choice.
        finish_reason: The reason why the model stopped generating.
    """
    index: int
    message: ChatCompletionMessage
    finish_reason: str = Field(..., alias="finish_reason")

class ChatCompletionResponse(BaseModel):
    """
    Response model for chat completion API endpoint.

    This model defines the structure of the response returned by the chat completion endpoint,
    following the OpenAI-compatible API format.

    Attributes:
        id: Unique identifier for this completion.
        object: Object type (always "chat.completion" for non-streaming).
        created: Unix timestamp of when the completion was created.
        model: The model used for this completion.
        choices: List of completion choices (typically one unless n > 1 in request).
        usage: Token usage statistics for this request.
    """
    id: str = Field(..., alias="id")
    object: str = Field(..., alias="object")
    created: int = Field(..., alias="created")
    model: str = Field(..., alias="model")
    choices: List[ChatCompletionChoice]
    usage: Usage
