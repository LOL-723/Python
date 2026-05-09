from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


class ChatResponse(BaseModel):
    answer: str


class SummaryOutput(BaseModel):
    title: str = Field(..., description="Summary title")
    summary: str = Field(..., description="Summary text")
    key_points: list[str] = Field(default_factory=list, description="Key points")
    action_items: list[str] = Field(default_factory=list, description="Action items")


class PersonInfoOutput(BaseModel):
    name: str = Field(..., description="Person name")
    age: int = Field(..., description="Person age")
    skills: list[str] = Field(default_factory=list, description="Skills")
