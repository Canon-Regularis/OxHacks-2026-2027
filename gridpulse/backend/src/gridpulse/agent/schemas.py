"""Versioned agent boundary. Model output is data, never executable UI code."""
from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Dates(Strict):
    from_date: date
    to_date: date

    @model_validator(mode="after")
    def ordered(self):
        if self.to_date < self.from_date or (self.to_date - self.from_date).days > 3659:
            raise ValueError("Invalid or excessive date range")
        return self


class Read(Dates):
    tool: Literal["coverage", "series", "runs", "analysis", "tables", "period", "execute"]
    ids: list[str] = Field(default_factory=list, max_length=9)
    algorithm: str = Field(default="", max_length=100)
    run_id: int | None = Field(default=None, ge=1)
    table: str = Field(default="", max_length=100)
    period: int = Field(default=1, ge=1, le=50)
    params: dict = Field(default_factory=dict)


class ViewWidget(Strict):
    key: str = Field(min_length=1, max_length=100)
    existing: str | None = Field(default=None, max_length=100)
    type: Literal["chart", "table", "comparison", "inspector", "context"]
    title: str = Field(min_length=1, max_length=100)
    series: list[str] = Field(default_factory=list, max_length=9)
    algorithm: str | None = Field(default=None, max_length=100)
    run_id: int | None = Field(default=None, ge=1)
    outputs: list[str] = Field(default_factory=list, max_length=30)
    table: str = Field(default="", max_length=100)
    group: str | None = Field(default="A", pattern=r"^[A-Z]$")
    relative_to: str | None = Field(default=None, max_length=100)
    position: Literal["left", "right", "top", "bottom", "tab"] = "right"
    size_step: int | None = Field(default=None, ge=1, le=23, description="Optional parent divider step: horizontal 1..23 out of 24, vertical 1..15 out of 16; measures the first child share.")
    metric: str = Field(default="", max_length=100)
    category: str = Field(default="", max_length=100)
    notes: str = Field(default="", max_length=2000)


class View(Dates):
    layout_mode: Literal["research", "preserve"] = "research"
    version: Literal[1] = 1
    widgets: list[ViewWidget] = Field(max_length=12)
    remove: list[str] = Field(default_factory=list, max_length=12)
    background_chat: bool = False
    new_workspace: bool = False
    name: str = Field(default="AI investigation", max_length=80)


class Decision(Strict):
    answer: str = Field(default="", max_length=10000)
    details: str = Field(default="", max_length=10000)
    reads: list[Read] = Field(default_factory=list, max_length=4)
    view: View | None = None
    evidence: list[str] = Field(default_factory=list, max_length=20)


class Message(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=12000)


class Turn(Strict):
    version: Literal[1] = 1
    turn_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    conversation: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    tab_id: str = Field(default="browser", pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    workspace_id: str = Field(max_length=100)
    revision: str = Field(max_length=100000)
    message: str = Field(min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list, max_length=16)
    summary: str = Field(default="", max_length=6000)
    snapshot: dict
    recent_states: list[dict] = Field(default_factory=list, max_length=3)
    approval: str | None = Field(default=None, max_length=100)
