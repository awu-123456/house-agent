from typing import TypedDict

from langgraph.graph import MessagesState


class State(MessagesState):
    user_preferences: dict # 用户偏好数据
    user_intent: str       # 用户意图

class NeedReserveOutput(TypedDict):
    reserve: str