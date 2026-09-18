from langchain_core.messages import SystemMessage, filter_messages, HumanMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import Field, BaseModel
from typing_extensions import Literal

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.state.main import State, NeedReserveOutput


def get_store_info(state: State, runtime: Runtime[ContextSchema], *, store: BaseStore):
    user_id = runtime.context.get("user_id")
    namespace = (user_id, "preferences")
    prefs_result = store.search(namespace)
    if prefs_result and prefs_result[0]:
        return {
            "user_preferences": prefs_result[0].value
        }
    else:
        return {
            "user_preferences": {}
        }

class UserMessage(BaseModel):
    type: Literal["recommend_house", "reserve_house", "get_info", "others"] = Field(
        description="根据用户问题描述判断问题类型：推荐房源、预定房源、获取信息、其它内容"
    )

def identify_question(state: State):
    user_intent = model.with_structured_output(UserMessage).invoke(
        [SystemMessage(content="你是一个根据描述提取信息的提取专家。请从用户的描述中提取想要咨询的相关信息。"
                              "严谨根据语义推断信息，但是不能猜测或者编造信息。"), state["messages"][-1]]
    )
    return {
        "user_intent": user_intent.type
    }

def need_reserve(state: State) -> NeedReserveOutput:
    prompt = f"已经为您推荐合适的房源，是否需要帮您预订房源？\n"
    prompt += "如果不需要,请输入'**不需要**'。\n"
    prompt += "如果需要,请输入'**需要**'。\n(注意输入其它值无效)\n"
    answer = interrupt(prompt)
    return {
        "reserve": answer
    }

def get_user_preferences(state: State):
    prefs = state.get("user_preferences", {})
    user_messages = filter_messages(state["messages"], include_types="human")
    reserved_info = prefs.get("reserved_info", [])
    if reserved_info:
        reserved_str = "\n"
        for i, item in enumerate(reserved_info, 1):
            reserved_str += f"{i}. 预定工单ID: {item.order_id}，" \
                            f"房源标题：{item.title}，" \
                            f"预定电话：{item.phone_number}\n"
    else:
        reserved_str = "无"
    result = model.invoke(
        [SystemMessage(content="""你是一个乐于助人的助手，可以根据用户偏好信息进行回复。
如果有的偏好数据为空，不要猜测或编造数据。
不要直接回复偏好数据是什么，要结合问题进行生动回复。
如果问题与用户偏好数据无关，直接回复即可。"""),
         HumanMessage(content="用户的历史偏好信息如下"
                      f"1. 最低预算：{prefs.get('budget_min')}"
                      f"2. 最高预算：{prefs.get('budget_max')}"
                      f"3. 已预定过的信息：{reserved_str}"),
         user_messages[-1]
         ]
    )
    return {
        "messages": [result]
    }