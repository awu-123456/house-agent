import os
import uuid

from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    filter_messages,
)
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agent.common.context import ContextSchema
from src.agent.common.llm import model
from src.agent.common.store import UserPreferences
from src.agent.state.recommend import RecommendState, get_recommend_info


# 定义用户信息的数据模型(结构化输出)
class UserInfo(BaseModel):
    """用户的租房需求信息"""
    city: str | None = Field(
        default=None,
        description="用户所在或想要租房的城市，例如：西安、北京、上海"
    )
    district: str | None = Field(
        default=None,
        description="用户想要租房的具体区域或⾏政区，例如：雁塔区、碑林区、海淀区"
    )
    budget_min: float | None = Field(
        default=None,
        description="用户的最低预算，单位为元/⽉，如果是xx元以内，要设置最小值为0"
    )
    budget_max: float | None = Field(
        default=None,
        description="用户的最⾼预算，单位为元/⽉，如果是xx元以上，最大值设置为50000"
    )
    room_type: str | None = Field(
        default=None,
        description="房屋类型，例如：整租、合租、公寓、⼀室⼀厅、两室⼀厅"
    )
    orientation: str | None = Field(
        default=None,
        description="房屋朝向，例如：朝南、朝北、东南、南北通透"
    )
    room_count: int | None = Field(
        default=None,
        description="需要推荐的房屋数量"
    )
    others: str | None = Field(
        default=None,
        description="特殊要求，例如：带阳台、独⽴卫⽣间、近地铁、可养宠物、有电梯等"
    )

def collect_user_info(state: RecommendState, runtime: Runtime[ContextSchema], *, store: BaseStore):
    """收集用户希望的推荐信息"""
    # 1. 获取需要被解析的数据：最新的用户消息 + 用户偏好数据
    user_messages = filter_messages(state["messages"], include_types="human")
    pref = state.get("user_preferences")
    if pref and (pref["budget_min"] or pref["budget_max"]):
        extract_messages = [
            HumanMessage(content="用户的历史偏好信息如下："
                         f"1. 最低预算：{pref['budget_min']}"
                         f"2. 最高预算：{pref['budget_max']}"),
            user_messages[-1]
        ]
    else:
        extract_messages = [user_messages[-1]]

    # 2. 提取信息（LLM结构化返回）
    def extract_info(messages) -> UserInfo:
        system_message = SystemMessage(
            content="""
        你是一个租房需求信息提取专家。请从用户的描述与历史信息中提取租房相关信息。
        如果用户历史偏好信息与最新用户消息冲突，以最新的用户消息为主。
        只提取用户明确提到的信息，不要猜测或推断。
        如果某个信息用户没有提到，就返回null。
        注意预算的单位可能是元/月、元/天等，请统一转换为元/月。
        如果用户提到价格范围，请分别提取最低和最高预算。
        如果用户提到推荐几套房，提取room_count字段。"""
        )
        return model.with_structured_output(schema=UserInfo).invoke([system_message] + messages)

    # 更新状态函数
    def update_state(current_state: dict, info: UserInfo) -> dict:
        if not info:
            return current_state

        user_info_dict = info.model_dump(exclude_none=True)
        current_state.update(user_info_dict)
        return current_state

    # 根据用户偏好和用户消息提取信息
    updated_state = {}
    extracted_info = extract_info(extract_messages)
    updated_state = update_state(updated_state, extracted_info)

    # 3. 中断咨询推荐的必须参数
    # 检查是否缺失关键信息：城市、预算范围
    missing_info = []
    if not updated_state.get("city"):
        missing_info.append("**城市**")
    if updated_state.get("budget_min") is None or updated_state.get("budget_max") is None:
        missing_info.append("**预算范围**")

    if missing_info:
        prompt = f"为了给您推荐合适的房源，请提供以下信息:{'，'.join(missing_info)}和其它信息。\n"
        prompt += "如果您不想提供，请输入'**不提供**',我会根据已有信息为您推荐房源。"
        # 根据缺失信息进行中断
        answer = interrupt(prompt)
        if str(answer).strip() == "不提供":
            # 已经缺失关联信息，而且用户不提供，需要给关键信息设置默认值
            if not updated_state.get("city"):
                updated_state["city"] = "随机城市"
            if not updated_state.get("budget_min"):
                updated_state["budget_min"] = 500.0
            if not updated_state.get("budget_max"):
                updated_state["budget_max"] = 5000.0
            if not updated_state.get("room_count"):
                updated_state["room_count"] = 5
        else:
            # 确实关键信息，但是用户已经补充
            # 将answer构建为HumanMessage
            user_response_msg = HumanMessage(content=str(answer))
            extracted_info = extract_info([user_response_msg])
            # updated_state就是包含了中断结果
            updated_state = update_state(updated_state, extracted_info)

    # 4. 持久化处理：更新预算
    if updated_state.get("budget_min") or updated_state.get("budget_max"):
        user_id = runtime.context.get("user_id")
        namespace = (user_id, "preferences")
        prefs_result = store.search(namespace)

        if len(prefs_result) == 0:
            prefs = UserPreferences(
                budget_min=updated_state.get("budget_min"),
                budget_max=updated_state.get("budget_max"),
            )
            store.put(
                namespace,
                str(uuid.uuid4()),
                prefs.model_dump(exclude_none=True)
            )
            updated_state["user_preferences"] = prefs.model_dump(exclude_none=True)
        else:
            prefs = prefs_result[0].value
            store_min = prefs["budget_min"]
            store_max = prefs["budget_max"]
            cur_min = updated_state.get("budget_min")
            cur_max = updated_state.get("budget_max")
            update_min = False
            update_max = False
            if store_min is not None and cur_min is not None and cur_min < store_min:
                update_min = True
            elif store_min is None and cur_min is not None:
                update_min = True

            if store_max is not None and cur_max is not None and cur_max > store_max:
                update_max = True
            elif store_max is None and cur_max is not None:
                update_max = True

            if update_min or update_max:
                if update_min:
                    prefs["budget_min"] = cur_min
                if update_max:
                    prefs["budget_max"] = cur_max
                store.put(
                    namespace,
                    prefs_result[0].key,
                    prefs
                )
                updated_state["user_preferences"] = prefs

    # 5. 准备最终的消息，并更新
    updated_state["messages"] = [HumanMessage(content=get_recommend_info(updated_state))]
    # 打印⽇志

    print(f"已收集⽤⼾信息:城市= {updated_state.get('city')}, "
          f"区域= {updated_state.get('district')}, "
          f"预算= {updated_state.get('budget_min')}{updated_state.get('budget_max')}, "
          f"房间数= {updated_state.get('room_count')}"
          )

    return updated_state


load_dotenv()
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# 获取数据库⼯具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
# print(tools)

# 节点：获取表信息
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")
# 节点：执行sql查询
run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")

# 节点2：获取全量表
def list_tables(state: RecommendState):
    # 1. 获取AIMessages（tool_call）
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "123123",
        "type": "tool_call"
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    # 2. 手动调用工具：sql_db_list_tables
    list_tables_tools = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tools.invoke(tool_call)

    # 3. 整合结果
    response = AIMessage(content=f"可用的表：{tool_message.content}")
    return {
        "messages": [tool_call_message, tool_message, response]
    }

# 节点：绑定工具（获取表信息），让LLM将来必定执行工具节点
def call_get_schema(state: RecommendState):

    print("========== call_get_schema ==========")
    for i, msg in enumerate(state["messages"]):
        print(
            i,
            type(msg).__name__,
            "id=", getattr(msg, "id", None),
            "tool_call_id=", getattr(msg, "tool_call_id", None),
            "tool_calls=", getattr(msg, "tool_calls", None)
        )

    llm_with_tool = model.bind_tools([get_schema_tool], tool_choice="any")
    # llm 根据历史消息筛选表，并获取需要的表信息
    response = llm_with_tool.invoke(state["messages"])  # AIMessage(tool_calls)
    # 下一步一定会调用get_schema_tool工具的
    return {
        "messages" : [response]
    }

# 构造SQL:

# 节点：生成SQL + 整合结果
def generate_query(state: RecommendState):

    print("========== generate_query ==========")
    for i, msg in enumerate(state["messages"]):
        print(
            i,
            type(msg).__name__,
            "id=", getattr(msg, "id", None),
            "tool_call_id=", getattr(msg, "tool_call_id", None),
            "tool_calls=", getattr(msg, "tool_calls", None)
        )

    generate_query_system_prompt = """
    您是⼀个设计⽤于与SQL数据库交互的代理。
    给定⼀个输⼊问题，创建⼀个语法正确的{dialect}查询来运⾏，然后查看查询的结果并返回答案。
    需要根据rows from table的⽰例设置真实查询的值。
    除⾮⽤⼾指定了他们希望获得的特定数量的⽰例，否则始终将查询限制为最多{top_k}个结果。
    您可以按相关列对结果排序，以返回最感兴趣的结果。不要查询特定表中的所有列，只查询给定问题的相关列。
    不要对数据库做任何DML语句（INSERT，UPDATE，DELETE，DROP等)。"""
    system_prompt = generate_query_system_prompt.format(
        dialect = db.dialect,
        top_k = state.get("room_count", 5)
    )
    system_message = SystemMessage(content=system_prompt)
    llm_with_tools = model.bind_tools([run_query_tool])
    return {
        "messages": llm_with_tools.invoke([system_message] + state["messages"])
    }

def check_query(state: RecommendState):

    print("========== check_query ==========")
    for i, msg in enumerate(state["messages"]):
        print(
            i,
            type(msg).__name__,
            "id=", getattr(msg, "id", None),
            "tool_call_id=", getattr(msg, "tool_call_id", None),
            "tool_calls=", getattr(msg, "tool_calls", None)
        )

    check_query_system_prompt = f"""你是⼀个⾮常注重细节的SQL专家。
    仔细检查{db.dialect}查询中的常⻅错误，包括：
    -使⽤NULL值的NOT IN
    -在应该使⽤UNION ALL时使⽤UNION
    -使⽤BETWEEN表⽰独占范围
    -谓词中的数据类型不匹配
    -正确引⽤标识符
    -使⽤正确数量的函数参数
    -转换为正确的数据类型
    -使⽤合适的列进⾏连接
    如果存在上述任何错误，请重写查询。如果没有错误，只需复制原始查询即可。
    在运⾏此检查之后，您将调⽤适当的⼯具来执⾏查询。"""
    system_message = SystemMessage(content=check_query_system_prompt)
    # 将SQL当作用户消息传入进行检查
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = HumanMessage(content=tool_call["args"]["query"])
    llm_with_tools = model.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    # 目前最新的一个消息是AI（t），现在又生成一个AI（t），则可以将两个合成一个
    response.id = state["messages"][-1].id
    return {
        "messages": [response]
    }