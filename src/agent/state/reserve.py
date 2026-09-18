from langgraph.graph import MessagesState


class ReserveState(MessagesState):
    title: str
    phone_number: str
    id_card: str