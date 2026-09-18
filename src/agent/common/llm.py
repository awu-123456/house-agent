from langchain.chat_models import init_chat_model

model = init_chat_model(
    "deepseek-v4-pro",
    temperature=0,
    model_kwargs={
        "extra_body": {
            "thinking": {
                "type": "disabled"
            }
        }
    }
)