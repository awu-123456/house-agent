
from pydantic import BaseModel, Field


class ReservedInfo(BaseModel):
    """预定信息"""
    order_id: str = Field(default="预定id")
    title: str = Field(default="预定的房源标题")
    phone_number: str = Field(default="预定电话")

    price: float | None = Field(
        default=None,
        description="预定的房源价格，单位为元/月"
    )
    intro: str | None = Field(
        default=None,
        description="预定的房源介绍"
    )
    city_name: str | None = Field(
        default=None,
        description="预定的房源所在城市名"
    )
    region_name: str | None = Field(
        default=None,
        description="预定的房源所在区/县"
    )

class UserPreferences(BaseModel):
    """用户偏好数据"""
    budget_min: float | None = Field(
        default=None,
        description="用户最低预算，单位为元/月"
    )
    budget_max: float | None = Field(
        default=None,
        description="用户最高预算，单位为元/月"
    )
    reserved_info: list[ReservedInfo] | None = Field(
        default=None,
        description="预定过的房源列表"
    )