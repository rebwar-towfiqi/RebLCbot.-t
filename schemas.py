# admin/schemas.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    telegram_id: str
    name: Optional[str] = None
    lang: Optional[str] = "fa"
    active: Optional[bool] = True
    expire_at: Optional[datetime] = None

class UserCreate(UserBase):
    pass

class UserOut(UserBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class PaymentOut(BaseModel):
    id: int
    user_id: str
    method: str
    tx_id: Optional[str]
    amount: str
    verified: bool
    submitted_at: datetime

    class Config:
        orm_mode = True

class ReceiptOut(BaseModel):
    id: int
    user_id: str
    image_path: Optional[str]
    text_note: Optional[str]
    status: str
    reviewed_by: Optional[str]
    reviewed_at: Optional[datetime]

    class Config:
        orm_mode = True
