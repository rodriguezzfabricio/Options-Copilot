from pydantic import BaseModel, Field
from datetime import date
from typing import Literal

class OptionsContract(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol")
    strike: float = Field(..., gt=0, description="Strike price")
    expiry: date = Field(..., description="Expiration date")
    option_type: Literal["call", "put"] = Field(..., description="Option type")
    contract_size: int = Field(default=100, description="Shares per contract")
    
    class Config:
        # This allows you to use the model like a constructor
        validate_assignment = True
        
    # Optional: Add custom methods
    def is_itm(self, current_price: float) -> bool:
        """Check if option is in-the-money"""
        if self.option_type == "call":
            return current_price > self.strike
        return current_price < self.strike