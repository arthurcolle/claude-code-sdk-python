from typing import Callable, Dict, Union

def calculate(x: float, y: float, operation: str) -> float:
    if not isinstance(x, (int, float)):
        raise TypeError(f"x must be int or float, got {type(x).__name__}")
    if not isinstance(y, (int, float)):
        raise TypeError(f"y must be int or float, got {type(y).__name__}")
    
    operations: Dict[str, Callable[[float, float], float]] = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else float('inf')
    }
    
    if operation in operations:
        return operations[operation](x, y)
    else:
        raise ValueError(f"Invalid operation: '{operation}'. Valid operations are: {', '.join(operations.keys())}")

result: float = calculate(10, 5, "add")
print(result)