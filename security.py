"""
安全确认逻辑
"""
def confirm_action(message: str) -> bool:
    resp = input(f"{message} [y/N]: ").strip().lower()
    return resp == "y"