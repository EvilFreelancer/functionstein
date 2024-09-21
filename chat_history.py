class ChatHistory:
    def __init__(self, functions: list, history_limit: int = None):
        self.functions = functions
        self.history_limit = history_limit

        self.system_prompt = \
            f"You are a helpful assistant with access to the following functions. Use them if required - \n" \
            f"{str(functions)}\n\n" \
            f"<functioncall> {{ \"name\": \"function_func\", \"arguments\": {{\"query\": \"search string\"}} }}\n\n" \
            f"Edge cases you must handle:\n" \
            f" - If there are no functions that match the user request, you will respond using your knowledge base."

        self.messages = [{
            "role": "system",
            "content": self.system_prompt
        }]

    def add_message(self, role, message):
        self.messages.append({
            "role": role,
            "content": message
        })
        self.trim_history()

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_assistant_message(self, message):
        self.add_message("assistant", message)

    def trim_history(self):
        appendix = 0
        if self.system_prompt is not None:
            appendix = 1
        if self.history_limit is not None and len(self.messages) > self.history_limit + appendix:
            overflow = len(self.messages) - (self.history_limit + appendix)
            self.messages = [self.messages[0]] + self.messages[overflow + appendix:]

    def get_last_message(self, role: str = "user"):
        for message in reversed(self.messages):
            if message["role"] == role:
                return message["content"]
        return None

    def get_messages(self) -> list:
        return self.messages
