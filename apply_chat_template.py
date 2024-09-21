def apply_chat_template(messages, add_generation_prompt=False):
    processed_messages = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        processed_messages.append(f"\n\n### {role.upper()}:\n{content}")

    if add_generation_prompt:
        processed_messages.append("\n\n### ASSISTANT:\n")

    return " ".join(processed_messages)
