import yaml
import json
from ollama import Client

from helpers import has_function_call, extract_function_call_from_string
from tools import is_available_function, arxiv_func, wiki_func, search_func
from chat_history import ChatHistory

from nanogpt import nanoGPT

# Init nanoGPT model
model = nanoGPT()
# test = model.predict('### SYSTEM:\n')
# print(test)
# exit()


# Load config from source root
config = yaml.safe_load(open('config.yml'))
functions = [function for function in config['functions'].values()]

# Select function call model
# https://ollama.com/calebfahlgren/natural-functions
model_name = "calebfahlgren/natural-functions"

# Start chat loop
chat_history = ChatHistory(functions=functions[0], history_limit=10)
while True:

    ### 1. Запрос пользователя

    user_message = input("User: ")
    # user_message = "Find in arxiv papers about chain of thoughts"
    # user_message = "Find in duckduckgo how reproduce segfault in php script"
    # user_message = "Find in wiki something about monkeys"

    # Reset chat command
    if user_message.strip() == "/reset":
        chat_history = ChatHistory(functions=functions[0], history_limit=10)
        print("History reset completed!")
        continue

    # Skip empty messages from user
    if user_message.strip() == "":
        continue

    ### 2. Запрос передаётся на вход LLM

    from apply_chat_template import apply_chat_template

    chat_history.add_user_message(user_message)
    messages = chat_history.get_messages()
    messages_formated = apply_chat_template(messages, add_generation_prompt=True)
    model_response = model.predict(messages_formated)

    print(model_response)
    exit()

    # Read model response and save it to chat history
    output = model_response['message']['content']
    chat_history.add_assistant_message(output)
    # print(output)

    ### 3. Решаем нужно ли вызвать и если да то какой

    # If response has function call
    function_call = None
    if has_function_call(output):
        # Extract functino call object
        function_call = extract_function_call_from_string(output)
        function_name = function_call['name']
        function_args = function_call['arguments']

        if not is_available_function(function_name):
            # print(f">>> {function_name} is not available")
            continue

        print(function_call)

        ### 4. Запрос в тул function call

        # print(f">>> {function_name} is available")
        function_response = ''
        if function_name == 'arxiv_func':
            function_response = arxiv_func().invoke(function_args['query'])
        if function_name == 'wiki_func':
            function_response = wiki_func().invoke(function_args['query'])
        if function_name == 'search_func':
            function_response = search_func().invoke(function_args['query'])

        print('<functionresponse>')
        print(function_response)
        print('</functionresponse>')

        # ---

        ### 5. Ответ тула function response

        # Add function response as user
        chat_history.add_user_message('FUNCTION RESPONSE: ' + json.dumps({"result": function_response}))

    ### 6. Интерпретация ответа

    messages = chat_history.get_messages()

    # print(messages)
    # break

    model_response = client.chat(
        model=model_name, messages=messages,
        options={
            "num_predict": 1024,
            "num_ctx": 32768,
            "temperature": 0,
            "repetition_penalty": 1.1
        }
    )

    # Read model response and save it to chat history
    output = model_response['message']['content']
    chat_history.add_assistant_message(output)
    print("Bot: " + output)
