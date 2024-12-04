from glob import iglob
import re
import logging
from openai import OpenAI, AzureOpenAI
from tenacity import retry,  stop_after_attempt,wait_random
from termcolor import colored
import time
import json
import traceback
import os
import os.path as path

if __name__ != '__main__':
    logger = logging.getLogger('root.divider')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('logs/'+__name__+'.log', mode='w')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@retry(wait=wait_random(0.2,2), stop=stop_after_attempt(1))
def chat_completion_request(key, base_url, messages, tools=None, tool_choice=None, key_pos=None,
                            model="gpt-3.5-turbo", stop=None, process_id=0, **args):
    use_messages = []
    for message in messages:
        if not ("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)

    for message in use_messages:
        if 'function_call' in message.keys():
            message.pop('function_call')

    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})

    try:
        if model.startswith("gpt"):
            if ('azure' in base_url):
                client = AzureOpenAI(azure_endpoint=base_url, api_key=key,
                                     api_version='2024-05-01-preview') if base_url else OpenAI(api_key=key)
            else:
                client = OpenAI(
                    api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
        else:
            raise NotImplementedError("Model not supported")
        openai_response = client.chat.completions.create(**json_data)
        json_data = openai_response.dict()
        return json_data

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        traceback.print_exc()
        # import pdb;  pdb.set_trace()
        return {"error": str(e), "total_tokens": 0}


class ChatGPTFunction:
    def __init__(self, model="gpt-4-turbo-2024-04-09", openai_key="", base_url=None):
        self.model = model
        self.conversation_history = []
        self.openai_key = openai_key
        self.base_url = base_url
        self.time = time.time()
        self.TRY_TIME = 1

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self, messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        print("before_print" + "*" * 50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + \
                    f"function_call: {message['function_call']}"
            if 'tool_calls' in message.keys():
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + \
                    f"number of tool calls: {len(message['tool_calls'])}"
            if detailed:
                print_obj = print_obj + \
                    f"function_call: {message['function_call']}"
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + \
                    f"function_call_id: {message['function_call_id']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )
        print("end_print" + "*" * 50)

    def parse(self, tools, process_id, key_pos=None, **args):

        self.time = time.time()
        conversation_history = self.conversation_history
        for _ in range(self.TRY_TIME):
            if _ != 0:
                time.sleep(2)
            if tools != []:
                response = chat_completion_request(
                    self.openai_key, self.base_url, conversation_history, tools=tools, process_id=process_id, key_pos=key_pos,
                    model=self.model, **args
                )
            else:
                response = chat_completion_request(
                    self.openai_key, self.base_url, conversation_history, process_id=process_id, key_pos=key_pos, model=self.model, **args
                )
            try:
                total_tokens = response['usage']['total_tokens']
                message = response["choices"][0]["message"]

                if process_id == 0:
                    print(
                        f"[process({process_id})]total tokens: {total_tokens}")

                return message, 0, total_tokens
            except BaseException as e:
                print(
                    f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                traceback.print_exc()
                if response is not None:
                    print(f"[process({process_id})]OpenAI return: {response}")

        return {"role": "assistant", "content": str(response)}, -1, 0


if __name__ == '__main__':
    history = [{'role': 'user', 'content': 'hello'}]
    llm = ChatGPTFunction(model='gpt-3.5-turbo', openai_key='sk-sfX6bec51b1311eeb2032f49df1be13371509de8360sQET4',
                          base_url='https://api.gptsapi.net/v1')
    llm.change_messages(history)
    res = llm.parse(tools = [], process_id=0)
    print(res)
