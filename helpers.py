import json
import re


def extract_function_call_from_string(string) -> str | bool:
    pattern = re.compile(r'<functioncall>(.*?)$', re.DOTALL)
    match = pattern.search(string)
    if match:
        json_str = match.group(1).strip().replace("'{", "{").replace("}'", "}")
        try:
            function_call = json.loads(json_str)
            return function_call
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return False
    else:
        print("Invalid JSON format")
        return False


def has_function_call(s) -> bool:
    """Checks if string has function call"""
    return '<functioncall>' in s


if __name__ == '__main__':
    string = '<functioncall> {"name": "wiki_func", "arguments": \'{"query": "monkeys"}\'}'
    test = extract_function_call_from_string(string)
    print(test)
