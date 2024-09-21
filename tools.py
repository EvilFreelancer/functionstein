from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchResults
)

AVAILABLE_FUNCTIONS = [
    'arxiv_func',
    'wiki_func',
    'search_func',
]


def is_available_function(functino_name: str) -> bool:
    if functino_name in AVAILABLE_FUNCTIONS:
        return True
    return False


def arxiv_func(top_k_results=2, doc_content_chars_max=8000):
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max
    )
    arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    # arxiv_tool.invoke("Chain-of-Thought")
    return arxiv_tool


def wiki_func(lang='en', top_k_results=1, doc_content_chars_max=8000):
    wiki_wrapper = WikipediaAPIWrapper(
        lang=lang,
        top_k_results=top_k_results,
        doc_content_chars_max=doc_content_chars_max
    )
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    # wiki_tool.invoke("Теорема Лапласа")
    return wiki_tool


def search_func():
    search_tool = DuckDuckGoSearchResults()
    # search_tool.invoke("Обучение ruGPT-3.5")
    return search_tool


if __name__ == '__main__':
    # arxiv_tool = arxiv_func()
    # test = arxiv_tool.invoke("Chain of Thoughts")

    # wiki_tool = wiki_func()
    # test = wiki_tool.invoke("Теорема Лапласа")

    search_tool = search_func()
    test = search_tool.invoke("Обучение ruGPT-3.5")

    print(test)
