import os
from dotenv import load_dotenv
from tavily import TavilyClient
import json

load_dotenv()
tavily_client = TavilyClient(api_key=os.environ["TAVILY"])


def search_items():
    query = input("What item would you like to search: ")
    WrongInput = True
    while WrongInput == True:
        engine = input("What engine do you wanna use:  \n 1 - Shopee \n 2 - Lazada \n > ")
        if engine == "1":
            results = search_shopee(query)
            WrongInput = False
        elif engine == "2":
            results = search_lazada(query)
            WrongInput = False
        else:
            print("Invalid Input, Try again")
            WrongInput = True
            continue

        print(json.dumps(results, indent=2))

def _search_site(query: str, site: str, max_results: int = 3) -> list[dict]:
    scoped_query = f"site:{site} {query}"
    response = tavily_client.search(
        query=scoped_query,
        max_results=max_results,
        include_raw_content=False,
    )
    return [
        {
            "title": r["title"][:100],
            "url": r["url"],
            "snippet": r["content"][:200],  # truncate to keep token usage low
        }
        for r in response.get("results", [])
    ]


def search_shopee(query: str) -> list[dict]:
    return _search_site(query, "shopee.com.my")


def search_lazada(query: str) -> list[dict]:
    return _search_site(query, "lazada.com")


if __name__ == "__main__":
    search_items()