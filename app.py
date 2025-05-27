import streamlit as st
from langgraph.graph import StateGraph, END
from typing import List, Dict, Optional, TypedDict
from langchain_core.tools import tool
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
import os
import requests


from dotenv import load_dotenv
load_dotenv(override=True)

# Your API Keys
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

LLM_DEPLOYMENT_ID = os.getenv("LLM_DEPLOYMENT_ID")

# Define the State Schema
class ProductSearchState(TypedDict):
    user_input: str
    amazon_products: Optional[List[Dict]]
    google_products: Optional[List[Dict]]
    ranked_products: Optional[List[Dict]]
    output_summary: Optional[str]

# Set up the LLM
llm = ChatOpenAI(temperature=0, deployment_id=LLM_DEPLOYMENT_ID)

# Define product search tools
@tool
def search_amazon_products(query: str) -> List[Dict]:
    """Search for products on Amazon using RapidAPI."""
    try:
        url = "https://real-time-amazon-data.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "real-time-amazon-data.p.rapidapi.com"
        }
        params = {"country": "IN", "query": query, "page": "1", "sort_by": "RELEVANCE", "product_condition": "ALL"}

        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            products = []
            for p in data.get("data", {}).get("products", [])[:5]:
                products.append({
                    "title": p.get("product_title"),
                    "price": p.get("product_price"),
                    "rating": float(p.get("product_star_rating", "0") or 0),
                    "link": p.get("product_url") or "Link not available"
                })
            return products if products else fallback_amazon()
        else:
            return fallback_amazon()
    except Exception as e:
        print(f"[Amazon Error] {e}")
        return fallback_amazon()

def fallback_amazon():
    return [
        {"title": "Fallback Product A", "price": "₹50,000", "rating": 4.3, "link": "https://amazon.in/fallback-a"},
        {"title": "Fallback Product B", "price": "₹45,000", "rating": 4.1, "link": "https://amazon.in/fallback-b"},
    ]

@tool
def search_google_shopping(query: str) -> List[Dict]:
    """Search for products on Google Shopping using SerpAPI."""

    try:
        url = "https://serpapi.com/search.json"
        params = {
            "q": query,
            "engine": "google_shopping",
            "hl": "en",
            "gl": "in",
            "api_key": SERPAPI_KEY
        }
        response = requests.get(url, params=params, timeout=10)

        products = []
        if response.status_code == 200:
            data = response.json()
            for item in data.get("shopping_results", [])[:5]:
                products.append({
                    "title": item.get("title"),
                    "price": item.get("price", "N/A"),
                    "rating": float(item.get("rating", 0.0)),
                    "link": item.get("link")
                })
        return products if products else fallback_google()
    except Exception as e:
        print(f"[Google Shopping Error] {e}")
        return fallback_google()

def fallback_google():
    return [
        {"title": "Fallback Google Product A", "price": "₹48,000", "rating": 4.0, "link": "https://google.com/fallback-a"},
        {"title": "Fallback Google Product B", "price": "₹52,000", "rating": 4.2, "link": "https://google.com/fallback-b"},
    ]

# Define each Node
def extract_product(state: ProductSearchState) -> ProductSearchState:
    return {"user_input": state["user_input"]}

def search_amazon(state: ProductSearchState) -> ProductSearchState:
    return {"amazon_products": search_amazon_products(state["user_input"])}

def search_google(state: ProductSearchState) -> ProductSearchState:
    return {"google_products": search_google_shopping(state["user_input"])}

def rank_products(state: ProductSearchState) -> ProductSearchState:
    amazon = state.get("amazon_products") or []
    google = state.get("google_products") or []
    all_products = amazon + google
    ranked = sorted(all_products, key=lambda x: x.get("rating", 0.0), reverse=True)[:3]

    summary_lines = [f"🔍 **Extracted product name:** {state['user_input']}", ""]
    summary_lines.append("--- 🛒 **Amazon Products** ---")
    for i, p in enumerate(amazon, 1):
        summary_lines.append(f"{i}. {p['title']} - {p['price']} - {p['rating']} ⭐ [Amazon]\n   Link: {p['link']}")
       

    summary_lines.append("--- 🛒 **Google Products** ---")
    for i, p in enumerate(google, 1):
        summary_lines.append(f"{i}. {p['title']} - {p['price']} - {p['rating']} ⭐ [Google]\n   Link: {p['link']}")
        
    summary_lines.append("=== 🏆 **Top 3 Products Overall** ===")
    for i, p in enumerate(ranked, 1):
        summary_lines.append(f"{i}. {p['title']} - {p['price']} - {p['rating']} ⭐\n   Link: {p['link']}")
        

    return {
        "ranked_products": ranked,
        "output_summary": "\n".join(summary_lines)
    }

# Build LangGraph
builder = StateGraph(ProductSearchState)
builder.add_node("extract_product", extract_product)
builder.add_node("search_amazon", search_amazon)
builder.add_node("search_google", search_google)
builder.add_node("rank_products", rank_products)

builder.set_entry_point("extract_product")
builder.add_edge("extract_product", "search_amazon")
builder.add_edge("search_amazon", "search_google")
builder.add_edge("search_google", "rank_products")
builder.add_edge("rank_products", END)

graph = builder.compile()

# Streamlit UI
def run_product_search():
    st.title("Product Comparison Tool")

    user_input_text = st.text_input("Enter your product query (e.g., 'iPhone 14'):")

    if st.button("Search"):
        if user_input_text:
            input_state = {"user_input": user_input_text}
            result = graph.invoke(input_state)

            st.subheader("Comparison Summary")
            st.markdown(result["output_summary"])
        else:
            st.warning("Please enter a product query.")

if __name__ == "__main__":
    run_product_search()



