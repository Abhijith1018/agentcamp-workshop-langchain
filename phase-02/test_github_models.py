"""
Phase 2: Test GitHub Models Connection
Run with: python test_github_models.py
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()


def main():
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token or github_token == "your_github_token_here":
        print("❌ Error: GITHUB_TOKEN not set in .env file")
        return

    print("🔄 Testing connection to GitHub Models...")

    llm = ChatOpenAI(
        model="openai/gpt-4.1-nano",
        api_key=github_token,
        base_url="https://models.github.ai/inference",
        temperature=0.7,
    )

    try:
        response = llm.invoke("Say 'Hello, Workshop!' and nothing else.")
        print(f"✅ Success! Model responded: {response.content}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
