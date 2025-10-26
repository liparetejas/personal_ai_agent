"""
Interactive Web Search Assistant with Gemini - Batak
An interactive AI assistant that performs web search and retrieves real-time results 
using Bright Data's Web MCP server, with Gemini Flash 2.5 extracting relevant information.
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
import asyncio
import os


async def perform_web_search_with_gemini(user_query: str, session, gemini_client, system_instruction: str):
    """
    Perform web search and use Gemini to extract relevant information from results.

    Args:
        user_query: Natural language search query from user
        session: Active MCP ClientSession
        gemini_client: Initialized Gemini client
        system_instruction: System prompt for Batak

    Returns:
        Gemini's processed response with extracted relevant information
    """
    # Prepare the full request with context
    search_context = (
        f"Search Query: {user_query}\n\n"
        "Please search the web for this query and provide a comprehensive, "
        "well-organized summary of the most relevant information you find."
    )

    # Use Gemini with MCP tools to search and process results
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=search_context,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            tools=[session],  # MCP web search tools available to Gemini
        ),
    )

    return response.text


async def main():
    # Path to the MCP server script; uses npx to launch brightdata-mcp
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@brightdata/mcp"],
        env={
            "API_TOKEN": os.getenv("BRIGHT_DATA_API_KEY"),
            "PRO_MODE": "true"  # Optional, exposes all tools
        }
    )

    # Initialize Gemini client
    gemini_client = genai.Client()

    # System prompt for Batak, your web search assistant
    system_instruction = """Your name is Batak. You are a helpful, real-time web search assistant for Tejas.
You can find and summarize live information, news, and current events from the internet when prompted.

Your responsibilities include:
1. Understanding the user's search intent
2. Using the search_engine tool to retrieve relevant search results from the web
3. Analyzing and extracting the most relevant information from search results
4. Presenting clear, comprehensive, and well-organized summaries
5. Citing sources when appropriate
6. Providing objective, unbiased information

When using the search_engine tool:
- Use parameter "query" for the search query
- Use parameter "engine" set to "google" for Google search
- Process the results to extract key information
- Organize information in a clear, readable format

Always be precise, comprehensive, and unbiased in your responses.
Format your responses with clear structure using headings, bullet points, or numbered lists when appropriate.
"""

    try:
        async with stdio_client(server_params) as transport:
            read, write = transport
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print("✅ MCP Server Connected (Bright Data Web MCP)")
                print(f"🔍 Available Tools: {[tool.name if hasattr(tool, 'name') else tool[0] for tool in tools]}")
                print("="*70)
                print("🦆 Batak - Your Real-Time Web Search Assistant")
                print("="*70)

                while True:
                    user_query = input("\n🔎 What should Batak search? (type 'quit' to exit)\n> ").strip()
                    if user_query.lower() == "quit":
                        print("\n👋 I hope I was useful. Goodbye Tejas!")
                        break
                    if not user_query:
                        print("⚠️  Please enter a valid search query.")
                        continue

                    print("🤔 Batak is searching the web and analyzing results...\n")
                    try:
                        response = await perform_web_search_with_gemini(
                            user_query=user_query,
                            session=session,
                            gemini_client=gemini_client,
                            system_instruction=system_instruction
                        )
                        print("📊 Batak's Analysis:")
                        print(response)
                        print("="*70)
                    except Exception as e:
                        print(f"❌ Error while processing request: {e}")
                        import traceback
                        traceback.print_exc()

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for required API keys
    missing_keys = []

    if not os.getenv("BRIGHT_DATA_API_KEY"):
        missing_keys.append("BRIGHT_DATA_API_KEY")
    else:
        print("✅ BRIGHT_DATA_API_KEY found")

    if not os.getenv("GEMINI_API_KEY"):
        missing_keys.append("GEMINI_API_KEY")
    else:
        print("✅ GEMINI_API_KEY found")

    if missing_keys:
        print(f"\n❌ ERROR: Missing required environment variables: {', '.join(missing_keys)}")
        print("\nPlease set them using:")
        for key in missing_keys:
            print(f"  export {key}='your-api-key'")
        if "BRIGHT_DATA_API_KEY" in missing_keys:
            print("\nGet Bright Data API key from: https://brightdata.com")
        if "GEMINI_API_KEY" in missing_keys:
            print("Get Gemini API key from: https://aistudio.google.com/apikey")
    else:
        print("\n🚀 Starting Batak Web Search Assistant...\n")
        asyncio.run(main())
