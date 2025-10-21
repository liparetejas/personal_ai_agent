"""

Interactive Gemini PDF Assistant - Batak

An interactive AI assistant that can read and analyze PDF documents based on conversational input.

"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
import asyncio
import os

async def query_pdf_with_gemini(user_request: str, session, gemini_client, system_instruction: str):
    """
    Query PDF documents using Gemini to understand and answer based on user request.

    Args:
        user_request: Natural language description of the PDF query
        session: Active MCP ClientSession
        gemini_client: Initialized Gemini client
        system_instruction: System prompt for Batak

    Returns:
        Gemini's response text
    """
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_request.strip(),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            tools=[session],  # MCP PDF tools available to Gemini
        ),
    )

    return response.text

async def main():
    # Path to the PDF MCP server script
    # Update this path to point to your pdf_server.py location
    server_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/mcp-pdf-reader/pdf_server.py")
    print(f"🚀 Starting MCP PDF server from: {server_path}\n")

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/sw/eb/sw/Python/3.10.8-GCCcore-12.2.0/lib:' + env.get('LD_LIBRARY_PATH', '')

    params = StdioServerParameters(
        command="python",
        args=[server_path],
        env=env  # Pass environment explicitly
    )

    # Define the command to launch the MCP PDF server subprocess
    # params = StdioServerParameters(command="python", args=[server_path])

    # Initialize Gemini client
    gemini_client = genai.Client()

    # System prompt for Batak, your personal PDF assistant
    system_instruction = """Your name is Batak. You are a personal assistant of Tejas.
Your primary responsibility is to help Tejas read, analyze, and extract information from PDF documents.

When asked to work with PDFs, you should:
1. Understand what information Tejas needs from the PDF(s)
2. Use the appropriate PDF reading tool based on the request:
   - Use read_pdf_text for normal text extraction from PDF pages
   - Use read_by_ocr for scanned PDFs or image-based documents
   - Use read_pdf_images to extract images from PDF pages
3. Analyze and summarize the content clearly
4. Answer questions based on the PDF content
5. Provide page references when citing information

You have access to PDF reading tools with these parameters:
- read_pdf_text: Extract text from PDF (file_path, start_page, end_page)
- read_by_ocr: OCR-based text extraction (file_path, start_page, end_page, language, dpi)
- read_pdf_images: Extract images from PDF (file_path, page_number)

PDF files should be placed in the pdf_resources/ directory or you can use absolute paths.

Always be helpful, accurate, and provide clear answers based on the PDF content.
After processing a PDF, summarize the key findings for Tejas."""

    try:
        # Launch the MCP server automatically as a subprocess and connect over STDIO
        async with stdio_client(params) as transport:
            read, write = transport
            async with ClientSession(read, write) as session:
                # Initialize MCP session
                await session.initialize()

                tools = await session.list_tools()
                print(f"✅ MCP Server Connected")
                print(f"📄 Available Tools: {[tool[0] for tool in tools]}\n")

                print("=" * 70)
                print("🦆 Batak - Your Personal PDF Assistant")
                print("=" * 70)
                print("💡 Tip: Place your PDF files in the pdf_resources/ directory")
                print("=" * 70)

                while True:
                    user_request = input("\n🗣️  What should Batak do? (type 'quit' to exit)\n> ").strip()

                    if user_request.lower() == "quit":
                        print("\n👋 I hope I was useful. Goodbye Tejas!")
                        break

                    if not user_request:
                        print("⚠️  Please enter a valid request.")
                        continue

                    print("🤔 Batak is analyzing your request...\n")

                    try:
                        response = await query_pdf_with_gemini(
                            user_request=user_request,
                            session=session,
                            gemini_client=gemini_client,
                            system_instruction=system_instruction
                        )

                        print(f"📖 Batak's Response:\n{response}\n")
                        print("=" * 70)

                    except Exception as e:
                        print(f"❌ Error while processing request: {e}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        print("\nGet your API key from: https://aistudio.google.com/apikey")
    else:
        print("✅ GEMINI_API_KEY found")
        asyncio.run(main())
