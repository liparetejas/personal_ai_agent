"""
Unified Multi-Function Assistant - Batak

A comprehensive AI assistant that can:
- Write and send emails
- Read and analyze PDF documents
- Schedule meetings on Google Calendar
- Search the Internet for real-time information
- Ask questions when uncertain or needing private information

Modified to use local Ollama GPT-OSS 20B model instead of Gemini
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import ollama
import asyncio
import os
from typing import List, Dict, Any

# Store conversation history for context management
conversation_history = []

def build_prompt_from_history(history):
    """Build a formatted conversation prompt from history."""
    messages = []
    for entry in history:
        role = entry["role"]
        for part in entry["parts"]:
            if role == "user":
                messages.append({
                    "role": "user",
                    "content": part
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": part
                })
    return messages

async def process_user_request(
    user_request: str,
    sessions: Dict[str, ClientSession],
    system_instruction: str
) -> str:
    """
    Process user request using local Ollama model with all available MCP tools.

    Args:
        user_request: Natural language description of what to do
        sessions: Dictionary of active MCP ClientSessions (pdf, email, calendar, websearch)
        system_instruction: System prompt for Batak

    Returns:
        Ollama's response text
    """
    global conversation_history

    # Add user request to conversation history
    conversation_history.append({
        "role": "user",
        "parts": [user_request.strip()]
    })

    # Prepare tools list from all sessions - for context
    all_tools_desc = []
    for session_name, session in sessions.items():
        if session:
            all_tools_desc.append(f"{session_name} tools available")

    # Build messages for Ollama
    messages = build_prompt_from_history(conversation_history)

    # Add system instruction to the first message or as separate context
    full_messages = [
        {"role": "system", "content": system_instruction}
    ] + messages

    try:
        # Call local Ollama model (synchronously, then we'll handle it)
        # Ollama python library doesn't have native async support for chat
        # So we'll use asyncio.to_thread to run it in a thread
        response = await asyncio.to_thread(
            ollama.chat,
            model='gpt-oss-20b',
            messages=full_messages
        )

        response_text = response['message']['content']

        # Add assistant response to conversation history
        conversation_history.append({
            "role": "model",
            "parts": [response_text]
        })

        return response_text

    except Exception as e:
        error_msg = f"Error calling Ollama: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

async def main():
    global conversation_history

    print("\n🚀 Starting Unified Batak Assistant with Local GPT-OSS 20B...\n")

    # Check if Ollama is running
    try:
        models = ollama.list()
        print("✅ Ollama is running")

        # Check if gpt-oss-20b model is available
        model_names = [model['name'] for model in models['models']]
        if 'gpt-oss-20b' in model_names or 'gpt-oss-20b:latest' in model_names:
            print("✅ GPT-OSS 20B model found")
        else:
            print("⚠️  Warning: GPT-OSS 20B model not found in available models")
            print(f"   Available models: {model_names}")
            print("   Make sure 'ollama serve' is running in another terminal")
            print("   and you have pulled the model with: ollama pull gpt-oss-20b")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("   Make sure 'ollama serve' is running in another terminal")
        return

    # ========== MCP Server Configuration ==========
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # PDF Server (Python)
    pdf_server_path = os.path.join(base_dir, "mcp-pdf-reader", "pdf_server.py")
    pdf_env = os.environ.copy()
    pdf_env['LD_LIBRARY_PATH'] = '/sw/eb/sw/Python/3.10.8-GCCcore-12.2.0/lib:' + pdf_env.get('LD_LIBRARY_PATH', '')
    pdf_params = StdioServerParameters(
        command="python",
        args=[pdf_server_path],
        env=pdf_env
    )

    # Email Server (Node)
    email_server_path = os.path.join(base_dir, "email-mcp", "dist", "index.js")
    email_params = StdioServerParameters(
        command="node",
        args=[email_server_path]
    )

    # Calendar Server (Node)
    calendar_server_path = os.path.join(base_dir, "google-calendar-mcp", "build", "index.js")
    calendar_credentials_path = os.path.join(base_dir, "google-calendar-mcp", "client_secret_1091068134059-p2r1i8vq4um0mevudull2rkfrejujrii.apps.googleusercontent.com.json")
    calendar_env = os.environ.copy()
    calendar_env['GOOGLE_OAUTH_CREDENTIALS'] = calendar_credentials_path
    calendar_params = StdioServerParameters(
        command="node",
        args=[calendar_server_path],
        env=calendar_env
    )

    # Web Search Server (NPX)
    websearch_params = StdioServerParameters(
        command="npx",
        args=["-y", "@brightdata/mcp"],
        env={
            "API_TOKEN": os.getenv("BRIGHT_DATA_API_KEY"),
            "PRO_MODE": "true"
        }
    )

    # ========== Unified System Instruction ==========
    system_instruction = """Your name is Batak 🦆. You are a comprehensive personal assistant for Tejas.

You have access to multiple capabilities and should intelligently choose the right tools based on user requests:

**EMAIL CAPABILITIES:**
- Compose and send professional emails
- Use send_email tool with parameters: to, subject, body, html (optional)
- Maintain professional tone and proper email etiquette
- Always add signature: "This email has been sent by Tejas' personal assistant Batak 🦆.\nRegards,\nBatak"

**PDF CAPABILITIES:**
- Read and analyze PDF documents
- Extract text, perform OCR, and extract images
- Tools: read_pdf_text, read_by_ocr, read_pdf_images
- PDF files can be in pdf_resources/ directory or use absolute paths
- Provide page references when citing information

**CALENDAR CAPABILITIES:**
- Manage Google Calendar events
- Tools: list-calendars, list-events, search-events, create-event, update-event, delete-event, get-freebusy
- Check for conflicts before scheduling
- Use ISO 8601 format for dates and times (e.g., "2025-10-21T14:00:00-05:00")
- Confirm event details before creating

**WEB SEARCH CAPABILITIES:**
- Search the internet for real-time information
- Use search_engine tool with parameters: query, engine (set to "google")
- Provide comprehensive, well-organized summaries
- Cite sources when appropriate
- Present objective, unbiased information

**ASKING QUESTIONS:**
- When you need private information (email addresses, specific dates, preferences) that isn't in the current context, ASK the user
- When you're uncertain about what action to take, ASK for clarification
- When multiple options exist, ASK which one the user prefers
- Be proactive in gathering missing information rather than making assumptions
- Format questions clearly and wait for user response

**GENERAL GUIDELINES:**
1. Intelligently determine which capability/tool to use based on user request
2. You can use multiple tools in sequence if needed (e.g., search for information, then compose email)
3. Maintain context across the conversation - remember previous interactions
4. Always confirm actions after completion with relevant details
5. Be helpful, accurate, and proactive
6. When uncertain, ask clarifying questions rather than guessing
7. If the user's request is ambiguous, ask for clarification before proceeding

Always provide clear, well-structured responses and take appropriate actions based on user requests."""

    # ========== Start All MCP Servers ==========
    try:
        # Start PDF server
        async with stdio_client(pdf_params) as pdf_transport:
            pdf_read, pdf_write = pdf_transport
            async with ClientSession(pdf_read, pdf_write) as pdf_session:
                await pdf_session.initialize()
                pdf_tools = await pdf_session.list_tools()
                print(f"✅ PDF Server Connected")
                print(f"   📄 Available Tools: {[tool[0] for tool in pdf_tools]}")

                # Start Email server
                async with stdio_client(email_params) as email_transport:
                    email_read, email_write = email_transport
                    async with ClientSession(email_read, email_write) as email_session:
                        await email_session.initialize()
                        email_tools = await email_session.list_tools()
                        print(f"✅ Email Server Connected")
                        print(f"   📧 Available Tools: {[tool[0] for tool in email_tools]}")

                        # Start Calendar server
                        async with stdio_client(calendar_params) as calendar_transport:
                            calendar_read, calendar_write = calendar_transport
                            async with ClientSession(calendar_read, calendar_write) as calendar_session:
                                await calendar_session.initialize()
                                calendar_tools = await calendar_session.list_tools()
                                print(f"✅ Calendar Server Connected")
                                print(f"   📅 Available Tools: {[tool[0] for tool in calendar_tools]}")

                                # Start Web Search server
                                async with stdio_client(websearch_params) as websearch_transport:
                                    websearch_read, websearch_write = websearch_transport
                                    async with ClientSession(websearch_read, websearch_write) as websearch_session:
                                        await websearch_session.initialize()
                                        websearch_tools = await websearch_session.list_tools()
                                        print(f"✅ Web Search Server Connected")
                                        print(f"   🔍 Available Tools: {[tool.name if hasattr(tool, 'name') else tool[0] for tool in websearch_tools]}")

                                        # Create sessions dictionary
                                        sessions = {
                                            "pdf": pdf_session,
                                            "email": email_session,
                                            "calendar": calendar_session,
                                            "websearch": websearch_session
                                        }

                                        # ========== Interactive Loop ==========
                                        print("\n" + "=" * 70)
                                        print("🦆 Batak - Your Unified Personal Assistant (Powered by Local GPT-OSS 20B)")
                                        print("=" * 70)
                                        print("💡 I can help you with:")
                                        print("   📧 Sending emails")
                                        print("   📄 Reading and analyzing PDFs")
                                        print("   📅 Managing your Google Calendar")
                                        print("   🔍 Searching the web for information")
                                        print("=" * 70)

                                        while True:
                                            user_request = input("\n🗣️  What can I help you with? (type 'quit' to exit)\n> ").strip()

                                            if user_request.lower() == "quit":
                                                print("\n👋 I hope I was useful. Goodbye Tejas!")
                                                break

                                            if not user_request:
                                                print("⚠️  Please enter a valid request.")
                                                continue

                                            print("\n🤔 Batak is working on your request...\n")

                                            try:
                                                response = await process_user_request(
                                                    user_request=user_request,
                                                    sessions=sessions,
                                                    system_instruction=system_instruction
                                                )

                                                print(f"🦆 Batak's Response:\n{response}\n")
                                                print("=" * 70)

                                            except Exception as e:
                                                print(f"❌ Error while processing request: {e}")
                                                import traceback
                                                traceback.print_exc()

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for required environment variables
    missing_keys = []

    # No longer need GEMINI_API_KEY since we're using local Ollama
    # if not os.getenv("GEMINI_API_KEY"):
    #     missing_keys.append("GEMINI_API_KEY")

    if not os.getenv("BRIGHT_DATA_API_KEY"):
        missing_keys.append("BRIGHT_DATA_API_KEY")
    else:
        print("✅ BRIGHT_DATA_API_KEY found")

    if missing_keys:
        print(f"\n❌ ERROR: Missing required environment variables: {', '.join(missing_keys)}")
        print("\nPlease set them using:")
        for key in missing_keys:
            print(f"   export {key}='your-api-key'")
        if "BRIGHT_DATA_API_KEY" in missing_keys:
            print("\nGet Bright Data API key from: https://brightdata.com")

    print("\n🔧 Using Local Ollama GPT-OSS 20B Model")
    print("   Make sure 'ollama serve' is running in another terminal!")
    print("   The model should be available via: ollama list\n")

    asyncio.run(main())
