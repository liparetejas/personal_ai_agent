"""
Interactive Gemini Email Assistant - Batak
An interactive AI assistant that can compose and send emails based on conversational input.
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
import asyncio
import os


async def send_email_with_gemini(user_request: str, session, gemini_client, system_instruction: str):
    """
    Send an email using Gemini to compose content based on user request.
    
    Args:
        user_request: Natural language description of the email to send
        session: Active MCP ClientSession
        gemini_client: Initialized Gemini client
        system_instruction: System prompt for Batak
    
    Returns:
        Gemini's response text
    """
    signature_text = ("\n\nThis email has been sent by Tejas' personal assistant Batak 🦆.\n"
                      "Regards,\nBatak")
    full_request = user_request.strip() + signature_text
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=full_request,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            tools=[session],  # MCP email tools available to Gemini
        ),
    )
    return response.text


async def main():
    # Expand environment variable to absolute path to MCP server script
    server_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/email-mcp/dist/index.js")
    print(f"🚀 Starting MCP email server from: {server_path}\n")
    
    # Define the command to launch the MCP email server subprocess
    params = StdioServerParameters(command="node", args=[server_path])
    
    # Initialize Gemini client
    gemini_client = genai.Client()
    
    # System prompt for Batak, your personal AI assistant
    system_instruction = """Your name is Batak. You are a personal assistant of Tejas.

Your primary responsibility is to help Tejas compose and send professional, well-written emails.

When asked to write an email, you should:
1. Understand the context and purpose of the email
2. Compose clear, concise, and appropriate email content
3. Use proper email etiquette and formatting
4. Maintain a professional yet friendly tone
5. Use the send_email tool to actually send the email

You have access to an email sending tool with these parameters:
- to: recipient's email address
- subject: clear and relevant subject line
- body: the email body content
- html (optional): set to true if using HTML formatting

Always be helpful, accurate, and considerate in your email compositions.
After sending an email, confirm the action to Tejas."""
    
    try:
        # Launch the MCP server automatically as a subprocess and connect over STDIO
        async with stdio_client(params) as transport:
            read, write = transport
            
            async with ClientSession(read, write) as session:
                # Initialize MCP session
                await session.initialize()
                tools = await session.list_tools()
                print(f"✅ MCP Server Connected")
                print(f"📧 Available Tools: {[tool[0] for tool in tools]}\n")
                print("=" * 70)
                print("🦆 Batak - Your Personal Email Assistant")
                print("=" * 70)
                
                while True:
                    user_request = input("\n🗣️  What should Batak do? (type 'quit' to exit)\n> ").strip()
                    if user_request.lower() == "quit":
                        print("\n👋 I hope I was useful. Goodbye Tejas!")
                        break

                    if not user_request:
                        print("⚠️ Please enter a valid request.")
                        continue

                    print("🤔 Batak is composing your email...\n")
                    try:
                        response = await send_email_with_gemini(
                            user_request=user_request,
                            session=session,
                            gemini_client=gemini_client,
                            system_instruction=system_instruction
                        )
                        print(f"✉️  Batak's Response:\n{response}\n")
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
