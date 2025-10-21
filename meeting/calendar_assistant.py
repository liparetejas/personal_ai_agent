"""

Interactive Gemini Calendar Assistant - Batak

An interactive AI assistant that can read and schedule meetings using Google Calendar.

"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
import asyncio
import os

async def manage_calendar_with_gemini(user_request: str, session, gemini_client, system_instruction: str):
    """
    Manage Google Calendar using Gemini to understand and execute based on user request.

    Args:
        user_request: Natural language description of the calendar operation
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
            tools=[session],  # MCP Google Calendar tools available to Gemini
        ),
    )

    return response.text

async def main():
    # Path to the Google Calendar MCP server script
    # Update this path to point to your google-calendar-mcp installation
    server_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/google-calendar-mcp/build/index.js")
    print(f"🚀 Starting MCP Google Calendar server from: {server_path}\n")

    # Get the OAuth credentials path
    credentials_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/google-calendar-mcp/client_secret_1091068134059-p2r1i8vq4um0mevudull2rkfrejujrii.apps.googleusercontent.com.json")

    # Prepare environment variables for the subprocess
    env = os.environ.copy()
    env['GOOGLE_OAUTH_CREDENTIALS'] = credentials_path

    # Define the command to launch the MCP Google Calendar server subprocess
    params = StdioServerParameters(
        command="node",
        args=[server_path],
        env=env
    )

    # Initialize Gemini client
    gemini_client = genai.Client()

    # System prompt for Batak, your personal calendar assistant
    system_instruction = """Your name is Batak. You are a personal assistant of Tejas.
Your primary responsibility is to help Tejas manage his Google Calendar effectively, including reading events and scheduling meetings.

When asked to work with the calendar, you should:
1. Understand the context and purpose of the request
2. Use appropriate calendar tools to query or modify events
3. Provide clear summaries of calendar information
4. Schedule meetings with appropriate details (title, time, location, attendees)
5. Handle conflicts and suggest alternative times when needed
6. Use natural language understanding for dates and times

You have access to Google Calendar management tools:
- list-calendars: List all available calendars
- list-events: List events with date filtering and options
- search-events: Search events by text query
- create-event: Create new calendar events with details like summary, description, start/end time, attendees, location
- update-event: Update existing events (modify time, add attendees, change details)
- delete-event: Delete events from calendar
- get-freebusy: Check availability across calendars to find free time slots

When scheduling meetings:
- Always confirm event details before creating
- Check for conflicts using get-freebusy if appropriate
- Provide event IDs after creating events
- Use ISO 8601 format for dates and times (e.g., "2025-10-21T14:00:00-05:00")

Always be helpful, accurate, and proactive in managing Tejas's calendar.
After performing calendar operations, confirm the action to Tejas with relevant details."""

    try:
        # Launch the MCP server automatically as a subprocess and connect over STDIO
        async with stdio_client(params) as transport:
            read, write = transport
            async with ClientSession(read, write) as session:
                # Initialize MCP session
                await session.initialize()

                tools = await session.list_tools()
                print(f"✅ MCP Server Connected")
                print(f"📅 Available Tools: {[tool[0] for tool in tools]}\n")

                print("=" * 70)
                print("🦆 Batak - Your Personal Calendar Assistant")
                print("=" * 70)
                print("💡 Tip: You can ask me to check your schedule, create meetings,")
                print("    find free time slots, or manage your calendar events!")
                print("=" * 70)

                while True:
                    user_request = input("\n🗣️  What should Batak do? (type 'quit' to exit)\n> ").strip()

                    if user_request.lower() == "quit":
                        print("\n👋 I hope I was useful. Goodbye Tejas!")
                        break

                    if not user_request:
                        print("⚠️  Please enter a valid request.")
                        continue

                    print("🤔 Batak is working on your calendar request...\n")

                    try:
                        response = await manage_calendar_with_gemini(
                            user_request=user_request,
                            session=session,
                            gemini_client=gemini_client,
                            system_instruction=system_instruction
                        )

                        print(f"📅 Batak's Response:\n{response}\n")
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
