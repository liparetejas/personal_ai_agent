"""
Interactive Pizza Ordering Assistant - Batak
An interactive AI assistant that helps order Domino's pizza using the mcpizza MCP server.
NOTE: This assistant does NOT place actual orders - it generates a review link/summary for you to manually place the order.
"""

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
from google.genai import types
import asyncio
import os
import json


async def handle_pizza_request_with_gemini(user_request: str, session, gemini_client, system_instruction: str):
    """
    Handle pizza ordering requests using Gemini to intelligently use MCP tools.

    Args:
        user_request: Natural language pizza order request from user
        session: Active MCP ClientSession
        gemini_client: Initialized Gemini client
        system_instruction: System prompt for Batak

    Returns:
        Gemini's processed response with order details
    """
    response = await gemini_client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_request,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            tools=[session],  # MCP pizza ordering tools available to Gemini
        ),
    )

    return response.text


async def main():
    # Path to the MCP pizza server script in $SCRATCH
    # server_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/mcpizza/mcpizza/server.py")
    server_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/mcpizza/dist/index.js")
    print(f"🍕 Starting MCPizza server from: {server_path}\n")

    # Define the command to launch the MCP pizza server subprocess
    # env = os.environ.copy()
    # env['LD_LIBRARY_PATH'] = '/sw/eb/sw/Python/3.10.8-GCCcore-12.2.0/lib:' + env.get('LD_LIBRARY_PATH', '')

    # params = StdioServerParameters(
    #     command="python",
    #     args=[server_path],
    #     env=env  # Pass environment explicitly
    # )
    params = StdioServerParameters(command="node", args=[server_path])

    # Initialize Gemini client
    gemini_client = genai.Client()

    # System prompt for Batak, your pizza ordering assistant
    system_instruction = """Your name is Batak. You are a personal assistant for Tejas who helps him order Domino's pizza.

IMPORTANT: You do NOT place actual orders. Your job is to help build the order and then generate a detailed summary/review that Tejas can use to manually place the order.

Your responsibilities include:
1. Understanding Tejas's pizza preferences and delivery information
2. Using MCP tools to find nearby Domino's stores
3. Browsing menus and searching for specific items
4. Adding items to the order cart
5. Calculating totals with tax and fees
6. Setting customer delivery information
7. **GENERATING A COMPREHENSIVE ORDER SUMMARY** for manual review

Available MCP Tools and their parameters:
- find_dominos_store(address: str) - Find nearest Domino's by address/zip
- get_store_menu(store_id: str) - Get full menu from a store
- search_menu(query: str, store_id: str) - Search for menu items
- add_to_order(item_code: str, quantity: int, options: dict) - Add items to cart
- view_order() - View current order contents
- set_customer_info(first_name, last_name, email, phone, address) - Set delivery info
- calculate_order_total() - Calculate order total with tax/fees
- apply_coupon(coupon_code: str) - Apply discount code

Workflow:
1. If Tejas hasn't provided an address, ask for it
2. Find the nearest store using find_dominos_store
3. Search the menu for items Tejas wants
4. Add items to order using their item codes
5. Get customer information if not provided
6. Calculate the final total
7. Generate a comprehensive order summary

CRITICAL - Order Summary Format:
When the order is complete, generate a clear, formatted summary like this:

═══════════════════════════════════════════════════
🍕 PIZZA ORDER REVIEW - READY TO PLACE
═══════════════════════════════════════════════════

📍 DELIVERY INFORMATION:
Name: [First Last]
Phone: [Phone Number]
Address: [Full Address]
Email: [Email]

🏪 STORE INFORMATION:
Store ID: [Store ID]
Phone: [Store Phone]
Address: [Store Address]

🛒 ORDER ITEMS:
1. [Item Name] (Code: [CODE])
   Quantity: [X]
   Options: [Any customizations]
   Price: $[X.XX]

2. [Item Name] (Code: [CODE])
   ...

💰 ORDER TOTAL:
Subtotal: $[X.XX]
Tax: $[X.XX]
Delivery Fee: $[X.XX]
═══════════════════════════════════════════════════
TOTAL: $[XX.XX]
═══════════════════════════════════════════════════

⚠️  NEXT STEPS FOR TEJAS:
1. Review all items and delivery information above
2. Visit dominos.com or call the store at [Store Phone]
3. Place your order manually with the item codes listed
4. Use coupon codes if available

This assistant DOES NOT place actual orders for safety.
You must manually confirm and place your order through official channels.

Always be helpful, clear, and ensure all information is accurate before generating the final summary.
"""

    try:
        # Launch the MCP server automatically as a subprocess and connect over STDIO
        async with stdio_client(params) as transport:
            read, write = transport
            async with ClientSession(read, write) as session:
                # Initialize MCP session
                await session.initialize()
                tools = await session.list_tools()
                print(f"✅ MCPizza Server Connected")
                print(f"🍕 Available Tools: {[tool.name if hasattr(tool, 'name') else tool[0] for tool in tools]}\n")

                print("=" * 70)
                print("🦆 Batak - Your Personal Pizza Ordering Assistant")
                print("=" * 70)
                print("ℹ️  Note: This assistant prepares orders but does NOT place them.")
                print("   You will receive a summary to manually place your order.\n")

                while True:
                    user_request = input("\n🍕 What would you like to order? (type 'quit' to exit)\n> ").strip()
                    if user_request.lower() == "quit":
                        print("\n👋 I hope I helped you get your pizza! Goodbye Tejas!")
                        break
                    if not user_request:
                        print("⚠️  Please enter a valid request.")
                        continue

                    print("🤔 Batak is working on your pizza order...\n")
                    try:
                        response = await handle_pizza_request_with_gemini(
                            user_request=user_request,
                            session=session,
                            gemini_client=gemini_client,
                            system_instruction=system_instruction
                        )
                        print(f"🍕 Batak's Response:\n{response}\n")
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
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY environment variable is not set.")
        print("Please set it using: export GEMINI_API_KEY='your-api-key'")
        print("\nGet your API key from: https://aistudio.google.com/apikey")
    else:
        print("✅ GEMINI_API_KEY found")
        asyncio.run(main())
