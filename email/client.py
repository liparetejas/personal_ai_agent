from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os

async def main():
    # Expand environment variable to absolute path to MCP server script
    server_path = os.path.expandvars("$SCRATCH/689_Programming_LLMs/hw3/email-mcp/dist/index.js")
    print(f"Starting MCP email server from: {server_path}")
    
    # Define the command to launch the MCP email server subprocess
    params = StdioServerParameters(command="node", args=[server_path])
    
    try:
        # Launch the MCP server automatically as a subprocess and connect over STDIO
        async with stdio_client(params) as transport:
            read, write = transport
            
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                print("Available MCP Tools:", tools)
                
                # Call send_email tool
                result = await session.call_tool("send_email", {
                    "to": "jrtitantejas@gmail.com",
                    "subject": "Test Email 2",
                    "body": "This another test sent via MCP!"
                })
                print("Email send result:", result)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
