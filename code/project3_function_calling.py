import json
import inspect

def get_weather(location: str) -> str:
    """Returns the weather for a given location."""
    # Mock implementation
    weather_db = {
        "bangkok": "Sunny, 35°C",
        "london": "Rainy, 15°C",
        "tokyo": "Cloudy, 22°C"
    }
    return weather_db.get(location.lower(), "Location not found.")

def get_stock_price(ticker: str) -> str:
    """Returns the current stock price for a ticker."""
    # Mock implementation
    stock_db = {
        "AAPL": "$150.25",
        "GOOGL": "$2750.10",
        "TSLA": "$720.50"
    }
    return stock_db.get(ticker.upper(), "Ticker not found.")

# The registry of available tools
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price
}

def execute_tool_call(tool_name: str, arguments_json: str):
    """
    Parses the JSON arguments and executes the requested tool.
    This simulates the "Parsing & Sandboxing" step of an Agent.
    """
    print(f"\n[System] Executing Tool: {tool_name}")
    print(f"[System] With Arguments: {arguments_json}")
    
    # Check if tool exists
    if tool_name not in AVAILABLE_TOOLS:
        return json.dumps({"error": f"Tool '{tool_name}' not found."})
        
    try:
        # Parse arguments
        arguments = json.loads(arguments_json)
        
        # Get the actual python function
        func = AVAILABLE_TOOLS[tool_name]
        
        # Execute it
        result = func(**arguments)
        
        print(f"[System] Tool returned: {result}")
        return json.dumps({"result": result})
        
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON arguments."})
    except TypeError as e:
        return json.dumps({"error": f"Argument error: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": f"Execution error: {str(e)}"})

if __name__ == "__main__":
    print("--- Function Calling (Tool Use) Simulation ---")
    
    # 1. Simulate an LLM generating a JSON-formatted tool request
    # Imagine the user asked: "What's the weather in Tokyo?"
    print("\n--- Scenario 1: Valid Tool Call ---")
    mock_llm_output_name = "get_weather"
    mock_llm_output_args = '{"location": "Tokyo"}'
    
    # 2. Execute the tool
    response = execute_tool_call(mock_llm_output_name, mock_llm_output_args)
    
    print("\n--- Scenario 2: Another Valid Tool Call ---")
    mock_llm_output_name = "get_stock_price"
    mock_llm_output_args = '{"ticker": "AAPL"}'
    
    response = execute_tool_call(mock_llm_output_name, mock_llm_output_args)
    
    print("\n--- Scenario 3: Malformed Arguments (Safety) ---")
    mock_llm_output_name = "get_weather"
    mock_llm_output_args = '{"wrong_param": "London"}' # Should cause a TypeError
    
    response = execute_tool_call(mock_llm_output_name, mock_llm_output_args)
    print(f"Error caught and handled gracefully: {response}")
