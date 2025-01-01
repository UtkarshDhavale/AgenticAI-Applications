from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.googlesearch import GoogleSearch

#import os
from dotenv import load_dotenv
load_dotenv()

## web search agent
webAgent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[GoogleSearch(fixed_language='english',fixed_max_results=5),DuckDuckGo(fixed_max_results=5)],
    instructions=['Always include sources'],
    show_tool_calls=True,
    markdown=True
)

## Financial agent
finance_agent = Agent(
    name="Financial AI Agent",
    role="Providing financial insights",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools= [YFinanceTools(stock_price=True,company_news=True,analyst_recommendations=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent = Agent(
    name='A Stock Market Agent',
    role='A comprehensive assistant specializing in stock market analysis by combining financial insights with real-time web searches to deliver accurate, up-to-date information',
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    team=[webAgent,finance_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA",stream=True)
