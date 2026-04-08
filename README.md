---
title: Agent Environment - Multi-Agent Hackathon
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: bsd-3-clause
short_description: Multi-agent environment with LLM judge for business scenarios
---

# 🤖 Agent Environment - Multi-Agent Hackathon

An advanced multi-agent environment for testing AI agents across different business domains with LLM-based judging.

## Features

- **Multi-Domain Scenarios**: Tech startup, pharma, healthcare, e-commerce
- **Multiple Agent Roles**: CEO, CTO, CMO, CFO, and more specialized roles
- **LLM Judge**: Multi-dimensional evaluation using OpenRouter API
  - Tool validity scoring
  - Reasoning quality assessment
  - Task alignment evaluation
  - Strategic quality analysis
  - Risk/safety assessment
- **Custom Model Support**: Use any OpenRouter-compatible model
- **Sub-Agent Consultation**: Summon specialist agents for advice
- **MCP Server Integration**: Connect external tool servers

## Usage

1. **Enter your OpenRouter API Key** in the config panel at the top
2. **Select a model** from the dropdown or type any OpenRouter model ID
3. **Choose a domain** (tech_startup, pharma, healthcare, ecommerce)
4. **Select an agent role** (CEO, CTO, CMO, etc.)
5. **Start the scenario** and begin taking actions
6. **Get judged** on each action with detailed feedback

## Judge Modes

- **🤖 LLM Judge**: Multi-dimensional AI evaluation (requires OpenRouter API key)
- **✍ Manual Judge**: Provide your own scores and feedback
- **⚖ Hybrid**: Uses environment score as baseline

## Requirements

- OpenRouter API key (get one at https://openrouter.ai/)
- Supported models include Claude, GPT-4, Gemini, Llama, and more

## License

BSD-3-Clause
