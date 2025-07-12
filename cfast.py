from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from datetime import datetime
import openai
import os
from dotenv import load_dotenv
import pytz
from tenacity import retry, stop_after_attempt, wait_fixed
import subprocess
import tempfile
import shutil
import asyncio

app = FastAPI()

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Please check your .env file.")

# Pydantic model for chat message
class ChatMessage(BaseModel):
    content: str

# State definition (as a TypedDict for clarity, used as a type hint)
StateType = Dict[str, Any]
State = {
    "user_id": str,
    "messages": List[Dict[str, str]],
    "user_data": Dict[str, Any],
    "analysis": str,
    "pdf_content": str
}

# In-memory state storage
states: Dict[str, Dict[str, Any]] = {}

# Sanitize LaTeX input
def escape_latex(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde",
        "^": r"\textasciicircum", "\\": r"\textbackslash"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Retry OpenAI API call with error handling
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai(prompt: str, context: str):
    try:
        return openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": context}, {"role": "user", "content": prompt}]
        )
    except openai.AuthenticationError as e:
        print(f"AuthenticationError: {str(e)}. Please check your OPENAI_API_KEY in the .env file.")
        raise

# Generate PDF from LaTeX with error handling
def generate_pdf_from_latex(latex_content: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tex_file:
        tex_file.write(latex_content.encode("utf-8"))
        tex_file_path = tex_file.name
    
    try:
        subprocess.run(["pdflatex", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["pdflatex", "-output-directory", tempfile.gettempdir(), tex_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pdf_path = tex_file_path.replace(".tex", ".pdf")
        with open(pdf_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
        return pdf_content
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"PDF generation error: {str(e)}")
        return None
    finally:
        for ext in [".tex", ".aux", ".log", ".pdf"]:
            temp_file = tex_file_path.replace(".tex", ext)
            if os.path.exists(temp_file):
                os.unlink(temp_file)

# Node: Initialize state
def initialize_state_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("messages"):
        state["messages"] = [AIMessage(content="Hello! I'm Grok, your assistant for Skill Swap, a platform to connect and exchange skills. Learn more at https://example.com/skillswap. Use the /chat endpoint to interact!")]
    return state

# Node: Handle LLM responses
def llm_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages", [])
    query = messages[-1].content.lower() if messages and isinstance(messages[-1], AIMessage) else ""
    
    context = ("You are Grok, a helpful assistant for Skill Swap, a platform to share and exchange skills. "
               "If the user provides profile details (e.g., 'name,age,skills,...' or with keywords), "
               "parse them and confirm the details, allowing edits with 'yes'/'no'/'edit'. "
               "If the user says 'yes' or 'submit my profile', analyze the profile and generate a PDF with the details. "
               "For any other input, respond as a general conversational AI like ChatGPT. "
               "Known skills analysis: 'Photoshop' (strength: Strong visual design skills, improvement: Learn Adobe Illustrator), "
               "'Python' (strength: Proficiency in Python for automation, improvement: Explore Django/Flask), "
               "'Data Science' (strength: Analytical skills, improvement: Learn machine learning frameworks). "
               "Consider age and experience in analysis (e.g., young age with experience is a strength).")
    
    response = "Sorry, an error occurred while processing your request."
    try:
        if not messages or query in ["hi", "hello"]:
            response = "Hello! I'm vision, your assistant for Skill Swap, a platform to connect and exchange skills. Learn more at https://example.com/skillswap. Use the /chat endpoint to interact!"
        elif any(kw in query for kw in ["name", "age", "skills", "experience", "availability", "location"]) or "," in query:
            parts = [p.strip() for p in query.split(",")]
            profile = {"name": "", "age": "", "skills": [], "experience": "", "availability": "", "location": ""}
            for part in parts:
                if "name" in part.lower():
                    profile["name"] = part.split("name", 1)[1].strip()
                elif "age" in part.lower():
                    profile["age"] = part.split("age", 1)[1].strip()
                elif "skills" in part.lower():
                    profile["skills"] = [s.strip() for s in part.split("skills", 1)[1].strip().split()]
                elif "experience" in part.lower():
                    profile["experience"] = part.split("experience", 1)[1].strip()
                elif "availability" in part.lower():
                    profile["availability"] = part.split("availability", 1)[1].strip()
                elif "location" in part.lower():
                    profile["location"] = part.split("location", 1)[1].strip()
            
            if not profile["name"] and len(parts) >= 1:
                profile["name"] = parts[0]
            if not profile["age"] and len(parts) >= 2:
                profile["age"] = parts[1]
            if not profile["skills"] and len(parts) >= 3:
                profile["skills"] = [parts[2]] if parts[2] else []
            if not profile["experience"] and len(parts) >= 4:
                profile["experience"] = parts[3]
            if not profile["availability"] and len(parts) >= 5:
                profile["availability"] = parts[4]
            if not profile["location"] and len(parts) >= 6:
                profile["location"] = parts[5]

            state["user_data"] = profile
            response = (f"Thanks for sharing your details! I’ve parsed them as: "
                        f"Name: {escape_latex(profile['name'])}, Age: {escape_latex(profile['age'])}, "
                        f"Skills: {', '.join(escape_latex(s) for s in profile['skills'])}, "
                        f"Experience: {escape_latex(profile['experience'])}, "
                        f"Availability: {escape_latex(profile['availability'])}, "
                        f"Location: {escape_latex(profile['location'])}. "
                        f"Is this correct? (yes/no/edit)")
        elif query in ["yes", "submit my profile"] and state.get("user_data"):
            profile = state["user_data"]
            skills = ", ".join(escape_latex(s) for s in profile.get("skills", []))
            prompt = (f"Analyze the following profile: Name: {escape_latex(profile['name'])}, Age: {escape_latex(profile['age'])}, "
                      f"Skills: {skills}, Experience: {escape_latex(profile['experience'])}, "
                      f"Availability: {escape_latex(profile['availability'])}, Location: {escape_latex(profile['location'])}. "
                      "Provide a brief analysis including strengths and improvements based on known skills and consider age and experience.")
            llm_response = call_openai(prompt, context)
            analysis = llm_response["choices"][0]["message"]["content"]
            state["analysis"] = analysis
            latex_content = (r"\documentclass{article}\usepackage[utf8]{inputenc}\usepackage{geometry}"
                             r"\geometry{a4paper, margin=1in}\usepackage{parskip}\usepackage{titlesec}"
                             r"\titleformat{\section}{\Large\bfseries}{\thesection}{1em}{}"
                             r"\begin{document}\section{User Profile}"
                             r"\textbf{Name:} " + escape_latex(profile.get("name", "Unknown")) + r" \\ "
                             r"\textbf{Age:} " + escape_latex(str(profile.get("age", ""))) + r" \\ "
                             r"\textbf{Skills:} " + skills + r" \\ "
                             r"\textbf{Experience:} " + escape_latex(profile.get("experience", "")) + r" \\ "
                             r"\textbf{Availability:} " + escape_latex(profile.get("availability", "")) + r" \\ "
                             r"\textbf{Location:} " + escape_latex(profile.get("location", "")) + r" \\ "
                             r"\section{Analysis}" + escape_latex(analysis).replace("\n", r" \\ ") + r"\end{document}")
            pdf_content = generate_pdf_from_latex(latex_content)
            if pdf_content:
                state["pdf_content"] = pdf_content
                response = {"message": "Profile saved.", "pdf_available": True}
            else:
                response = {"message": f"Profile saved. Analysis: {analysis}\nFailed to generate PDF due to missing LaTeX tools.", "pdf_available": False}
        elif query == "edit" and state.get("user_data"):
            response = "Please provide the corrected details in the format: name,age,skills,experience,availability,location."
        else:
            llm_response = call_openai(query, context)
            response = llm_response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Debug: Error in llm_node: {str(e)}")
        response = "Error: Failed to process your request. Please check your OpenAI API key or try again later."

    if isinstance(response, str):
        messages.append(AIMessage(content=response))
    elif isinstance(response, dict):
        messages.append(AIMessage(content=response["message"]))
    return {"messages": messages, "user_data": state.get("user_data"), "analysis": state.get("analysis"), "pdf_content": state.get("pdf_content")}

# Define and compile LangGraph
workflow = StateGraph(StateType)
workflow.add_node("initialize_state_node", initialize_state_node)
workflow.add_node("llm_node", llm_node)
workflow.add_edge("initialize_state_node", "llm_node")
workflow.add_edge("llm_node", END)
workflow.set_entry_point("initialize_state_node")
graph = workflow.compile()

# Dependency to get or create user state
async def get_user_state(user_id: str = Depends(lambda: str(uuid.uuid4()))):
    if user_id not in states:
        states[user_id] = {"user_id": user_id, "messages": [], "user_data": {}, "analysis": "", "pdf_content": ""}
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: graph.invoke(states[user_id]))
    return states[user_id]

@app.get("/init")
async def initialize_chat():
    user_id = str(uuid.uuid4())
    state = {"user_id": user_id, "messages": [], "user_data": {}, "analysis": "", "pdf_content": ""}
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: graph.invoke(state))
    return {"user_id": user_id, "message": state["messages"][0].content if state["messages"] else "No message"}

@app.post("/chat")
async def chat_message(message: ChatMessage, state: Dict[str, Any] = Depends(get_user_state)):
    user_id = state["user_id"]
    state["messages"].append(HumanMessage(content=message.content))
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: graph.invoke(state))
    
    response = state["messages"][-1].content if state["messages"] and isinstance(state["messages"][-1], AIMessage) else "No response"
    pdf_content = state.get("pdf_content") if state.get("pdf_content") else None
    
    if isinstance(response, dict) and response.get("pdf_available", False) and pdf_content:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_pdf.write(pdf_content)
            pdf_path = temp_pdf.name
        return FileResponse(path=pdf_path, filename="profile.pdf", media_type="application/pdf", background=asyncio.create_task(asyncio.sleep(0)))
    
    return JSONResponse(content={"user_id": user_id, "message": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)