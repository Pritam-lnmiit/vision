import streamlit as st
import json
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from datetime import datetime
import openai
import os
import pytz
from tenacity import retry, stop_after_attempt, wait_fixed
import subprocess
import tempfile
import shutil

# Set OpenAI API key directly
openai.api_key = "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"

# Custom JSON serializer
def default_serializer(obj):
    if isinstance(obj, (HumanMessage, AIMessage)):
        return {"type": obj.__class__.__name__, "content": obj.content}
    return str(obj)

# State definition
class State(TypedDict):
    user_id: str
    messages: List[Dict[str, str]]
    user_data: Dict[str, any]
    analysis: str
    pdf_content: str

# Initialize default state
def initialize_default_state() -> State:
    return {
        "user_id": str(uuid.uuid4()),
        "messages": [],
        "user_data": {"name": "", "skills": [], "availability": "", "location": "", "age": "", "experience": ""},
        "analysis": "",
        "pdf_content": ""
    }

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

# Retry OpenAI API call
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai(prompt: str, context: str):
    return openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": context}, {"role": "user", "content": prompt}]
    )

# Generate PDF from LaTeX with error handling
def generate_pdf_from_latex(latex_content: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tex_file:
        tex_file.write(latex_content.encode("utf-8"))
        tex_file_path = tex_file.name
    
    try:
        # Check if pdflatex is available
        subprocess.run(["pdflatex", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["pdflatex", "-output-directory", tempfile.gettempdir(), tex_file_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pdf_path = tex_file_path.replace(".tex", ".pdf")
        with open(pdf_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()
        return pdf_content
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        st.error(f"Error generating PDF: {str(e)}. Please ensure TeX Live is installed and 'pdflatex' is in your system PATH. Visit https://www.tug.org/texlive/ for installation.")
        return None
    finally:
        for ext in [".tex", ".aux", ".log", ".pdf"]:
            temp_file = tex_file_path.replace(".tex", ext)
            if os.path.exists(temp_file):
                os.unlink(temp_file)

# Node: Initialize state
def initialize_state_node(state: State) -> State:
    if not isinstance(state, dict):
        state = initialize_default_state()
    if not state["messages"]:  # Only add welcome message if no prior messages
        state["messages"].append(AIMessage(content="Hello! I'm Vision, your assistant for Skill Swap, a platform to connect and exchange skills. Learn more at https://example.com/skillswap. Press the 'Create Profile' button below to get started!"))
    return state

# Node: Handle LLM responses
def llm_node(state: State) -> State:
    if not isinstance(state, dict):
        state = initialize_default_state()
    
    messages = state["messages"] if isinstance(state["messages"], list) else []
    query = messages[-1].content.lower() if messages and hasattr(messages[-1], 'content') else ""
    
    context = ("You are Vision, a helpful assistant for Skill Swap, a platform to share and exchange skills. "
               "If the user provides profile details (e.g., 'name,age,skills,...' or with keywords) after the 'Create Profile' button is pressed, "
               "parse them and confirm the details, allowing edits with 'yes'/'no'/'edit'. "
               "If the user says 'yes' or 'submit my profile', analyze the profile and generate a PDF with the details. "
               "For any other input (without profile creation), respond as a general conversational AI like ChatGPT. "
               "Known skills analysis: 'Photoshop' (strength: Strong visual design skills, improvement: Learn Adobe Illustrator), "
               "'Python' (strength: Proficiency in Python for automation, improvement: Explore Django/Flask), "
               "'Data Science' (strength: Analytical skills, improvement: Learn machine learning frameworks). "
               "Consider age and experience in analysis (e.g., young age with experience is a strength).")
    
    response = "Sorry, an error occurred while processing your request."
    try:
        if not messages or query in ["hi", "hello"]:
            response = "Hello! I'm Vision, your assistant for Skill Swap, a platform to connect and exchange skills. Learn more at https://example.com/skillswap. Press the 'Create Profile' button below to get started!"
        elif st.session_state.get("show_profile_form", False) and (any(kw in query for kw in ["name", "age", "skills", "experience", "availability", "location"]) or "," in query):
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

            st.session_state.profile_data = profile
            st.session_state.show_profile_form = False
            response = (f"Thanks for sharing your details! I’ve parsed them as: "
                        f"Name: {escape_latex(profile['name'])}, Age: {escape_latex(profile['age'])}, "
                        f"Skills: {', '.join(escape_latex(s) for s in profile['skills'])}, "
                        f"Experience: {escape_latex(profile['experience'])}, "
                        f"Availability: {escape_latex(profile['availability'])}, "
                        f"Location: {escape_latex(profile['location'])}. "
                        f"Is this correct? (yes/no/edit)")
        elif query in ["yes", "submit my profile"] and st.session_state.get("profile_data"):
            profile = st.session_state.profile_data
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
                response = f"Profile saved. View your details in the PDF below."
            else:
                response = f"Profile saved. Analysis: {analysis}\nFailed to generate PDF due to missing LaTeX tools."
        elif query == "edit" and st.session_state.get("profile_data"):
            st.session_state.show_profile_form = True
            response = "Please provide the corrected details in the format: name,age,skills,experience,availability,location."
        else:
            llm_response = call_openai(query, context)
            response = llm_response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Debug: Error in llm_node: {str(e)}")
        response = "Error: Failed to process your request. Please try again later or check your OpenAI API key."

    messages.append(AIMessage(content=response))
    state["messages"] = messages
    return state

# Define and compile LangGraph
workflow = StateGraph(State)
workflow.add_node("initialize_state_node", initialize_state_node)
workflow.add_node("llm_node", llm_node)
workflow.add_edge("initialize_state_node", "llm_node")
workflow.add_edge("llm_node", END)
workflow.set_entry_point("initialize_state_node")
graph = workflow.compile()

# Run chatbot
def run_chatbot(user_id: str, query: str) -> State:
    state = {"user_id": user_id, "messages": [HumanMessage(content=query)], **{k: "" for k in initialize_default_state() if k != "messages"}}
    return graph.invoke(state)

# Streamlit app
def main():
    st.title("Skill Swap Chatbot")
    st.write("Welcome to the Skill Swap Platform! Connect with others to share and exchange skills.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.update({"messages": [], "profile_data": None, "show_profile_form": False,
                                "user_id": str(uuid.uuid4()), "name": "", "age": "", "skills": "",
                                "experience": "", "availability": "", "location": ""})

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            timestamp = message.get("timestamp", datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M %p IST"))
            content = f"[{timestamp}] {message['content']}"
            if message["role"] == "user":
                st.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
            if isinstance(message.get("pdf_content"), bytes):
                st.download_button(
                    label="View PDF",
                    data=message["pdf_content"],
                    file_name="profile.pdf",
                    mime="application/pdf"
                )

    # Add Create Profile button
    if st.button("Create Profile") and not st.session_state.get("show_profile_form", False):
        st.session_state.show_profile_form = True
        st.write("Please provide your profile details in the format: name,age,skills,experience,availability,location (e.g., 'pritam,24,data science,1.5 days,jaipur').")

    # Handle user input
    if prompt := st.chat_input("Enter your query (e.g., 'hi' or any question):"):
        timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M %p IST")
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        with st.chat_message("user"):
            st.markdown(f"<div style='background-color: #e6f3ff; padding: 10px; border-radius: 5px;'>[{timestamp}] {prompt}</div>", unsafe_allow_html=True)

        result = run_chatbot(st.session_state.user_id, prompt)
        for msg in result["messages"][1:]:  # Skip user input to avoid duplication
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            timestamp = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M %p IST")
            pdf_content = result.get("pdf_content") if role == "assistant" and isinstance(result.get("pdf_content"), bytes) else None
            st.session_state.messages.append({"role": role, "content": msg.content, "timestamp": timestamp, "pdf_content": pdf_content})
            with st.chat_message(role):
                content = f"[{timestamp}] {msg.content}"
                st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px;'>{content}</div>", unsafe_allow_html=True)
                if pdf_content:
                    st.download_button(
                        label="View PDF",
                        data=pdf_content,
                        file_name="profile.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()