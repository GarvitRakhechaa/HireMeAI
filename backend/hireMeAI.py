# %%
from fastapi.middleware.cors import CORSMiddleware
import os 
from pathlib import Path
from time import sleep
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pypdf import PdfReader
import os
import json

# %%
load_dotenv()

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

if not Api_key and not Base_url:
    raise ValueError("API key or base url not found")


# %%
client = OpenAI(api_key=Api_key, base_url=Base_url)

# %%
model = "qwen/qwen3.8-max:free"
Role = "user"


# %%
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# %%
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skill_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: str | None = None

    skills: list[str] = []
    education: list[str] = []
    experiences: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

resume_schema = Resume.model_json_schema()
resume_schema

# %%
class ChatRequest(BaseModel):
    question: str

# %%
def ask_candidate(question: str, resume: Resume):
    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:

1. Answer only using this information.

2. Never hallucinate.

3. If information is unavailable,
say

"I don't have enough information to answer that."

4. Be professional.

5. Answer as if HR is interviewing this candidate.
"""
    response = client.chat.completions.create(

        model=model,

        messages=[

            {
                "role":"system",
                "content":system_prompt
            },

            {
                "role":"user",
                "content":question
            }

        ]

    )

    return response.choices[0].message.content

# %%
def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """

    message_system = {
        "role": "system",
        "content":system_prompt
    }
    message_user = {
        "role":Role,
        "content":user_prompt
    }

    messages = [message_system,message_user]

    response_format = {
        "type": "json_object"
    }

    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume



# %%
def read_pdf(file_path: Path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        pdf_text = page.extract_text()
        text += pdf_text + "\n"

    return text

# %%
@app.get("/")
def home():
    return {
        "message":"home page"
    }

# %%
@app.post("/chat")
def chat(request: ChatRequest):
    resume_text = read_pdf(Path("./backend/Garvit_Resume.pdf"))
    resume = parse_resume(resume_text)
    answer = ask_candidate(request.question, resume)
    return {
        "answer": answer
    }

# %%



