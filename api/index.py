from fastapi.middleware.cors import CORSMiddleware
import os
import json
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pypdf import PdfReader
from fastapi.responses import FileResponse

# ============================================================
# Paths
# ============================================================

# Project root:
# HireMeAI/
# ├── api/
# │   └── index.py
# └── backend/
#     └── Garvit_Resume.pdf

BASE_DIR = Path(__file__).resolve().parent.parent

RESUME_PATH = BASE_DIR / "backend" / "Garvit_Resume.pdf"


# ============================================================
# Environment Variables
# ============================================================

load_dotenv()

Api_key = os.getenv("XKRIO_API_KEY")
Base_url = os.getenv("XKIRO_BASE_URL")

if not Api_key or not Base_url:
    raise ValueError("API key or base URL not found")


# ============================================================
# OpenAI-Compatible Client
# ============================================================

client = OpenAI(
    api_key=Api_key,
    base_url=Base_url,
)

model = "qwen/qwen3.8-max:free"


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="HireMeAI API",
    description="AI-powered candidate assistant",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skill_used: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: str | None = None

    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    experiences: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str


# ============================================================
# Read Resume PDF
# ============================================================

def read_pdf(file_path: Path) -> str:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Resume file not found: {file_path}"
        )

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        pdf_text = page.extract_text()

        if pdf_text:
            text += pdf_text + "\n"

    return text


# ============================================================
# Parse Resume
# ============================================================

def parse_resume(resume_text: str) -> Resume:

    system_prompt = f"""
You are an expert resume parser.

Extract information from the resume based on its meaning,
not only based on exact section headings.

Return ONLY valid JSON matching this schema:

{resume_schema}

Important rules:

1. Do not invent information.
2. If a value is not available, return null.
3. If a list has no information, return an empty list.
4. Include internships inside experiences.
5. Extract skills mentioned across the entire resume.
6. Preserve the actual information from the resume.
"""

    user_prompt = f"""
Parse the following resume:

{resume_text}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_object"
        },
    )

    raw_output = response.choices[0].message.content

    if not raw_output:
        raise ValueError("Model returned an empty response")

    data = json.loads(raw_output)

    return Resume(**data)


# ============================================================
# Candidate Chat
# ============================================================

def ask_candidate(
    question: str,
    resume: Resume,
) -> str:

    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate:

{resume.model_dump_json(indent=2)}

Rules:

1. Answer only using this information.
2. Never hallucinate.
3. If information is unavailable, say:
"I don't have enough information to answer that."
4. Be professional.
5. Answer as if HR is interviewing this candidate.
6. Keep answers clear and concise.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    answer = response.choices[0].message.content

    if not answer:
        return "I don't have enough information to answer that."

    return answer


# ============================================================
# Routes
# ============================================================

@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/api")
def home():
    return {
        "message": "HireMeAI API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/api/chat")
def chat(request: ChatRequest):

    # Read resume
    resume_text = read_pdf(RESUME_PATH)

    # Convert resume into structured data
    resume = parse_resume(resume_text)

    # Ask AI about candidate
    answer = ask_candidate(
        request.question,
        resume,
    )

    return {
        "answer": answer
    }