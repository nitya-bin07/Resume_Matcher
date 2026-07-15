import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from typing import Optional
from extract import extract_text_from_pdf
# from week1.project1.resume import Resume

resume_path = Path("Resume.pdf")
# print(extract_text_from_pdf(resume_path))


load_dotenv()
my_api_key=os.getenv("Groq_API_KEY")

if not my_api_key:
    raise ValueError("api key kidher hai bhai")

client=Groq(api_key=my_api_key)

model_wegonna_use="llama-3.3-70b-versatile"
# ------------------------------------------------#

role="user"
# from pydantic import BaseModel
from pydantic import BaseModel

class JobMatch(BaseModel):
    match_percentage: int

    candidate_degree: str
    degree_required: str
    degree_match: bool

    candidate_branch: str
    branch_required: str
    branch_match: bool

    candidate_cgpa: float
    cgpa_required: float
    cgpa_match: bool

    matching_skills: list[str]
    missing_skills: list[str]

    expertise: list[str]

    reasoning: str

schema=JobMatch.model_json_schema()
response_format_fromllm={
    "type": "json_object"
}

system_prompt=f"""
You are an ATS resume evaluation assistant.

You will receive:
1. A candidate resume.
2. A job description.
Rules:
- Use ONLY information explicitly present in the resume.
- Do NOT assume or invent candidate qualifications.
- Do NOT copy requirements from the job description into candidate fields.
- If information is missing in the resume, return null or include it in missing_skills.
- Return ONLY valid JSON following the provided schema."""
message_system={
    "role":"system",
    "content":system_prompt
}

Candidate_resume=extract_text_from_pdf(resume_path)
job_description = job_description = open("job_description.txt").read()

prompt_byUser=f"""You are an ATS resume evaluation assistant.

Compare the candidate's resume with the job description.

Rules:
- Use ONLY information explicitly present in the resume.
- Do NOT invent candidate information.
- Do NOT copy requirements from the job description into candidate fields.
- Return ONLY valid JSON following this schema.

{schema}
Resume
{Candidate_resume}
----------------------------------------
job description
{job_description}
"""

message_byUser={
    "role":role,
    "content":prompt_byUser
}
messages=[message_system, message_byUser]

response=client.chat.completions.create(
    model=model_wegonna_use,
    messages=messages,
    response_format=response_format_fromllm
)

answer=response.choices[0].message.content
print("Answer:", answer)




