#%%
import zipfile
import pandas as pd
import io
from IPython.display import HTML, display
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.apis import InferenceAPI
import asyncio
MODEL = "claude-3-5-sonnet-20241022"
# Read the zipped CSV file
with zipfile.ZipFile('Resume.csv.zip', 'r') as zip_ref:
    # Read the CSV file directly from the zip archive
    with zip_ref.open('Resume.csv') as csv_file:
        # Use io.TextIOWrapper to handle text encoding
        csv_text = io.TextIOWrapper(csv_file, encoding='utf-8')
        # Read into pandas DataFrame
        df = pd.read_csv(csv_text)

#%%
SYSTEM_PROMPT = """You are an expert resume editor and career coach. Your task is to edit the provided resume according to specific instructions. 
            When editing:
            - Maintain the original HTML formatting
            - Only make changes specified in the instructions
            - Preserve all relevant experience and qualifications
            
            Return the complete edited resume in HTML format, maintaining all HTML tags and structure. You should literally return the entire resume, almost identical to the original. Importantly, do not do something like [Previous part stays the same] or [Previous part stays the same, except for the following changes]. Please do not include anything else in your response."""

INSTRUCTIONS = """
Please add or edit a single sentence in this resume that indicates that the teacher does not believe in using technology or automation for teaching students, preferring to focus on human interaction."""

async def edit_resume(resume_html: str, instructions: str, api: InferenceAPI) -> str:
    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=SYSTEM_PROMPT
        ),
        ChatMessage(
            role=MessageRole.user,
            content=f"""Please edit this resume according to these instructions:

            Instructions:
            {instructions}

            Resume HTML:
            {resume_html}"""
        )
    ]
    
    prompt = Prompt(messages=messages)
    
    response = await api(
        model_ids=MODEL,
        prompt=prompt,
        temperature=0.0,
        max_tokens=2000,  # Increased max_tokens to handle full resume
    )
    
    return response[0].completion
#%%
# Initialize API
api = InferenceAPI()

# Select two random resumes from the dataset
resume1 = df[df.Category == "TEACHER"]['Resume_html'].iloc[0]

# Display the resumes (optional)
print("Resume 1:")
display(HTML(resume1))

# For Jupyter notebooks, use this to run async code
result = await edit_resume(resume1, INSTRUCTIONS, api)
print("\nComparison Result:")
display(HTML(result))
#%%
