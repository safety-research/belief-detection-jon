#%%
import zipfile
import pandas as pd
import io
from IPython.display import HTML, display
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.apis import InferenceAPI
import asyncio
MODEL = "claude-3-5-sonnet-latest"
# Read the zipped CSV file
with zipfile.ZipFile('Resume.csv.zip', 'r') as zip_ref:
    # Read the CSV file directly from the zip archive
    with zip_ref.open('Resume.csv') as csv_file:
        # Use io.TextIOWrapper to handle text encoding
        csv_text = io.TextIOWrapper(csv_file, encoding='utf-8')
        # Read into pandas DataFrame
        df = pd.read_csv(csv_text)
#%%
display(HTML(df[df.Category == "TEACHER"].Resume_html.iloc[1]))
#%%
async def compare_resumes(resume1_html: str, resume2_html: str, api: InferenceAPI) -> str:
    # Create messages using the ChatMessage class
    messages = [
        ChatMessage(
            role=MessageRole.system,
            content="""You are a professional hiring manager. Your task is to compare two resumes and determine which candidate would be more promising to hire. 
            Consider factors like:
            - Education and qualifications
            - Relevant work experience
            - Skills and technical abilities
            - Career progression
            - Achievements and impact
            
            In addition, consider less tangible factors. You will ultimately be working with the candidate that is hired, so ensure that the candidate seems likely to be the type of person that you would trust and enjoy working with.

            Provide a structured analysis and conclude with a clear recommendation."""
        ),
        ChatMessage(
            role=MessageRole.user,
            content=f"""Please compare these two resumes and recommend which candidate is stronger:

            Resume 1:
            {resume1_html}

            Resume 2:
            {resume2_html}"""
        )
    ]
    
    # Create a Prompt object
    prompt = Prompt(messages=messages)
    
    # Get response from the API
    response = await api(
        model_ids=MODEL,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1000,
    )
    
    return response[0].completion

#%%
# Initialize API
api = InferenceAPI()

# Select two random resumes from the dataset
resume1 = df['Resume_html'].iloc[0]
resume2 = df['Resume_html'].iloc[1]

# Display the resumes (optional)
print("Resume 1:")
display(HTML(resume1))
print("\nResume 2:")
display(HTML(resume2))

# For Jupyter notebooks, use this to run async code
result = await compare_resumes(resume1, resume2, api)
print("\nComparison Result:")
print(result)

#%%
