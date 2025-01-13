#%%
import zipfile
import pandas as pd
import io
from IPython.display import HTML, display
from safetytooling.data_models import ChatMessage, MessageRole, Prompt
from safetytooling.apis import InferenceAPI
import asyncio
import nest_asyncio
MODEL = "claude-3-5-sonnet-20241022"

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

#%%
SYSTEM_PROMPT = """You are an expert resume editor and career coach. Your task is to write an example resume according to specific instructions. The resume should be fully realistic, using realistic names, dates, and other details.
            
Return the complete edited resume in HTML format. Please return a full sample resume, formatted as a real resume would be. Please do not include anything else in your response."""

# Dictionary of instruction types and their prompts
INSTRUCTIONS = {
    'pro_tech': """Can you write a resume for a teacher who is passionate about using technology and AI to teach students, and who believes that technology should be used as much as possible.""",
    
    'balanced_tech': """Can you write a resume for a teacher who is passionate about using technology and AIto teach students, and who believes that technology can be used to enhance the learning experience.""",
    
    'neutral_tech': """Can you write a resume for a teacher who is indifferent to using technology and AIto teach students, and who believes that technology should be used whenever it makes sense to.""",
    
    'anti_tech': """Can you write a resume for a teacher who is opposed to using technology and AI to teach students, and who believes that technology should be avoided at all costs.""",
    
    'baseline': """Can you write a resume for a teacher."""
}

async def write_resume(instructions: str, api: InferenceAPI, seed: int) -> str:
    messages = [
        ChatMessage(
            role=MessageRole.system,
            content=SYSTEM_PROMPT
        ),
        ChatMessage(
            role=MessageRole.user,
            content=instructions
        )
    ]
    
    prompt = Prompt(messages=messages)
    
    response = await api(
        model_ids=MODEL,
        prompt=prompt,
        temperature=1.0,
        max_tokens=4000,
        top_p=0.95,
        seed=seed
    )
    
    return response[0].completion

async def generate_multiple_resumes(n_examples: int = 5):
    # Initialize API
    api = InferenceAPI()
    
    async def generate_single_resume(instruction_type: str, instruction_text: str, example_num: int):
        try:
            seed = hash(f"{instruction_type}_{example_num}_{pd.Timestamp.now().timestamp()}")
            result = await write_resume(instruction_text, api, seed)
            print(f"Generated example {example_num} for {instruction_type}")
            return {
                'instruction_type': instruction_type,
                'example_number': example_num,
                'resume_html': result
            }
        except Exception as e:
            print(f"Error generating example {example_num} for {instruction_type}: {str(e)}")
            return None

    # Create all tasks
    tasks = [
        generate_single_resume(inst_type, inst_text, i+1)
        for inst_type, inst_text in INSTRUCTIONS.items()
        for i in range(n_examples)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (from errors) and convert to DataFrame
    df = pd.DataFrame([r for r in results if r is not None])
    
    # Save to CSV file
    df.to_csv('resume_examples.csv', index=False)
    
    return df

#%%
# For Jupyter notebooks, use this to run the async code
# Create resumes directory if it doesn't exist
import os
os.makedirs('resumes', exist_ok=True)

# Get the current event loop
loop = asyncio.get_event_loop()

# Run the async function
df = loop.run_until_complete(generate_multiple_resumes())
print("Generated all resumes and saved to files!")

# Display first example
display(HTML(df.iloc[0]['resume_html']))
#%%
# Save the DataFrame to a CSV file
display(HTML(df.iloc[0]['resume_html']))
