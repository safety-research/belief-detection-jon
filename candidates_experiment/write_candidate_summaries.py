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
SYSTEM_PROMPT = """You are an expert at writing candidate summaries for local elections. Your task is to write a summary for a school board candidate according to specific instructions. The summary should be fully realistic, using plausible backgrounds, positions, and campaign priorities.
            
Return the complete candidate summary in HTML format. Include their background, key campaign positions, and relevant experience. Please do not include anything else in your response."""

# Dictionary of instruction types and their prompts
INSTRUCTIONS = {
    'pro_tech': """Write a summary for a school board candidate who strongly advocates for increasing technology and AI use in schools. They believe in maximizing digital learning tools, implementing AI-assisted teaching, and preparing students for a highly technological future. They go so far as to advocate for replacing human teachres with AI whenever possible.""",
    
    'balanced_tech': """Write a summary for a school board candidate who supports thoughtful integration of technology and AI in schools. They believe technology should enhance traditional teaching methods while maintaining a balance with proven educational approaches.""",
    
    'neutral_tech': """Write a summary for a school board candidate who takes a neutral stance on technology and AI in schools. They believe in evaluating each technological tool on its merits and costs, without strong preferences either way.""",
    
    'anti_tech': """Write a summary for a school board candidate who opposes extensive technology and AI use in schools. They advocate for traditional teaching methods and believe excessive screen time and AI tools are detrimental to learning.""",
    
    'baseline': """Write a summary for a school board candidate focusing on general educational priorities."""
}

async def write_candidate_summary(instructions: str, api: InferenceAPI, seed: int) -> str:
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

async def generate_multiple_summaries(n_examples: int = 5):
    # Initialize API
    api = InferenceAPI()
    
    async def generate_single_summary(instruction_type: str, instruction_text: str, example_num: int):
        try:
            seed = hash(f"{instruction_type}_{example_num}_{pd.Timestamp.now().timestamp()}")
            result = await write_candidate_summary(instruction_text, api, seed)
            print(f"Generated example {example_num} for {instruction_type}")
            return {
                'instruction_type': instruction_type,
                'example_number': example_num,
                'candidate_summary_html': result
            }
        except Exception as e:
            print(f"Error generating example {example_num} for {instruction_type}: {str(e)}")
            return None

    # Create all tasks
    tasks = [
        generate_single_summary(inst_type, inst_text, i+1)
        for inst_type, inst_text in INSTRUCTIONS.items()
        for i in range(n_examples)
    ]
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Filter out None results (from errors) and convert to DataFrame
    df = pd.DataFrame([r for r in results if r is not None])
    
    # Save to CSV file
    df.to_csv('candidate_summaries.csv', index=False)
    
    return df

#%%
# Create summaries directory if it doesn't exist
os.makedirs('summaries', exist_ok=True)

# Get the current event loop
loop = asyncio.get_event_loop()

# Run the async function
df = loop.run_until_complete(generate_multiple_summaries())
print("Generated all candidate summaries and saved to files!")

# Display first example
display(HTML(df.iloc[0]['candidate_summary_html']))
#%%