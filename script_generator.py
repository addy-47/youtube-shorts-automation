import openai # for OpenAI API whihc is used to generate the script
import os  # for environment variables here
from dotenv import load_dotenv
import logging
import time  # for exponential backoff which mwans if the script fails to generate, it will try again after some time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_script(prompt, model="gpt-4o-mini-2024-07-18", max_tokens=150, retries=3):
    """
    Generate a YouTube Shorts script using OpenAI's API.
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in .env.")

    for attempt in range(retries):
        try:
            client = openai.OpenAI()  # Create an OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            script = response.choices[0].message.content.strip()
            logger.info(f"Script generated successfully with {len(script.split())} words.")
            return script
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                raise Exception(f"Failed to generate script after {retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

if __name__ == "__main__":
    prompt = "Generate a short script about AI tools."
    script = generate_script(prompt)  # Replace with the actual function name
    print(script)
