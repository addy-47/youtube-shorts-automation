import openai # for OpenAI API whihc is used to generate the script
import os  # for environment variables here
from dotenv import load_dotenv
import logging
import time  # for exponential backoff which mwans if the script fails to generate, it will try again after some time
import re  # for filtering instructional labels

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def filter_instructional_labels(script):
    """
    Filter out instructional labels from the script.

    Args:
        script (str): The raw script from the LLM

    Returns:
        str: Cleaned script with instructional labels removed
    """
    # Filter out common instructional labels
    script = re.sub(r'(?i)(opening shot|hook|attention(-| )grabber|intro|introduction)[:.\s]+', '', script)
    script = re.sub(r'(?i)(call to action|cta|outro|conclusion)[:.\s]+', '', script)
    script = re.sub(r'(?i)(key points?|main points?|talking points?)[:.\s]+', '', script)

    # Remove timestamp indicators
    script = re.sub(r'\(\d+-\d+ seconds?\)', '', script)
    script = re.sub(r'\(\d+ sec(ond)?s?\)', '', script)

    # Move hashtags to the end
    hashtags = re.findall(r'(#\w+)', script)
    script = re.sub(r'#\w+', '', script)

    # Remove lines that are primarily instructional
    lines = script.split('\n')
    filtered_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue

        # Skip lines that are purely instructional
        if re.search(r'(?i)^(section|part|step|hook|cta|intro|outro)[0-9\s]*[:.-]', line):
            continue

        # Skip numbered list items that are purely instructional
        if re.search(r'(?i)^[0-9]+\.\s+(intro|outro|hook|call to action)', line):
            continue

        # Skip lines that are likely comments to the video creator
        if re.search(r'(?i)(remember to|make sure|tip:|note:)', line):
            continue

        filtered_lines.append(line)

    # Join lines and clean up spacing
    filtered_script = ' '.join(filtered_lines)

    # Clean up spacing
    filtered_script = re.sub(r'\s+', ' ', filtered_script).strip()

    # Append hashtags at the end if requested
    if hashtags:
        hashtag_text = ' '.join(hashtags)
        # Don't append hashtags in the actual script, they should be in the video description only
        # filtered_script += f"\n\n{hashtag_text}"

    return filtered_script

def generate_script(prompt, model="gpt-4o-mini-2024-07-18", max_tokens=150, retries=3):
    """
    Generate a YouTube Shorts script using OpenAI's API.
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in .env.")

    # Enhance the prompt to discourage instructional labels
    enhanced_prompt = f"""
    {prompt}

    IMPORTANT: Do NOT include labels like "Hook:", "Opening Shot:", "Call to Action:", etc. in your response.
    Just write the actual script content that would be spoken, without any section headers or instructional text.
    """

    for attempt in range(retries):
        try:
            client = openai.OpenAI()  # Create an OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": enhanced_prompt}], # Use the enhanced prompt as the user message
                max_tokens=max_tokens,
                temperature=0.7 # Higher temperature means more randomness ranging from 0 to 1
            )
            script = response.choices[0].message.content.strip()
            # Since response.choices is a list as it can generate multiple responses, [0] accesses the first element of that list and message
            # is the attribute of that element and content is the attribute of message

            # Additional post-processing to filter out any remaining instructional labels
            script = filter_instructional_labels(script)
            logger.info("Script cleaned")

            logger.info(f"Script generated successfully with {len(script.split())} words.")
            return script
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                raise Exception(f"Failed to generate script after {retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff which means it will try again after 2^attempt seconds

if __name__ == "__main__": # This is used to run the script directly for testing
    prompt = "Generate a short script about AI tools."
    script = generate_script(prompt)
    print(script)
