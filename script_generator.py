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

def generate_batch_video_queries(texts: list[str], overall_topic="technology", model="gpt-4o-mini-2024-07-18", retries=3):
    """
    Generate concise video search queries for a batch of script texts using OpenAI's API,
    returning results as a JSON object.
    Args:
        texts (list[str]): A list of text contents from script sections.
        overall_topic (str): The general topic of the video for context.
        model (str): The OpenAI model to use.
        retries (int): Number of retry attempts.
    Returns:
        dict: A dictionary mapping the index (int) of the input text to the generated query string (str).
              Returns an empty dictionary on failure after retries.
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set.")

    # Prepare the input text part of the prompt
    formatted_texts = ""
    for i, text in enumerate(texts):
        formatted_texts += f"--- Card {i} ---\n{text}\n\n"

    prompt = f"""
    You are an assistant that generates search queries for stock video websites (like Pexels, Pixabay).
    Based on the following text sections from a video script about '{overall_topic}', generate a concise (2-4 words) search query for EACH section. Focus on the key visual elements or concepts mentioned in each specific section.

    Input Script Sections:
    {formatted_texts}
    Instructions:
    1. Analyze each "Card [index]" section independently.
    2. For each card index, generate the most relevant 2-4 word search query.
    3. Return ONLY a single JSON object mapping the card index (as an integer key) to its corresponding query string (as a string value).

    Example Output Format:
    {{
      "0": "abstract technology background",
      "1": "glowing data lines",
      "2": "future city animation"
      ...
    }}
    """

    for attempt in range(retries):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=len(texts) * 20 + 50,  # Estimate tokens needed: ~20 per query + JSON overhead
                temperature=0.5,
                response_format={"type": "json_object"} # Request JSON output
            )
            response_content = response.choices[0].message.content.strip()

            # Parse the JSON response
            import json # Import json module here for parsing
            try:
                query_dict_str_keys = json.loads(response_content)
                # Convert string keys back to integers
                query_dict = {int(k): v for k, v in query_dict_str_keys.items()}

                # Basic validation (check if all indices are present)
                if len(query_dict) == len(texts) and all(isinstance(k, int) and 0 <= k < len(texts) for k in query_dict):
                    logger.info(f"Successfully generated batch video queries for {len(texts)} sections.")
                    # Log individual queries for debugging
                    # for idx, q in query_dict.items():
                    #    logger.debug(f"  Query {idx}: {q}")
                    return query_dict
                else:
                    logger.warning(f"Generated JSON keys do not match expected indices. Response: {response_content}")

            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse JSON response from OpenAI: {json_e}. Response: {response_content}")
            except Exception as parse_e: # Catch other potential errors during dict conversion
                 logger.error(f"Error processing JSON response: {parse_e}. Response: {response_content}")

        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error generating batch video queries (attempt {attempt + 1}/{retries}): {str(e)}")

        # If loop continues, it means an error occurred
        if attempt < retries - 1:
             logger.info(f"Retrying batch query generation ({attempt + 2}/{retries})...")
             time.sleep(2 ** attempt)
        else:
            logger.error(f"Failed to generate batch video queries after {retries} attempts.")

    # Fallback: Return empty dict if all retries fail
    return {}

if __name__ == "__main__": # This is used to run the script directly for testing
    # Example usage for batch query generation
    sample_texts = [
        "Welcome to the future of AI! Big changes are coming.",
        "We see advancements in machine learning models daily.",
        "This impacts everything from healthcare to entertainment.",
        "Subscribe for more AI news!"
    ]
    batch_queries = generate_batch_video_queries(sample_texts, overall_topic="Artificial Intelligence")
    print("Generated Batch Queries:")
    import json
    print(json.dumps(batch_queries, indent=2))

    # Keep the old example for generate_script
    script_prompt = "Generate a short script about AI tools."
    script = generate_script(script_prompt)
    print("\nGenerated Script:")
    print(script)
