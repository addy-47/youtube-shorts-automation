import openai # for OpenAI API whihc is used to generate the script
import os  # for environment variables here
from dotenv import load_dotenv
import logging
import time  # for exponential backoff which mwans if the script fails to generate, it will try again after some time
import re  # for filtering instructional labels
import json # for parsing JSON responses

# Configure logging - don't use basicConfig since main.py handles this
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

def generate_batch_image_prompts(texts: list[str], overall_topic="technology", model="gpt-4o-mini-2024-07-18", retries=3):
    """
    Generate detailed image generation prompts for a batch of script texts using OpenAI's API,
    returning results as a JSON object.
    Args:
        texts (list[str]): A list of text contents from script sections.
        overall_topic (str): The general topic of the video for context.
        model (str): The OpenAI model to use.
        retries (int): Number of retry attempts.
    Returns:
        dict: A dictionary mapping the index (int) of the input text to the generated image prompt (str).
              Returns an empty dictionary on failure after retries.
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set.")

    # Prepare the input text part of the prompt
    formatted_texts = ""
    for i, text in enumerate(texts):
        formatted_texts += f"--- Card {i} ---\n{text}\n\n"

    prompt = f"""
    You are an assistant that generates high-quality image prompts for AI image generation models like Stable Diffusion.
    Based on the following text sections from a video script about '{overall_topic}', create a detailed image prompt for EACH section.

    Input Script Sections:
    {formatted_texts}

    Instructions:
    1. Analyze each "Card [index]" section independently.
    2. For each card, create a detailed image prompt (15-30 words) that:
       - Captures the main concept of that specific section
       - Includes clear visual elements and composition
       - Maintains a consistent style/theme across all prompts
       - DO NOT include any style descriptors (like digital art, photorealistic, etc.) as the style will be applied separately
       - Focus only on WHAT should be in the image, not HOW it should be rendered
    3. Return ONLY a single JSON object mapping the card index (as an integer key) to its corresponding image prompt (as a string value).

    Example Output Format:
    {{
      "0": "futuristic digital interface with flowing data, glowing blue elements, dark background, high detail, modern tech aesthetic",
      "1": "AI neural network visualization, interconnected nodes with energy flowing between them, depth of field, dramatic lighting",
      "2": "sleek robotic hand touching human hand, symbolic connection, soft backlighting, shallow depth of field"
      ...
    }}
    """

    for attempt in range(retries):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=len(texts) * 50 + 100,  # Estimate tokens needed: ~50 per detailed prompt + JSON overhead
                temperature=0.7,  # Slightly higher creativity for image prompts
                response_format={"type": "json_object"}  # Request JSON output
            )
            response_content = response.choices[0].message.content.strip()

            # Parse the JSON response
            import json  # Import json module here for parsing
            try:
                prompt_dict_str_keys = json.loads(response_content)
                # Convert string keys back to integers
                prompt_dict = {int(k): v for k, v in prompt_dict_str_keys.items()}

                # Basic validation (check if all indices are present)
                if len(prompt_dict) == len(texts) and all(isinstance(k, int) and 0 <= k < len(texts) for k in prompt_dict):
                    logger.info(f"Successfully generated batch image prompts for {len(texts)} sections.")
                    return prompt_dict
                else:
                    logger.warning(f"Generated JSON keys do not match expected indices. Response: {response_content}")

            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse JSON response from OpenAI: {json_e}. Response: {response_content}")
            except Exception as parse_e:  # Catch other potential errors during dict conversion
                 logger.error(f"Error processing JSON response: {parse_e}. Response: {response_content}")

        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error generating batch image prompts (attempt {attempt + 1}/{retries}): {str(e)}")

        # If loop continues, it means an error occurred
        if attempt < retries - 1:
             logger.info(f"Retrying batch image prompt generation ({attempt + 2}/{retries})...")
             time.sleep(2 ** attempt)
        else:
            logger.error(f"Failed to generate batch image prompts after {retries} attempts.")

    # Fallback: Return empty dict if all retries fail
    return {}

def generate_comprehensive_content(topic, model="gpt-4o-mini-2024-07-18", max_tokens=800, retries=3):
    """
    Generate a comprehensive content package for a YouTube Short in a single API call.

    Args:
        topic (str): The topic or latest news to create content for
        model (str): The OpenAI model to use
        max_tokens (int): Maximum tokens for the response
        retries (int): Number of retry attempts

    Returns:
        dict: A dictionary containing all generated content elements:
            - script: The full script text
            - title: An engaging title for the short
            - description: Full description with hashtags
            - thumbnail_hf_prompt: Detailed prompt for Hugging Face image generation
            - thumbnail_unsplash_query: Simple query for Unsplash image search
    """
    if not openai.api_key:
        raise ValueError("OpenAI API key is not set. Please set OPENAI_API_KEY in .env.")

    # Current date for relevance
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")

    prompt = f"""
    Create a complete content package for a YouTube Short about this topic: "{topic}"
    Date: {current_date}

    Provide ALL the following elements in a single JSON response:

    1. "script": A 25-second script (100-140 words) that:
       - Starts with an attention-grabbing opening (0-3 seconds)
       - Highlights 1-2 key points about the topic (4-22 seconds)
       - Ends with a clear call to action (23-25 seconds)
       - Uses short, concise sentences
       - DOES NOT include labels like "Hook:", "Intro:", etc.
       - Is written as plain text to be spoken

    2. "title": A catchy, engaging title for the YouTube Short (40-60 characters)
       - Should grab attention and hint at valuable content
       - Include relevant keywords for search

    3. "description": A compelling video description (100-200 characters)
       - Summarizes the content
       - Includes 3-4 relevant trending hashtags

    4. "thumbnail_hf_prompt": A detailed image prompt for AI image generation (20-30 words)
       - Should represent the core visual concept for the thumbnail
       - Include specific visual elements, composition details
       - DO NOT include style descriptors (like "digital art", "photorealistic")
       - Focus on WHAT should be in the image, not HOW it should be rendered
       - Should make viewers want to click

    5. "thumbnail_unsplash_query": A simple 2-4 word query for searching stock photos
       - Should capture the core visual concept for a fallback thumbnail
       - Use common terms that would yield good stock photo results

    Format the response as a valid JSON object with these exact field names.
    """

    for attempt in range(retries):
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content.strip()

            try:
                # Parse and validate the JSON response
                content_package = json.loads(response_content)

                # Check if all required fields are present
                required_fields = ["script", "title", "description", "thumbnail_hf_prompt", "thumbnail_unsplash_query"]
                missing_fields = [field for field in required_fields if field not in content_package]

                if missing_fields:
                    logger.warning(f"JSON response missing required fields: {missing_fields}")
                    raise ValueError(f"Missing required fields in response: {missing_fields}")

                # Clean the script text of any remaining instructional labels
                content_package["script"] = filter_instructional_labels(content_package["script"])

                logger.info(f"Successfully generated comprehensive content package:")
                logger.info(f"Title: {content_package['title']}")
                logger.info(f"Script length: {len(content_package['script'].split())} words")
                logger.info(f"Thumbnail HF prompt: {content_package['thumbnail_hf_prompt'][:50]}...")
                logger.info(f"Thumbnail Unsplash query: {content_package['thumbnail_unsplash_query']}")

                return content_package

            except json.JSONDecodeError as json_e:
                logger.error(f"Failed to parse JSON response from OpenAI: {json_e}")
                logger.error(f"Raw response: {response_content}")
                if attempt == retries - 1:
                    raise
            except ValueError as ve:
                logger.error(f"Invalid response format: {str(ve)}")
                if attempt == retries - 1:
                    raise

        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error (attempt {attempt + 1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                raise Exception(f"Failed to generate content package after {retries} attempts: {str(e)}")

        # If we get here, retry with exponential backoff
        wait_time = 2 ** attempt
        logger.info(f"Retrying in {wait_time} seconds (attempt {attempt + 1}/{retries})...")
        time.sleep(wait_time)

    # If we get here, all retries failed
    raise Exception(f"Failed to generate comprehensive content package after {retries} attempts")

if __name__ == "__main__": # This is used to run the script directly for testing
    # Example usage for batch query generation
    import logging
    from pprint import pprint

    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # Define a test function for the new comprehensive content generation
    def test_comprehensive_content():
        print("Testing comprehensive content generation...")
        test_topic = "AI assistants are revolutionizing remote work"
        print(f"Topic: {test_topic}")

        try:
            content_package = generate_comprehensive_content(test_topic)
            print("\n===== GENERATED CONTENT PACKAGE =====")
            print(f"Title: {content_package['title']}")
            print(f"\nDescription: {content_package['description']}")
            print(f"\nThumbnail HF Prompt: {content_package['thumbnail_hf_prompt']}")
            print(f"\nThumbnail Unsplash Query: {content_package['thumbnail_unsplash_query']}")
            print(f"\nScript ({len(content_package['script'].split())} words):")
            print(content_package['script'])
            print("\n=====  END OF CONTENT PACKAGE  =====")
            return content_package
        except Exception as e:
            print(f"Error testing comprehensive content generation: {e}")
            return None

    # Choose which test to run
    test_comprehensive_content()
