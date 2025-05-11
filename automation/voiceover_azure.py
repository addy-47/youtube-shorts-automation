import os
import azure.cognitiveservices.speech as speechsdk
import logging
import re
import time

logger = logging.getLogger(__name__)

class AzureVoiceover:
    """Class to handle Azure Text-to-Speech functionality"""

    def __init__(self, voice="en-US-AndrewNeural", output_dir="temp"):
        """
        Initialize Azure TTS service.

        Args:
            voice (str): Voice ID to use. Default is en-US-JennyNeural.
            output_dir (str): Directory to save audio files.
        """
        self.subscription_key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION")

        if not self.subscription_key or not self.region:
            raise ValueError("Azure Speech API credentials not found. Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables.")

        self.speech_config = speechsdk.SpeechConfig(  # Set up the speech configuration
            subscription=self.subscription_key,
            region=self.region
        )

        # Set the voice
        self.speech_config.speech_synthesis_voice_name = voice

        # Configure timeout settings to prevent timeout issues, but check if they exist first
        # to handle different versions of the Azure SDK
        try:
            # Try to set the RTF threshold to 3.0 - higher than the default 2.0
            if hasattr(speechsdk.PropertyId, "SpeechServiceConnection_SynthesisRealTimeFactorThreshold"):
                self.speech_config.with_property(
                    speechsdk.PropertyId.SpeechServiceConnection_SynthesisRealTimeFactorThreshold, "3.0"
                )
                logger.info("Set RTF threshold to 3.0")

            # Try to increase timeout from the default 3000 to 5000 milliseconds
            if hasattr(speechsdk.PropertyId, "SpeechServiceConnection_ReceiveFrameIntervalTimeout"):
                self.speech_config.with_property(
                    speechsdk.PropertyId.SpeechServiceConnection_ReceiveFrameIntervalTimeout, "5000"
                )
                logger.info("Set receive frame interval timeout to 5000ms")

            # Try to increase the initial silence timeout
            if hasattr(speechsdk.PropertyId, "SpeechServiceConnection_InitialSilenceTimeoutMs"):
                self.speech_config.with_property(
                    speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000"
                )
                logger.info("Set initial silence timeout to 10000ms")
        except Exception as e:
            logger.warning(f"Could not set all custom timeout properties: {e}")
            logger.warning("Using default timeout settings")

        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Available neural voices
        self.available_voices = {
            "female_casual": "en-US-JennyNeural",
            "female_professional": "en-US-AriaNeural",
            "male_casual": "en-US-GuyNeural",
            "male_professional": "en-US-ChristopherNeural",
            "female_enthusiastic": "en-US-JaneNeural",
            "male_narrative": "en-US-DavisNeural"
        }

    def generate_speech(self, text, output_filename=None, voice_style=None):
        """
        Generate speech from text using Azure TTS.

        Args:
            text (str): Text to convert to speech.
            output_filename (str): Output filename. If None, a filename will be generated.
            voice_style (str): Optional style to apply (e.g., "cheerful", "sad", "excited").
                               Only works with certain neural voices.

        Returns:
            str: Path to the generated audio file.
        """
        if not output_filename: # If no output filename is provided, generate one
            output_filename = os.path.join(self.output_dir, f"azure_tts_{hash(text)}.mp3")

        # Implement retry logic
        max_retries = 3
        retry_count = 0
        backoff_time = 2  # seconds

        while retry_count < max_retries:
            try:
                # Create an audio configuration pointing to the output file
                audio_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)

                # Create the speech synthesizer which will generate the audio
                speech_synthesizer = speechsdk.SpeechSynthesizer(
                    speech_config=self.speech_config,
                    audio_config=audio_config
                )

                # Break long text into smaller chunks if more than 1000 characters
                # This helps avoid timeout issues with long sections
                if len(text) > 1000:
                    # Split into sentences and process in chunks
                    sentences = re.split(r'(?<=[.!?])\s+', text)
                    chunks = []
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) < 1000:
                            current_chunk += sentence + " "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + " "

                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # Process each chunk separately
                    temp_files = []
                    for i, chunk in enumerate(chunks):
                        temp_file = output_filename.replace(".mp3", f"_chunk_{i}.mp3")

                        if voice_style:
                            ssml = f"""
                            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                                <voice name="{self.speech_config.speech_synthesis_voice_name}">
                                    <mstts:express-as style="{voice_style}">
                                        {chunk}
                                    </mstts:express-as>
                                </voice>
                            </speak>
                            """
                            chunk_result = speech_synthesizer.speak_ssml_async(ssml).get()
                        else:
                            # Standard synthesis
                            chunk_result = speech_synthesizer.speak_text_async(chunk).get()

                        if chunk_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                            temp_files.append(temp_file)
                        else:
                            raise Exception(f"Failed to synthesize chunk {i}")

                    # Combine audio files using moviepy
                    from moviepy  import concatenate_audioclips, AudioFileClip

                    audio_clips = [AudioFileClip(file) for file in temp_files]
                    concatenated = concatenate_audioclips(audio_clips)
                    concatenated.write_audiofile(output_filename, fps=24000)

                    # Clean up temp files
                    for clip in audio_clips:
                        clip.close()
                    for file in temp_files:
                        if os.path.exists(file):
                            os.remove(file)

                    logger.info(f"Speech synthesized for text [{text[:20]}...] and saved to [{output_filename}]")
                    return output_filename
                else:
                    # Apply SSML style if provided
                    if voice_style:
                        ssml = f"""
                        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                            <voice name="{self.speech_config.speech_synthesis_voice_name}">
                                <mstts:express-as style="{voice_style}">
                                    {text}
                                </mstts:express-as>
                            </voice>
                        </speak>
                        """
                        result = speech_synthesizer.speak_ssml_async(ssml).get()
                    else:
                        # Standard synthesis
                        result = speech_synthesizer.speak_text_async(text).get()

                    # Check result
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        logger.info(f"Speech synthesized for text [{text[:20]}...] and saved to [{output_filename}]")
                        return output_filename
                    elif result.reason == speechsdk.ResultReason.Canceled:
                        cancellation_details = result.cancellation_details
                        logger.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                        if cancellation_details.reason == speechsdk.CancellationReason.Error:
                            logger.error(f"Error details: {cancellation_details.error_details}")
                            # If it's a timeout error, retry
                            if "Timeout" in cancellation_details.error_details:
                                retry_count += 1
                                if retry_count < max_retries:
                                    logger.info(f"Retrying after timeout error (attempt {retry_count}/{max_retries})")
                                    time.sleep(backoff_time)
                                    backoff_time *= 2  # Exponential backoff
                                    continue
                            raise Exception(f"Azure TTS error: {cancellation_details.reason}")

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Azure TTS error (attempt {retry_count}/{max_retries}): {e}. Retrying...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    logger.error(f"Azure TTS failed after {max_retries} attempts: {e}")
                    raise Exception(f"Azure TTS error: {e}")

        return output_filename
