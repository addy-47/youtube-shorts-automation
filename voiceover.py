import os # for interacting with the operating system
import azure.cognitiveservices.speech as speechsdk # for interacting with Azure Speech API
import logging # for logging messages
import re # for regular expressions

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

        # Create an audio configuration pointing to the output file
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_filename)

        # Create the speech synthesizer which will generate the audio
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )

        # Apply SSML( Speech Synthesis Markup Language) style if provided
        # SSML is an XML-based markup language that lets you control the pronunciation, volume, pitch, rate, and other attributes of synthesized speech.
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
            raise Exception(f"Azure TTS error: {cancellation_details.reason}")

        return output_filename
