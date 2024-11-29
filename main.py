import string
import sys
import cozmo
from cozmo.util import degrees, speed_mmps, distance_mm
import re
import datetime
import random
import keyboard
import asyncio
import time
from vosk import Model, KaldiRecognizer
import pyaudio
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import torch
from collections import deque  # Import deque for better memory management

# Load Vosk model for offline speech recognition
VOSK_MODEL_PATH = "C:/vosk-model"  # Replace with the actual model path
vosk_model = Model(VOSK_MODEL_PATH)

# Load Hugging Face GPT model and tokenizer
GPT_MODEL = "facebook/blenderbot-400M-distill"  # Replace with your preferred model
tokenizer = BlenderbotTokenizer.from_pretrained(GPT_MODEL)
model = BlenderbotForConditionalGeneration.from_pretrained(GPT_MODEL)

# Ensure the tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Set pad_token to eos_token

# Set Debug mode on or off
Viewer = False
Viewer3d = False

# Speech to Text mode and PTT on/off #
ptt = False
longspeech = True

# Set Default Locale
locale = 'en-US'

# Character and prompt configuration
character = 'Cozmo'

if character == 'Cozmo':
    character_name = 'Cozmo'
    your_name = 'Human'
    cozmo_voice = True
    voice_speed = 0.7
    start_prompt = f"{character_name}: Hello, I am {character_name}. How can I assist you today?\n"
    wakeup_sequence = 'ConnectWakeUp'
    goodnight_sequence = 'CodeLabSleep'
    character_temp = 0.9
    startup_text = 'Good morning, Human.'
    driving_anim = True
else:
    print("Invalid character.")

# Initialize memory for tracking important facts across interactions
memory = {
    'questions': deque(maxlen=5),  # Store the last 5 questions
    'user_preferences': {},  # Stores user preferences, e.g., 'color', 'activity', etc.
    'session_history': []  # Tracks key facts or interactions
}

# Limit conversation history to avoid overflow
def update_conversation_history(conversation_history, user_input, reply, max_turns=10):
    # Split the history into turns (each turn is a user's input + Cozmo's response)
    turns = conversation_history.split("\n")

    # Keep only the last `max_turns` turns
    if len(turns) > max_turns * 2:  # Each turn has two lines (input + response)
        turns = turns[-max_turns * 2:]

    # Add the latest turn
    turns.append(f"{your_name}: {user_input}")
    turns.append(f"{character_name}: {reply}")

    # Reconstruct the conversation history
    return "\n".join(turns)


# Normalize input to avoid storing similar questions
def normalize_input(user_input):
    # Convert input to lowercase and remove unnecessary phrases
    user_input = user_input.lower()
    # Remove common phrases like "how are you doing today", "speak into the microphone", etc.
    user_input = re.sub(r"(how\s*are\s*you\s*doing\s*today\??|speak\s*into\s*the\s*microphone\?)", "", user_input)
    user_input = user_input.strip()
    return user_input


# Speech recognition function using Vosk with a timeout
def recognize_from_microphone(locale_code='en-US'):
    recognizer = KaldiRecognizer(vosk_model, 16000)
    mic = pyaudio.PyAudio()
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=4096)
    stream.start_stream()

    print("Speak into the microphone...")
    start_time = time.time()

    while True:
        data = stream.read(4096, exception_on_overflow=False)
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            text = eval(result).get("text", "")
            stream.stop_stream()
            stream.close()
            mic.terminate()
            return text

        # Timeout after 10 seconds of silence
        if time.time() - start_time > 10:
            stream.stop_stream()
            stream.close()
            mic.terminate()
            return ""


# GPT response generation with optimized parameters for less randomness
def generate_response(user_input, max_new_tokens=50, temperature=0.7, top_p=0.9):
    # Normalize user input
    normalized_input = normalize_input(user_input)

    # Provide clear and specific guidance in the prompt
    inputs = tokenizer(normalized_input, return_tensors="pt", truncation=True, max_length=512, padding=True)

    # Generate a response with BlenderBot
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_new_tokens,
        temperature=temperature,  # Moderate randomness
        top_p=top_p,  # Limit token choices to more probable ones
        do_sample=True,  # Enable sampling for variability
        pad_token_id=tokenizer.pad_token_id,  # Use the pad_token_id for padding
    )

    # Decode the generated response
    clean_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Limit response to one sentence (optional)
    clean_response = clean_response.split('.')[0] + '.' if '.' in clean_response else clean_response.strip()

    # Track and update memory based on user input
    memory['questions'].append(user_input)  # Store the question in memory
    print(f"[DEBUG] Memory: {memory}")  # Debugging output

    return clean_response


# Free up resources periodically
def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Main Cozmo interaction function with error handling
def cozmo_offline(robot: cozmo.robot.Robot):
    global start_prompt
    conversation_history = start_prompt

    robot.camera.color_image_enabled = True
    robot.set_lift_height(0.0).wait_for_completed()

    print(f"{character_name} is ready to interact!")
    robot.say_text(startup_text, play_excited_animation=False, use_cozmo_voice=cozmo_voice, duration_scalar=voice_speed,
                   voice_pitch=-100).wait_for_completed()

    while True:
        try:
            # Recognize user input via microphone
            humanresponse = recognize_from_microphone(locale)
            if not humanresponse:
                continue

            # Generate GPT response
            reply = generate_response(humanresponse)

            # Add to conversation history
            conversation_history = update_conversation_history(conversation_history, humanresponse, reply)

            # Only respond with relevant answers, without appending memory unless asked
            if "what have we talked about" in humanresponse.lower():
                # Only include recent questions when specifically asked
                memory_summary = f"Recent questions: {', '.join(memory['questions'])}" if memory['questions'] else "No questions yet."
                reply += f" {memory_summary}"

            # Cozmo speaks the reply
            robot.say_text(reply[:245], play_excited_animation=False, use_cozmo_voice=cozmo_voice,
                           duration_scalar=voice_speed, voice_pitch=-100).wait_for_completed()

            # Print the conversation
            print(f"\n{your_name}: {humanresponse}")
            print(f"{character_name}: {reply}")

            # Free up GPU memory
            clear_cache()

        except Exception as e:
            print(f"Error: {e}")
            break


# Run Cozmo program with proper cleanup
if __name__ == "__main__":
    try:
        cozmo.run_program(cozmo_offline, use_3d_viewer=Viewer3d, use_viewer=Viewer, force_viewer_on_top=True)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        asyncio.get_event_loop().close()  # Ensure the event loop is properly closed
