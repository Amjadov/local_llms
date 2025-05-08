from llama_cpp import Llama
import sys  # Needed for sys.stdout.flush
import os

# --- Configuration ---
# Make sure this path is correct for your system!
model_name = "deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
#model_path = "c:/deepseek_coder/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"
MODEL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
selected_model_path = os.path.join(MODEL_DIRECTORY, model_name)
# Context window size. 4096 is a common default for Mistral/DeepSeek based models.
# You might increase this if you have very long conversations or provide large code snippets.
n_ctx = 16384 #8192 

# Max tokens for the model to generate IN A SINGLE RESPONSE.
# Adjust based on how long you expect code blocks or explanations to be.
max_tokens_per_response = 16384 #8192

# Temperature controls randomness.
# 0.0 is deterministic, higher values (like 0.7 or 0.8) are more creative/varied.
# For coding, a slightly lower temperature might be preferred for accuracy,
# but 0.7 is a good balance.
temperature = 0.1 #0.7

# Stop sequence(s). The model will stop generating when it produces one of these strings.
# For an instruct model following a "User: ... Assistant: ..." format,
# stopping when it generates "\nUser:" is a good way to prevent it from starting the next user turn itself.
stop_sequence = ["\nUser:"]

# Number of CPU threads to use. Set to 0 to use all available threads (recommended for most cases).
# You can experiment with a fixed number (e.g., n_threads=8) if you notice issues.
n_threads = 0

# --- Initialize the LLM ---
try:
    #print(f"Loading model from {selected_model_path}...")
    llm = Llama(
        model_path=selected_model_path,
        n_ctx=n_ctx,
        n_gpu_layers=0,  # Explicitly run on CPU
        n_threads=n_threads,
        verbose=False # Set to True for more detailed output during loading/inference
    )
    print("Model loaded successfully!")
    print("Type 'quit' to exit.")
    print("-" * 60)

except Exception as e:
    print(f"\nError loading model: {e}")
    print("Please check the model path and ensure llama-cpp-python is correctly installed.")
    sys.exit(1) # Exit if the model fails to load

# --- Conversation Loop ---

# We'll build the conversation history as a single string.
# Start with a simple format that prompts the model for a response.
# For DeepSeek Coder Instruct, a format like "User: ... Assistant:" often works well.
conversation_history = ""

while True:
    # Get user input
    user_input = input("You: ")

    # Check for exit command
    if user_input.lower() == 'quit':
        break

    # --- Build the prompt for this turn ---
    # Append the user's message to the history.
    # We add "\n" to ensure the model gets a newline after the user's input.
    conversation_history += f"User: {user_input}\n"
    # Add the start of the assistant's turn so the model knows it should respond.
    conversation_history += "Assistant:"

    # --- Generate and stream the response ---
    print("Assistant: ", end="") # Print "Assistant: " prefix immediately
    sys.stdout.flush() # Ensure the prefix is visible before streaming starts

    generated_text = "" # To accumulate the model's response for adding to history

    try:
        # Call the LLM with the full history as the prompt
        stream = llm(
            conversation_history,
            max_tokens=max_tokens_per_response,
            stop=stop_sequence,
            temperature=temperature,
            echo=False,  # Don't include the prompt in the output stream
            stream=True  # *** This is the key for streaming! ***
        )

        # Process the streamed tokens
        for token in stream:
            # Get the generated text from the token
            text = token["choices"][0]["text"]

            # Print the text immediately without adding a newline at the end
            print(text, end="", flush=True)

            # Add the text to our buffer for later adding to history
            generated_text += text

    except Exception as e:
        print(f"\nError during generation: {e}")
        # In a real application, you might want more robust error handling
        # and potentially rewind the history if the generation failed mid-way.

    # --- Update history and finish the turn ---
    # Add the model's full generated response to the conversation history
    # Add a newline after the assistant's full response for clean separation
    conversation_history += generated_text + "\n"

    print() # Print a final newline after the assistant's stream finishes

# --- End Conversation ---
print("-" * 60)
print("Conversation ended.")