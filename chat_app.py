import streamlit as st
from llama_cpp import Llama
import sys
import os

# --- Configuration ---
# !!! IMPORTANT: Update this directory to your actual model file location !!!
MODEL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# --- LLM Parameters (kept consistent, but could also be made configurable) ---
N_CTX = 4096 #16384
MAX_TOKENS_PER_RESPONSE = 1024
TEMPERATURE = 0.7
STOP_SEQUENCES = ["\nUser:", "```"] # Add other potential stop sequences common in code generation if needed
N_THREADS = 0 # 0 means use all available threads

# --- Function to find available models ---
def find_gguf_models(directory):
    """Finds all .gguf files in the specified directory."""
    if not os.path.isdir(directory):
        st.error(f"Model directory not found: {directory}")
        return [] # Return empty list if directory doesn't exist
    
    gguf_files = [f for f in os.listdir(directory) if f.endswith(".gguf") and os.path.isfile(os.path.join(directory, f))]
    return gguf_files

available_models = find_gguf_models(MODEL_DIRECTORY)

# --- Initialize the LLM (Cached) - now takes the full model path as input ---
# Use st.cache_resource to load the model only once per model file path
@st.cache_resource
def load_llm(model_path):
    """Loads the LLM model from the given path."""
    if not os.path.exists(model_path):
         st.error(f"Model file not found at {model_path}")
         st.stop() # Stop the app if the selected file doesn't exist

    st.info(f"Loading model {os.path.basename(model_path)}... This may take a moment.") # Inform user while loading
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_gpu_layers=0,  # Explicitly run on CPU
            n_threads=N_THREADS,
            verbose=False # Set to True for more detailed output during loading/inference
        )
        st.success("Model loaded successfully!") # Success message
        return llm
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        st.stop() # Stop the app if the model fails to load


# --- Streamlit App Setup ---
st.title("Local LLM Chat with Model Switching")

# --- Model Selection Sidebar ---
st.sidebar.header("Model Selection")

if not available_models:
    st.sidebar.error(f"No .gguf models found in the directory: {MODEL_DIRECTORY}")
    st.stop() # Stop the app if no models are found

# Create a selectbox in the sidebar for model selection
selected_model_filename = st.sidebar.selectbox(
    "Choose a model:",
    available_models
)

# Construct the full path for the selected model
selected_model_path = os.path.join(MODEL_DIRECTORY, selected_model_filename)

# Load the selected model using the cached function
llm = load_llm(selected_model_path)

# --- Main App Display ---
# Display info about the currently loaded model in the main area
st.info(f"Currently using: **{selected_model_filename}**")
st.warning(f"**Context Window Limit:** The model can only 'see' about {N_CTX} tokens. Long conversations or documents may exceed this.")

# Initialize chat history in session state, specific to the *selected model*
# This prevents conversations from one model appearing for another
# We'll use the model filename as a key for the history
chat_history_key = f"messages_{selected_model_filename}"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

# Get the current chat history based on the selected model
current_messages = st.session_state[chat_history_key]


# --- Display Chat Messages ---
# Display messages from the current model's history
for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Your message:"):
    # Add user message to the current model's chat history
    current_messages.append({"role": "user", "content": prompt})
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Generate Assistant Response ---
    # Only generate if the last message in the current history is from the user
    if current_messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            # Use st.write_stream for streaming output into the chat bubble
            # We define a generator function to yield tokens from the LLM
            def generate_response():
                # Build the full prompt string from the current conversation history
                # Format: User: ...\nAssistant: ...\nUser: ...\nAssistant:
                full_prompt = ""
                for message in current_messages:
                    if message["role"] == "user":
                        full_prompt += f"User: {message['content']}\n"
                    elif message["role"] == "assistant":
                         full_prompt += f"Assistant: {message['content']}\n" # Include previous assistant replies

                # Add the start of the assistant's turn for the model to complete
                full_prompt += "Assistant:"
                # print(f"\n--- PROMPT SENT TO LLM ---\n{full_prompt}\n-------------------------\n") # Optional: debug prompt

                try:
                    # Call the LLM with the full history as the prompt and enable streaming
                    # The 'llm' variable here refers to the currently loaded model via @st.cache_resource
                    stream = llm(
                        full_prompt,
                        max_tokens=MAX_TOKENS_PER_RESPONSE,
                        stop=STOP_SEQUENCES,
                        temperature=TEMPERATURE,
                        echo=False,  # Don't include the prompt in the output stream
                        stream=True  # Enable streaming
                    )

                    generated_text = ""
                    # Iterate over the tokens yielded by the LLM
                    for token in stream:
                        text = token["choices"][0]["text"]
                        generated_text += text
                        yield text # Yield the token text to st.write_stream

                except Exception as e:
                    st.error(f"Error during generation: {e}")
                    yield f"\nError: {e}" # Yield error message to the chat
                    return "" # Return empty string on error

                # We need to return the full generated text so it can be saved to history
                return generated_text

            # Run the generator and stream output to the chat bubble
            full_response = st.write_stream(generate_response())

        # Add the full generated response to the current chat history
        # This happens *after* the streaming is complete
        current_messages.append({"role": "assistant", "content": full_response})


# --- Optional Buttons ---
# Add buttons below the chat input or in the sidebar
col1, col2 = st.columns(2) # Using columns for layout
with col1:
    if st.button("Clear Current Chat History"):
        st.session_state[chat_history_key] = [] # Clear only the history for the selected model
        st.rerun() # Rerun the app to clear the display

# You could add other buttons here if needed
# with col2:
#     st.button("Another Action")