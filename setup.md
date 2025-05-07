Prerequisites:
Python 3.3
 Visual Studio Build Tools 2022 (or Visual Studio IDE with "Desktop development with C++")
 .gguf llm models

Step 1: Open the Developer Command Prompt for VS 2022
    Open the Windows Start Menu.
    Search for "Developer Command Prompt for VS 2022".
    Click on the shortcut to open it.
    Navigate to Your Project Directory

Step 2: Create a Virtual Environment and activate it
    python -m venv .venv
    .\.venv\Scripts\activate

Step 3: Install Streamlit
    pip install --upgrade streamlit

Step 4: Set Environment Variables for llama-cpp-python Build
    set CMAKE_ARGS="-DLLAMA_ACCELERATE=on -DLLAMA_AVX=on -DLLAMA_AVX2=on -DLLAMA_AVX512=on -DLLAMA_FMA=on  -DLLAMA_F16=on"
    set FORCE_CMAKE=1

Step 5: Install llama-cpp-python
    pip install --upgrade --no-cache-dir llama-cpp-python

Step 6: Run Application
    python chat_app.py
    OR
    streamlit run chat_app.py