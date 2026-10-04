# CourseMate AI

CourseMate AI is a document question-answering app built with Streamlit, LangChain,
Chroma, and Google's Gemini API. Upload a text-based PDF, process it into a vector
store, and ask questions about its contents.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your Google AI API key:

   ```text
   GOOGLE_API_KEY=your_google_api_key
   ```

4. Start the app:

   ```powershell
   streamlit run app.py
   ```

The PDF files and generated Chroma database are local data and are not included in
the repository. Upload a PDF through the app to create a local vector store.
