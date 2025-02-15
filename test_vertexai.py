from vertexai.generative_models import GenerativeModel
import vertexai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_vertexai():
    """Initialize Vertex AI with project and location."""
    project_id = os.getenv('GCP_PROJECT_ID')
    location = os.getenv('GCP_REGION', 'us-central1')
    vertexai.init(project=project_id, location=location)

def test_gemini():
    """Test Gemini model with a simple prompt."""
    print("\n=== Testing Gemini Models ===\n")
    
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        location = os.getenv('GCP_REGION', 'us-central1')
        init_vertexai()

        # Test default model
        print("1. Testing gemini-2.0-flash-001:")
        model = GenerativeModel("gemini-2.0-flash-001")
        print("Model initialized, sending test prompt...")
        
        response = model.generate_content(
            "Write a short haiku about coding.",
        )
        print(f"Response: {response.text}\n")

        # Test reasoning model
        print("\n2. Testing gemini-2.0-flash-thinking-exp-01-21:")
        model = GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
        print("Model initialized, sending test prompt...")
        
        response = model.generate_content(
            "Analyze the pros and cons of using Python for AI development in one sentence.",
        )
        print(f"Response: {response.text}\n")

        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nGCP Configuration:")
        print(f"Project ID: {project_id}")
        print(f"Location: {location}")
        raise

if __name__ == "__main__":
    test_gemini()
