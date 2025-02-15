from pydantic_ai.models.vertexai import VertexAIModel
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()

async def test_vertex_ai_connection():
    """Test Vertex AI connection and models"""
    print("\n=== Testing Vertex AI Connection ===")
    
    try:
        project_id = os.getenv('GCP_PROJECT_ID')
        region = os.getenv('GCP_REGION', 'us-central1')

        print("\n1. Testing default model (gemini-2.0-flash-001)...")
        default_model = VertexAIModel('gemini-2.0-flash-001', project_id=project_id, region=region)
        test_prompt = "Write a short haiku about coding."
        
        result = await default_model.ainit()
        response = await default_model.predict(test_prompt)
        print("✅ Successfully tested default model")
        print(f"Response: {response.text if hasattr(response, 'text') else response}")

        print("\n2. Testing reasoning model (gemini-2.0-flash-thinking-exp-01-21)...")
        reasoning_model = VertexAIModel('gemini-2.0-flash-thinking-exp-01-21', project_id=project_id, region=region)
        test_prompt = "Analyze the pros and cons of using Python for AI development in one sentence."
        
        result = await reasoning_model.ainit()
        response = await reasoning_model.predict(test_prompt)
        print("✅ Successfully tested reasoning model")
        print(f"Response: {response.text if hasattr(response, 'text') else response}")

    except Exception as e:
        print(f"❌ Error testing Vertex AI: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise

async def main():
    """Run Vertex AI connection test"""
    print("\n🚀 Starting Vertex AI Connection Test\n")
    
    try:
        await test_vertex_ai_connection()
        print("\n✨ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")
        
    finally:
        print("\n--- Test Run Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
