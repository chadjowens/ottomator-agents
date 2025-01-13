import streamlit as st
import os
from supabase import create_client
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment variables
load_dotenv()

# Initialize clients
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def preprocess_query(query: str) -> str:
    """Clean and enhance the search query."""
    # Remove any special characters but keep spaces
    query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
    
    # Convert to lowercase
    query = query.lower()
    
    # Remove extra whitespace
    query = ' '.join(query.split())
    
    # Add common variations of terms
    replacements = {
        'how do i': 'how to',
        'how to': 'how do i',
        'what is': 'what are explain definition',
        'explain': 'what is definition',
    }
    
    for old, new in replacements.items():
        if old in query:
            query = f"{query} {new}"
    
    return query

def get_embedding(text: str) -> list[float]:
    """Get an embedding for the given text using OpenAI's API."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def generate_natural_response(query: str, results: list) -> str:
    """Generate a natural language response using GPT."""
    if not results:
        return None
    
    # Prepare context from results
    context = "\n\n".join([
        f"Source: {r.get('url', 'No URL')}\n"
        f"Content: {r.get('content', 'No content')}"
        for r in results[:5]  # Use top 5 results for more context
    ])
    
    # Create prompt for GPT
    prompt = f"""Based on the following documentation excerpts, provide a clear and concise answer to the query: "{query}"

Documentation excerpts:
{context}

Please provide a response that:
1. Directly answers the query
2. Includes relevant code examples if present (format them with ```python)
3. References specific documentation sources
4. Is formatted in markdown
5. Includes any relevant links to full documentation pages

Note: If the documentation excerpts don't fully answer the query, please mention what aspects are missing."""

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful documentation assistant for Pydantic AI. Provide clear, concise answers with relevant code examples when available. If information is missing or unclear, acknowledge this and suggest where the user might find more details."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content

def search_documents(query: str, match_count: int = 10):
    """Search for documents using vector similarity."""
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    # Debug info
    st.sidebar.markdown("### 🔍 Search Debug Info")
    st.sidebar.markdown(f"**Original Query:** {query}")
    st.sidebar.markdown(f"**Processed Query:** {processed_query}")
    
    # Get embedding
    query_embedding = get_embedding(processed_query)
    
    # Search with more results
    response = supabase.rpc(
        'match_site_pages',
        {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'filter': {}
        }
    ).execute()
    
    # Debug info about results
    if response.data:
        st.sidebar.markdown(f"**Number of results:** {len(response.data)}")
        st.sidebar.markdown("**Similarity Scores:**")
        for i, r in enumerate(response.data[:5], 1):
            st.sidebar.markdown(f"{i}. {r.get('similarity', 0)*100:.1f}%")
    
    return response.data

def main():
    st.title("🔍 Pydantic AI Documentation Search")
    st.markdown("""
    Ask questions about Pydantic AI in natural language and get clear, concise answers!
    """)

    # Search input
    query = st.text_input(
        "❓ What would you like to know?", 
        placeholder="e.g., 'How do I create an agent?' or 'Explain system prompts'"
    )

    if query:
        with st.spinner('🤔 Searching and analyzing documentation...'):
            results = search_documents(query)
            
            if not results:
                st.warning("No relevant documentation found. Try rephrasing your query.")
            else:
                # Generate natural language response
                response = generate_natural_response(query, results)
                
                # Display main response
                st.markdown("### 📝 Answer")
                st.markdown(response)
                
                # Show sources in an expander
                with st.expander("🔍 View Source Documents"):
                    for i, result in enumerate(results, 1):
                        similarity_percentage = result.get('similarity', 0) * 100
                        st.markdown(f"### Source {i}: {result.get('title', 'Untitled')} ({similarity_percentage:.1f}% match)")
                        st.markdown(f"**URL:** [{result.get('url', 'No URL')}]({result.get('url', '#')})")
                        if result.get('summary'):
                            st.markdown(f"**Summary:** {result.get('summary')}")
                        st.markdown("**Full Content:**")
                        st.markdown(result.get('content', 'No content available'))
                        st.markdown("---")

if __name__ == "__main__":
    main()
