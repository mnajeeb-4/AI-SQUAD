import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Configuration variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not all([GROQ_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("Missing required environment variables in .env file.")

# Initialize Supabase client using the service key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def get_recent_messages(thread_id: str, limit: int = 10) -> list:
    """
    Fetch the last `limit` messages for a given thread from Supabase.
    This implements 'Memory Slicing' to save tokens.
    """
    response = (
        supabase.table("messages")
        .select("role, content")
        .eq("thread_id", thread_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    # Reverse the list so the oldest message comes first for the LLM
    return response.data[::-1]


def save_message(thread_id: str, role: str, content: str, agent: str = "general"):
    """
    Save a new message (user or assistant) to the database.
    """
    supabase.table("messages").insert({
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "agent": agent
    }).execute()