"""MongoDB connection and database management."""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.core.config.logging_config import LoggerAdapter
from src.core.config.settings import settings

logger = LoggerAdapter(__name__)

# Global MongoDB client and database instances
mongodb_client: Optional[AsyncIOMotorClient] = None
mongodb: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongodb() -> None:
    """Create MongoDB connection."""
    global mongodb_client, mongodb
    
    try:
        logger.info("Connecting to MongoDB", url=settings.mongodb_url)
        
        mongodb_client = AsyncIOMotorClient(
            settings.mongodb_url,
            maxPoolSize=50,
            minPoolSize=10,
            serverSelectionTimeoutMS=5000,
        )
        
        # Verify connection
        await mongodb_client.admin.command("ping")
        
        mongodb = mongodb_client[settings.mongodb_database]
        
        # Create indexes
        await create_indexes()
        
        logger.info("Successfully connected to MongoDB")
        
    except Exception as e:
        logger.error("Failed to connect to MongoDB", error=str(e))
        raise


async def close_mongodb_connection() -> None:
    """Close MongoDB connection."""
    global mongodb_client
    
    if mongodb_client:
        logger.info("Closing MongoDB connection")
        mongodb_client.close()
        mongodb_client = None
        logger.info("MongoDB connection closed")


async def create_indexes() -> None:
    """Create database indexes for optimal performance."""
    if not mongodb:
        return
    
    try:
        # Conversations collection indexes
        conversations = mongodb["conversations"]
        await conversations.create_index("user_id")
        await conversations.create_index("session_id")
        await conversations.create_index([("created_at", -1)])
        await conversations.create_index([("user_id", 1), ("created_at", -1)])
        
        # Documents collection indexes
        documents = mongodb["documents"]
        await documents.create_index("user_id")
        await documents.create_index("document_id")
        await documents.create_index([("created_at", -1)])
        await documents.create_index([("user_id", 1), ("status", 1)])
        
        # Users collection indexes
        users = mongodb["users"]
        await users.create_index([("email", 1)], unique=True)
        await users.create_index("username")
        
        # Vector metadata collection indexes
        vector_metadata = mongodb["vector_metadata"]
        await vector_metadata.create_index("document_id")
        await vector_metadata.create_index("chunk_id")
        await vector_metadata.create_index([("document_id", 1), ("chunk_id", 1)])
        
        logger.info("Database indexes created successfully")
        
    except Exception as e:
        logger.error("Failed to create indexes", error=str(e))


def get_database() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    if not mongodb:
        raise RuntimeError("MongoDB is not connected")
    return mongodb


def get_collection(name: str):
    """Get MongoDB collection."""
    db = get_database()
    return db[name]