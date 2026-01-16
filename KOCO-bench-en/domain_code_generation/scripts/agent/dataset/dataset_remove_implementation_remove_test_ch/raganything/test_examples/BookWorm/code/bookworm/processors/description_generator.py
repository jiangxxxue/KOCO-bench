"""
Document description generator using AI
"""
import logging
from typing import Optional

# Optional imports for LLM integration
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

from ..utils import BookWormConfig


class DocumentDescriptionGenerator:
    """Generate AI descriptions for documents"""
    
    def __init__(self, config: BookWormConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ollama client (same pattern as mindmap generator)
        self.ollama_client = None
        try:
            if AsyncOpenAI and self.config.api_provider == "OLLAMA":
                ollama_url = f"{self.config.llm_host}/v1"
                self.ollama_client = AsyncOpenAI(
                    api_key="ollama",  # Ollama doesn't require a real API key
                    base_url=ollama_url
                )
        except ImportError:
            self.logger.warning("OpenAI client not available for Ollama integration")
        
    async def generate_description(self, markdown_mindmap) -> Optional[str]:
        """Generate an AI description for a document"""
        try:
            # Prepare the text for description generation
            text_content = markdown_mindmap
            
            # Create prompt for description generation
            prompt = f"""Analyze the following mindmap built from a document and provide a concise, informative description (2-3 sentences) that captures:

1. The main topic or subject matter
2. The type of content (academic, technical, business, etc.)
3. Key themes or focus areas

Text to analyze:
{text_content}

Provide only the description, no additional formatting or labels."""

            # Generate description using Ollama client
            try:
                if self.ollama_client:
                    response = await self.ollama_client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=50000,
                        temperature=0.3
                    )
                    
                    description = response.choices[0].message.content or ""
                    
                    if description:
                        # Clean up the description
                        description = description.strip()
                        # Remove any prefix like "Description:" if the model includes it
                        if description.lower().startswith("description:"):
                            description = description[12:].strip()
                        
                        self.logger.info(f"📝 Generated description: {description[:100]}...")
                        return description
                    else:
                        msg = "⚠️ Empty description generated"
                        self.logger.warning(msg)
                        return msg
                        
    
                else:
                    msg = "⚠️ Ollama client not available, using fallback description"
                    self.logger.warning(msg)
                    return msg

                    
            except Exception as e:
                msg = f"❌ Error calling Ollama for description generation: {e}"
                self.logger.error(msg)
                return msg
                
        except Exception as e:
            msg = f"❌ Error generating description for document: {e}"
            self.logger.error(msg)
            return f"❌ Error generating description: {e}"

    

