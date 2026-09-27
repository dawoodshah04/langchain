from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

from typing import TypedDict, Annotated, Optional


# Load environment variables
load_dotenv()


# Create Hugging Face endpoint
_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    provider="groq",
    huggingfacehub_api_token=os.getenv("HF_ACCESS_TOKEN"),
)


llm = ChatHuggingFace(
    llm=_endpoint
)


class Review(TypedDict):
    key_themes: Annotated[
        list[str],
        "Write down all the key themes discussed in this review"
    ]

    summary: Annotated[
        str,
        "A brief summary of the review"
    ]

    sentiment: Annotated[
        str,
        "Return sentiment of the review either negative, positive or neutral"
    ]

    pros: Annotated[
        Optional[list[str]],
        "Write down all pros inside a list"
    ]

    cons: Annotated[
        Optional[list[str]],
        "Write down all cons inside a list"
    ]



structured_output = llm.with_structured_output(Review)


# Review text
review = """
Category    Summary
Performance    Excellent A19 Pro performance, suitable for gaming and demanding tasks
Camera    Excellent triple 48MP camera system with strong photo and video quality
Battery    Very good battery life with faster charging
Display    Excellent 6.3-inch OLED display with 120Hz ProMotion
Design    Premium, but heavier and the new camera design is divisive
AI    Useful features, but Apple's AI capabilities still have limitations
Price    Expensive, especially with higher storage

Pros:
- Excellent performance
- Strong camera and video quality
- Very good battery life
- High-quality 120Hz display
- Improved front camera
- Fast charging
- Long-term software support

Cons:
- Expensive
- Relatively small upgrade from the iPhone 16 Pro
- Heavier than previous Pro models
- Design changes may not appeal to everyone
- AI features are not as advanced as some competitors
"""


# Invoke the structured model
result = structured_output.invoke(review)


# Print complete result
print("\nComplete Result:")
print(result)


# Access individual fields
print("\nKey Themes:")
print(result["key_themes"])

print("\nSummary:")
print(result["summary"])

print("\nSentiment:")
print(result["sentiment"])

print("\nPros:")
print(result["pros"])

print("\nCons:")
print(result["cons"])
