# text_summary.py
from transformers import pipeline

# Create a summarization pipeline
summarizer = pipeline("summarization", model="google-t5/t5-small")

def summarize_text(text: str, max_length: int = 50, min_length: int = 10) -> str:
    """
    Summarize a given text using Hugging Face Transformers.

    Args:
        text (str): The input text to summarize.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: Summarized text.
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example usage
if __name__ == "__main__":
    example_text = "This patient has 80% risk due to high cholesterol and age."
    result = summarize_text(example_text)
    print("Original text:", example_text)
    print("Summary:", result)
