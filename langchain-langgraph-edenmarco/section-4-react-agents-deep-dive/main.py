from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the tools
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    return len(text)


def main():
    print(get_text_length(text="Hello from section-4-react-agents-deep-dive!"))


if __name__ == "__main__":
    main()
