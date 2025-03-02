"""Main entry point for the chat application."""
from app.biochat import BioChat

def main():
    """Initialize and run the chat application."""
    app = BioChat()
    app.run()

if __name__ == "__main__":
    main()
