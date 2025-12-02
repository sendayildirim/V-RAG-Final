import requests
import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Class for downloading book and NarrativeQA questions
    """

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.book_url = "http://www.gutenberg.org/ebooks/21558.txt.utf-8"
        self.qaps_url = "https://raw.githubusercontent.com/google-deepmind/narrativeqa/refs/heads/master/qaps.csv"
        self.document_id = "04d0a3d15a1e39a94524a3958e433a88ca01fdf9"

    def download_book(self):

        response = requests.get(self.book_url)
        response.raise_for_status()

        book_path = self.data_dir / "children_of_new_forest.txt"
        with open(book_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"Book saved at {book_path}")
        return book_path

    def download_and_filter_questions(self):

        df = pd.read_csv(self.qaps_url)

        book_df = df[df['document_id'] == self.document_id].copy()

        print(f"Total {len(book_df)} questions found")
        print(f"Test: {len(book_df[book_df['set'] == 'test'])}")

        test_df = book_df[book_df['set'] == 'test']
        test_path = self.data_dir / "questions_test.csv"
        test_df.to_csv(test_path, index=False)

        print(f"Questions saved")
        print(f"  Test: {test_path}")


        return test_path

    def load_all_data(self):
        book_path = self.download_book()
        test_path = self.download_and_filter_questions()

        return {
            'book': book_path,
            'test': test_path
        }

if __name__ == "__main__":
    loader = DataLoader()
    paths = loader.load_all_data()
    print("Book and test questions downloaded")