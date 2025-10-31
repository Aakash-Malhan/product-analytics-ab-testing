import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DEST = DATA_DIR / "ml-1m.zip"


def download_movielens():
    """
    Downloads MovieLens 1M and converts the latin-1 encoded .dat files
    into UTF-8 CSVs saved under data/ (ratings.csv, users.csv, movies.csv).
    Safe to call repeatedly; it will skip work if CSVs already exist.
    """
    ratings_csv = DATA_DIR / "ratings.csv"
    users_csv = DATA_DIR / "users.csv"
    movies_csv = DATA_DIR / "movies.csv"

    if ratings_csv.exists() and users_csv.exists() and movies_csv.exists():
        return

    print("Downloading MovieLens 1M...")
    r = requests.get(ML1M_URL, timeout=120)
    r.raise_for_status()
    DEST.write_bytes(r.content)

    # MovieLens .dat files are latin-1 (titles contain accented chars)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("ml-1m/ratings.dat") as f:
            ratings = pd.read_csv(
                f,
                sep="::",
                engine="python",
                encoding="latin-1",
                names=["userId", "movieId", "rating", "timestamp"],
            )
        with z.open("ml-1m/users.dat") as f:
            users = pd.read_csv(
                f,
                sep="::",
                engine="python",
                encoding="latin-1",
                names=["userId", "gender", "age", "occupation", "zip"],
            )
        with z.open("ml-1m/movies.dat") as f:
            movies = pd.read_csv(
                f,
                sep="::",
                engine="python",
                encoding="latin-1",
                names=["movieId", "title", "genres"],
            )

    # Save normalized UTF-8 CSVs
    ratings.to_csv(ratings_csv, index=False, encoding="utf-8")
    users.to_csv(users_csv, index=False, encoding="utf-8")
    movies.to_csv(movies_csv, index=False, encoding="utf-8")
    print("Saved data to data/ (UTF-8 CSVs).")


if __name__ == "__main__":
    download_movielens()
