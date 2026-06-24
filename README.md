# RecoMovie
A simple content-based movie recommendation engine built with Streamlit and Scikit-learn. The app uses movie metadata (genres, keywords, director) to find similar movies and recommend them to the user.

## Features

- **Content-Based Filtering**: Recommends movies based on the similarity of their metadata.
- **Movie Metadata**: Uses genres, keywords, and director information to build a feature matrix.
- **Cosine Similarity**: Calculates similarity scores between movies.
- **Clean UI**: Simple and intuitive interface for movie selection and recommendation display.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1.  **Clone the repository** (if applicable) or copy the code.

2.  **Install dependencies**:

    ```bash
    pip install streamlit pandas scikit-learn
    ```

3.  **Prepare the dataset**:
    - Ensure `movie_data.csv` is present in the same directory as `app.py`.
    - The dataset should contain at least the following columns:
        - `title`
        - `genres`
        - `keywords`
        - `director`
        - `poster_url`

## Usage

1.  **Run the app**:

    ```bash
    streamlit run app.py
    ```

2.  **Use the interface**:
    - Select a movie from the dropdown list in the sidebar.
    - The app will display the selected movie's poster and information.
    - Scroll down to see the top 5 recommended movies based on your selection.
