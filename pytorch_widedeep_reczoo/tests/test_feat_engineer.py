import pandas as pd
import pytest

from rec_zoo.feat_engineering.feat_engineer import FeatureEngineer


@pytest.fixture
def sample_movies_df():
    # Sample movie titles with years (matching MovieLens format)
    titles_with_years = [
        "The Big Adventure (1995)",
        "Star Wars (1977)",
        "The Lord of the Rings (2001)",
        "Matrix (1999)",
        "The Incredible Journey (1963)",
    ] * 20  # 5 movies repeated 20 times = 100 rows

    # Sample genres with varying counts per movie
    genres = [
        "Action|Adventure|Sci-Fi",
        "Drama|Romance",
        "Fantasy|Adventure",
        "Sci-Fi|Action",
        "Adventure|Family",
    ] * 20

    # Additional columns
    user_ids = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ] * 10  # 10 users, each with 10 interactions
    item_ids = list(range(1, 6)) * 20  # 5 movies repeated 20 times
    genders = ["M", "F"] * 50  # Alternating M/F for 100 rows
    occupations = [
        "student",
        "engineer",
        "teacher",
        "doctor",
        "artist",
        "lawyer",
        "programmer",
        "scientist",
        "writer",
        "other",
    ] * 10  # 10 occupations repeated 10 times

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "item_id": item_ids,
            "title": titles_with_years,
            "genres": genres,
            "gender": genders,
            "occupation": occupations,
        }
    )

    return df


def test_year_and_title_extraction(sample_movies_df):
    fe = FeatureEngineer(title_col="title", genre_col="genres", cat_cols=["genres"])
    transformed_df = fe._extract_year_and_title(sample_movies_df)
    assert "year" in transformed_df.columns
    assert fe.clean_title_col in transformed_df.columns
    assert transformed_df["year"].dtype == object
    assert all(transformed_df["year"].str.match(r"^\d{4}$"))
    # Check specific year extraction
    assert transformed_df.loc[0, "year"] == "1995"
    # Check clean title
    assert transformed_df.loc[0, fe.clean_title_col] == "The Big Adventure"


def test_title_word_count(sample_movies_df):
    fe = FeatureEngineer(title_col="title", genre_col="genres", cat_cols=["genres"])
    # First need to extract year and title
    df = fe._extract_year_and_title(sample_movies_df)
    transformed_df = fe._add_title_word_count(df)

    assert "title_word_count" in transformed_df.columns
    assert transformed_df["title_word_count"].dtype == int
    assert all(transformed_df["title_word_count"] > 0)
    # Check specific word counts
    assert transformed_df.loc[0, "title_word_count"] == 3  # "The Big Adventure"
    assert transformed_df.loc[1, "title_word_count"] == 2  # "Star Wars"


def test_format_genres(sample_movies_df):
    fe = FeatureEngineer(title_col="title", genre_col="genres", cat_cols=["genres"])
    transformed_df = fe._format_genres(sample_movies_df)

    # Check if genres are properly formatted
    assert all(transformed_df[fe.genre_col].str.contains("_"))
    assert not any(transformed_df[fe.genre_col].str.contains(r"\|"))
    assert all(transformed_df[fe.genre_col].str.islower())

    # check specific genres
    assert transformed_df.loc[0, fe.genre_col] == "action_adventure_sci-fi"


def test_full_transformation(sample_movies_df):
    fe = FeatureEngineer(
        title_col="title",
        genre_col="genres",
        cat_cols=[
            "user_id",
            "item_id",
            "genres",
            "gender",
            "occupation",
            "year",
            "title_word_count",
        ],
    )

    transformed_df = fe.fit_transform(sample_movies_df)

    # Check all expected columns exist
    assert "year" in transformed_df.columns
    assert "title_word_count" in transformed_df.columns
    assert fe.genre_col in transformed_df.columns
    assert fe.clean_title_col in transformed_df.columns

    # Check if genres are properly encoded
    assert transformed_df[fe.genre_col].dtype == int
    assert not transformed_df[fe.genre_col].isna().any()

    # Check all categorical columns are part of the label_encoder.encoding_dict
    for col in fe.cat_cols:
        assert col in fe.label_encoder.encoding_dict
