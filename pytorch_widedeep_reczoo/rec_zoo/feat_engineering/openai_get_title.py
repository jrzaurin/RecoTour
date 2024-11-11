import json
import asyncio
from typing import Dict, List

from openai import AsyncOpenAI

from rec_zoo.tokens_and_api_keys import OPENAI_API_KEY


async def get_movie_details_async(title: str, api_key: str) -> Dict[str, str]:
    client = AsyncOpenAI(api_key=api_key)

    prompt = f"""
    For the movie "{title}", provide a few-sentences-overview and runtime in
    minutes. Do not include the title of the movie in the overview. If you
    are unable to find the movie, or are unsure of the details, please leave
    the fields empty.

    Return ONLY a JSON object with this exact format:
    {{"overview": "description here", "runtime": "XXX minutes"}}
    """

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a movie database API. Always respond with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        return {"overview": result.get("overview"), "runtime": result.get("runtime")}

    except Exception as e:
        print(f"Error getting movie details for {title}: {str(e)}")
        return {"overview": "", "runtime": ""}


async def process_movies_concurrent(
    titles: List[str], api_key: str, max_concurrent: int = 5
):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_get_details(title):
        async with semaphore:
            return await get_movie_details_async(title, api_key)

    tasks = [bounded_get_details(title) for title in titles]
    return await asyncio.gather(*tasks)


# Example usage
async def main():
    api_key = OPENAI_API_KEY
    movie_titles = [
        "The Shawshank Redemption",
        "The Godfather",
        "The Dark Knight",
        # Add more titles as needed
    ]

    results = await process_movies_concurrent(movie_titles, api_key, max_concurrent=3)

    return dict(zip(movie_titles, results))


if __name__ == "__main__":
    res = asyncio.run(main())
