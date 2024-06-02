from flask import Flask, request, jsonify
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


@app.route('/analyze', methods=['POST'])
def analyze_image():
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({"error": "No image URL provided"}), 400

    image_url = data['image_url']

    prompt = (
        "You are an assistant that helps in understanding product labels. "
        "Please analyze the provided image and explain in layman terms the details "
        "about all ingredients, it should be so simple my grandma could understand but complete,"
        "whether they should consume them or not, this should be a straightforward non misguiding answer,"
        "and the description should be short, otherwise no one will read it, maybe 100 words would do"
        "provide descriptions for any coded items. Format the response in markdown."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant in understanding product labels."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}
        ],
        temperature=0.0
    )

    explanation = response.choices[0].message.content
    return jsonify({"markdown": explanation})


if __name__ == '__main__':
    app.run(debug=True)
