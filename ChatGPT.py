from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a test."},
        {"role": "user", "content": "Give me a hello world, with some creativity"}
    ]
)

print(completion.choices[0].message)