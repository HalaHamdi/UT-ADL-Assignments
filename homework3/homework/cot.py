from pyexpat.errors import messages

from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        # I used sonnet 5.5 to generate EXAMPLES mentioning that the llm role is to unit coversion, give me few examples for different unit conversions
        EXAMPLES = {
            "feet_to_meters": [
                {"role": "user", "content": "How many meters are in 5 feet?"},
                {"role": "assistant", "content": "1 foot = 0.3048 meters. 5 * 0.3048 = 1.524. <answer>1.524</answer>"},
            ],
            "yards_to_feet": [
                {"role": "user", "content": "How many feet are in 2 yards?"},
                {"role": "assistant", "content": "1 yard = 3 feet. 2 * 3 = 6. <answer>6</answer>"},
            ],
            "meters_to_cm": [
                {"role": "user", "content": "How many centimeters are in 4 meters?"},
                {"role": "assistant", "content": "1 meter = 100 centimeters. 4 * 100 = 400. <answer>400</answer>"},
            ],
            "yards_to_meters": [
                {"role": "user", "content": "How many meters are in 10 yards?"},
                {"role": "assistant", "content": "1 yard = 0.9144 meters. 10 * 0.9144 = 9.144. <answer>9.144</answer>"},
            ],
            "miles_to_km": [
                {"role": "user", "content": "How many kilometers are in 3 miles?"},
                {"role": "assistant", "content": "1 mile = 1.609344 kilometers. 3 * 1.609344 = 4.828032. <answer>4.828032</answer>"},
            ],
            "feet_to_yards": [
                {"role": "user", "content": "How many yards are in 9 feet?"},
                {"role": "assistant", "content": "1 yard = 3 feet. 9 / 3 = 3. <answer>3</answer>"},
            ],
            "meters_to_feet": [
                {"role": "user", "content": "How many feet are in 6 meters?"},
                {"role": "assistant", "content": "1 foot = 0.3048 meters. 6 / 0.3048 = 19.685. <answer>19.685</answer>"},
            ],
            "pounds_to_ounces": [
                {"role": "user", "content": "How many ounces are in 2 pounds?"},
                {"role": "assistant", "content": "1 pound = 16 ounces. 2 * 16 = 32. <answer>32</answer>"},
            ],
            "minutes_to_seconds": [
                {"role": "user", "content": "How many seconds are in 3 minutes?"},
                {"role": "assistant", "content": "1 minute = 60 seconds. 3 * 60 = 180. <answer>180</answer>"},
            ],
            # "kb_to_bits": [
            #     {"role": "user", "content": "What does 6 KB equal in bit terms?"},
            #     {"role": "assistant", "content": "1 KB = 1000 byte, so 6 KB = 6 * 1000 = 6000 byte. 1 byte = 8 bit, so 6000 * 8 = 48000. <answer>48000</answer>"},
            # ],
            # "ton_to_kg": [
            #     {"role": "user", "content": "Can you change 9 metric ton to its equivalent in kg?"},
            #     {"role": "assistant", "content": "1 metric ton = 1000 kg. 9 * 1000 = 9000. <answer>9000</answer>"},
            # ],
            # "ton_to_pound": [
            #     {"role": "user", "content": "How do we express 3 ton in terms of pound?"},
            #     {"role": "assistant", "content": "1 ton = 1000 kg, so 3 ton = 3000 kg. 1 kg = 2.2046226 pound, so 3000 * 2.2046226 = 6613.8678. <answer>6613.8678</answer>"},
            # ],
        }
        chat: list[dict[str, str]] = [
            {"role": "system", "content": "You convert units. Think step by step, be concise, "
                       "and put the final number in <answer></answer> tags."},
            ]
        for pair in EXAMPLES.values():
            chat.extend(pair)
        chat.append({"role": "user", "content": question})
        res=self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        return res


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
