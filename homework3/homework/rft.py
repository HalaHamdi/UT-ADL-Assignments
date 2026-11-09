from .base_llm import BaseLLM
from .sft import TokenizedDataset, Dataset
from .data import benchmark


class RFTModel(BaseLLM):
    def batched_generate(self, prompts, num_return_sequences=None, temperature=0, new_max_tokens=120):
        return super().batched_generate(prompts, num_return_sequences, temperature, new_max_tokens)


def test_model(ckpt_path: str):
    from peft import PeftModel

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    benchmark_result = benchmark(llm, Dataset("valid"), 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def format_example(question: str, answer: float, reasoning: str) -> dict[str, str]:
    return {"question": question, "answer": reasoning}

def train_model(
    output_dir: str,
    **kwargs,
):
    from datetime import datetime
    from pathlib import Path
    from peft import LoraConfig,get_peft_model
    from transformers import Trainer,TrainingArguments

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "rft_model")
        print(f"Output directory not specified. Using default: {output_dir}")
    print(f"Using output directory: {output_dir}")

    # 1. perft config (LoraConfig)
    r = 8
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=r,
        lora_alpha=r*5
    )
    llm = BaseLLM()
    # 2. create a peft model using `get_peft_model`
    model = get_peft_model(llm.model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    dataset = Dataset("rft")
    print(f"Loaded dataset sample input: {dataset[0]}")

    # 3. Train your model using `Trainer`
    # 3.a. Define a TrainingArguments class that will contain all the hyperparameters 
    # the Trainer will use for training and evaluation. 
    train_dataset = TokenizedDataset(tokenizer=llm.tokenizer, data=dataset, format_fn=format_example)
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=2e-4,
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs/{datetime.now():%Y-%m-%d_%H-%M-%S}",
        report_to="tensorboard",
        logging_steps=3,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_strategy="epoch",
        logging_first_step=True
    )
    # 3.b define a Trainer by passing it all the objects constructed 
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    # 3.c  call the train() method of our Trainer
    trainer.train()
    # 4. save model using  `save_pretrained
    trainer.save_model(output_dir)
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
