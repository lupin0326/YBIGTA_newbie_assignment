from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    corpus = [example["verse_text"] for example in dataset["train"]]
    return corpus