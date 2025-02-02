import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

import random
# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        
        tokenized_corpus = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in corpus]

        for epoch in range(num_epochs):
            total_loss: float = 0.0
            for sentence in tokenized_corpus:
                if self.method == "cbow":
                    loss = self._train_cbow(sentence, criterion, optimizer)
                else:
                    loss = self._train_skipgram(sentence, criterion, optimizer)
                total_loss += float(loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    def _train_cbow(self, sentence: list[int], criterion: nn.CrossEntropyLoss, optimizer: Adam) -> float:
        loss_sum: float = 0.0
        for i in range(self.window_size, len(sentence) - self.window_size):
            context = sentence[i - self.window_size:i] + sentence[i + 1:i + 1 + self.window_size]
            target = sentence[i]

            context_embeds = self.embeddings(LongTensor(context)).mean(dim=0)
            output = self.weight(context_embeds)
            loss = criterion(output.unsqueeze(0), LongTensor([target]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
        return loss_sum

    def _train_skipgram(self, sentence: list[int], criterion: nn.CrossEntropyLoss, optimizer: Adam) -> float:
        loss_sum: float = 0.0
        for i in range(self.window_size, len(sentence) - self.window_size):
            target = sentence[i]
            context = sentence[i - self.window_size:i] + sentence[i + 1:i + 1 + self.window_size]

            target_embed = self.embeddings(LongTensor([target]))  # Skip-gram은 개별 문맥 단어를 예측
            optimizer.zero_grad()  # 기존의 gradient를 초기화하여 누적 방지

            for ctx in context:
                output = self.weight(target_embed).squeeze(0)
                loss = criterion(output.unsqueeze(0), LongTensor([ctx]))

                loss.backward(retain_graph=True)  # 그래프 유지
                optimizer.step()  # 가중치 업데이트
                optimizer.zero_grad()  # 한 번 업데이트 후 gradient 초기화

                loss_sum += float(loss.item())

        return loss_sum