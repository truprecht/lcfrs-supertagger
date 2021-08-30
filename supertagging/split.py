from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from typing import Generic, Iterable, Iterator, Tuple, TypeVar


I = TypeVar("I")

class Split(ABC, Generic[I]):
    @abstractmethod
    def iter_items(self, dataset: Iterable[I]) -> Iterator[Tuple[str, I]]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def names(self):
        return ("train", "dev", "test")


@dataclass
class SequenceSplit(Split[I]):
    train: int
    dev: int
    test: int
    skip_before_train: int = 0

    def iter_items(self, dataset: Iterable[I]) -> Iterator[Tuple[str, I]]:
        dataset = iter(islice(dataset, self.skip_before_train, self.skip_before_train+len(self)))
        for name in ("train", "dev", "test"):
            for _ in range(self.__getattribute__(name)):
                yield name, next(dataset)

    def __len__(self):
        return self.train + self.dev + self.test


@dataclass
class DebugSplit(Split[I]):
    train: int
    test: int

    def __post_init__(self):
        assert self.test <= self.train

    def iter_items(self, dataset: Iterable[I]) -> Iterator[Tuple[str, I]]:
        for i in range(self.train):
            item = next(dataset)
            yield "train", item
            yield "dev", item
            if i < self.test:
                yield "test", item

    def __len__(self):
        return 2*self.train + self.test


class SplitFactory:
    def produce(self, splitstring: str) -> Split:
        portions = splitstring.split()
        if portions[0] == "debug":
            assert len(portions) == 3
            return DebugSplit(*(int(n) for n in portions[1:]))
        assert len(portions) in (3,4)
        skips = 0
        if len(portions) == 4:
            skips = int(portions[0])
            portions = portions[1:]
        return SequenceSplit(*(int(n) for n in portions), skip_before_train=skips)