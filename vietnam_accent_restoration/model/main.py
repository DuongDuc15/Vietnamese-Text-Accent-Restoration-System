#!/usr/bin/env python3
import warnings
from .result.mothermodel import test_model

warnings.filterwarnings("ignore")

sentence = "thoi tiet hom nay rat nong"
sentence = "hom nay toi di choi xa"


def main(string):
    return (test_model(sentence=string))


if __name__ == '__main__':
    print(main())
