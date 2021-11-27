#!/usr/bin/env python3

from .model import add_accent

def test_model(sentence):
    return add_accent(sentence=sentence)

def _test():
    assert test_model(" ") == (None, None)

if __name__ == 'main':
    _test()