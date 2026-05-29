# SPDX-License-Identifier: AGPL-3.0-or-later
# pylint: disable=missing-module-docstring


import hashlib
import random
import string
import uuid

from searx.result_types import Answer
from searx.result_types.answer import BaseAnswer
from src.searx.answerers._core import AnswererInfo

from . import Answerer, AnswererInfo


def random_characters() -> list[LiteralString]:
    random_string_letters = (
        string.ascii_lowercase + string.digits + string.ascii_uppercase
    )
    return [random.choice(random_string_letters) for _ in range(random.randint(8, 32))]


def random_string() -> LiteralString:
    return "".join(random_characters())


def random_float() -> str:
    return str(random.random())


def random_int() -> str:
    random_int_max = 2**31
    return str(random.randint(-random_int_max, random_int_max))


def random_sha256() -> str:
    m = hashlib.sha256()
    m.update("".join(random_characters()).encode())
    return str(m.hexdigest())


def random_uuid() -> str:
    return str(uuid.uuid4())


def random_color() -> str:
    color = f"{random.randint(0, 0xFFFFFF):06x}"
    return f"#{color.upper()}"


class SXNGAnswerer(Answerer):
    """Random value generator"""

    keywords = ["random"]

    random_types = {
        "string": random_string,
        "int": random_int,
        "float": random_float,
        "sha256": random_sha256,
        "uuid": random_uuid,
        "color": random_color,
    }

    def info(self) -> AnswererInfo:

        return AnswererInfo(
            name=gettext(self.__doc__),
            description=gettext("Generate different random values"),
            keywords=self.keywords,
            examples=[f"random {x}" for x in self.random_types],
        )

    def answer(self, query: str) -> list[BaseAnswer]:

        parts = query.split()
        if len(parts) != 2 or parts[1] not in self.random_types:
            return []

        return [Answer(answer=self.random_types[parts[1]]())]
