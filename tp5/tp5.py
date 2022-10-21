import time
from typing import IO, Generator, Tuple

from apyori import apriori

FILE_NAME = "results.txt"

configs: tuple = (
    (0.01, 0.7, 1.2, 2),
    (0.01, 0.7, 1.2, 3),
    (0.02, 0.7, 1.2, 2),
    (0.01, 0.7, 1.3, 2),
    (0.01, 0.8, 1.2, 2),
    (0.05, 0.7, 1.2, 2),
    (0.01, 0.3, 1.2, 2),
    (0.01, 0.7, 0.8, 2),
    (0.01, 0.7, 1.2, 1),
    (0.01, 0.14, 1.2, 2),
    (0.01, 0.7, 1.6, 2),
    (0.01, 0.7, 1.2, 6),
)


def main():
    init_save_file(FILE_NAME)

    f: IO = open('corpus.txt', 'r', encoding='utf-8')

    stopwords = {"the", "a", "of", "for ", "in", "and", "de", "et", "pour"}
    transactions = []

    for line in f:
        if len(line) > 1 and not line.startswith('#'):
            words = normalize(line).split()
            items = set()
            for word in words:
                if word not in stopwords:
                    items.add(word)
            if len(items) != 0:
                transactions.append(sorted(items))

    for config in configs:
        apiori_with_config(transactions, config[0], config[1], config[2], config[3])

    return 0


def apiori_with_config(transactions: list, support: float, confidence: float, lift: float, length: int):
    # Benchmark time for apriori
    start = time.time()

    results: Generator[Tuple, any, any] = apriori(transactions, min_support=support, min_confidence=confidence,
                                                  min_lift=lift,
                                                  min_length=length)

    end = time.time()

    # convert time to ns and round to 3 decimals
    time_elapsed = round((end - start) * 1000000000, 3)

    save_results(results, (support, confidence, lift, length), time_elapsed)


def init_save_file(filename: str):
    f = open(filename, 'w', encoding='utf-8')
    f.write(f"TP 5 - Apriori Algorithm\n\n")
    f.close()


def save_results(results: Generator[Tuple, any, any], config: tuple, time: float):
    counter = 0
    text = ""
    text += "Support: " + str(config[0]) + "\n"
    text += "Confidence: " + str(config[1]) + "\n"
    text += "Lift: " + str(config[2]) + "\n"
    text += "Length: " + str(config[3]) + "\n"
    text += "Time: " + str(time) + " ns\n"
    text += "Results: " + "\n"
    for result in results:
        text += str(result) + "\n"
        counter += 1
    text += "Number of Results: " + str(counter) + "\n"
    text += "-----------------------------------\n"
    write_to_file(FILE_NAME, text)


def write_to_file(filename: str, text: str):
    f = open(filename, 'a', encoding='utf-8')
    f.write(text)
    f.close()


def normalize(text: str) -> str:
    return text.strip("\n").replace("!", "").replace("?", "").replace(".", "").replace(",", "").replace(":", "").replace(";", "").replace("-", "")


if __name__ == '__main__':
    main()
