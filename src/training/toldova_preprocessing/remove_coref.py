import json

import fire


def main(workdir):
    corpus = json.loads(open(f"{workdir}/cleared_corpus.json", "r").read())

    for i in range(1):
        for j in range(len(corpus[i][1])):
            for k in range(len(corpus[i][1][j])):
                corpus[i][1][j][k]['sem'] = ""
                corpus[i][1][j][k]['sem2'] = ""

    f = open(f"{workdir}/cleared_corpus_nocoref.json", "w")
    f.write(json.dumps(corpus))
    f.flush()
    f.close()


if __name__ == "__main__":
    fire.Fire(main)
