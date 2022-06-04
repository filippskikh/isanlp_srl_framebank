import copy
import json

import conllu
import fire
import os
import pandas as pd
from bs4 import BeautifulSoup
from isanlp_srl_framebank.pipeline_default import PipelineDefault
import re

from isanlp.processor_remote import ProcessorRemote
from isanlp.processor_syntaxnet_remote import ProcessorSyntaxNetRemote
from isanlp import PipelineCommon
from isanlp.ru.converter_mystem_to_ud import ConverterMystemToUd


class CorefProcessor:

    def _parse_groups_txt(self):
        result = {'heads': {}}

        lines = open(f"{self._workdir}/Groups.txt").read().split("\n")
        header = lines[0].split("\t")
        for line in lines[1:]:
            res = {}
            if not line.strip():
                continue

            parts = line.strip().split("\t")
            for i in range(len(parts)):
                res[header[i]] = parts[i]

            if res['doc_id'] not in result:
                result[res['doc_id']] = {}

            if res['chain_id'] not in result[res['doc_id']]:
                result[res['doc_id']][res['chain_id']] = {}

            if res['group_id'] not in result[res['doc_id']][res['chain_id']]:
                result[res['doc_id']][res['chain_id']][res['group_id']] = res

            if res['doc_id'] not in result['heads']:
                result['heads'][res['doc_id']] = {}
            if res['chain_id'] not in result['heads'][res['doc_id']]:
                result['heads'][res['doc_id']][res['chain_id']] = res['group_id']

        self._docs = result

    def __init__(self, workdir):
        self._name = 'coref_pipeline'
        self._workdir = workdir
        self._parse_groups_txt()

    def __call__(self, tokens, sentences, postag, srl):
        token_coref = []
        doc = self.get_doc()

        sentence_by_token = {}
        token_in_sentence = {}
        sentence_token_to_token = {}
        for i, sentence in enumerate(sentences):
            sentence_token_to_token[i] = {}
            for j in range(sentence.begin, sentence.end):
                sentence_by_token[j] = i
                sentence_token_to_token[j - sentence.begin] = j
                token_in_sentence[j] = j - sentence.begin

        token_index = {}
        for i, tok in enumerate(tokens):
            token_index[str(tok.begin)] = i
            token_coref.append({})

        def find_group_tokens(group):
            grp_token_shifts = [ int(x) for x in group['tk_shifts'].split(',') ]
            grp_tokens = [ x.strip(",") if len(x) > 1 else x for x in entry['content'].split(' ') ]

            assert len(grp_tokens) == len(grp_token_shifts)

            result = []

            for i in range(len(grp_token_shifts)):
                result.append(-1)

                tk_shift = grp_token_shifts[i]
                tk_content = grp_tokens[i]

                for offset in range(max(0, int(tk_shift) - 50), int(tk_shift) + 50):
                    if str(offset) in token_index and tk_content == tokens[token_index[str(offset)]].text:
                        result[i] = token_index[str(offset)]
                        break

            if -1 in result:
                print(entry['content'])
                return None
            return result

        def find_head(group):
            grp_token_shifts = [ int(x) for x in group['tk_shifts'].split(',') ]
            if 'head' not in group:
                return [ True for _ in grp_token_shifts ]
            result = [ False for _ in grp_token_shifts ]
            head_shifts = set([ int(x) for x in group['hd_shifts'].split(',') ])
            for i in range (len(grp_token_shifts)):
                if grp_token_shifts[i] in head_shifts:
                    result[i] = True

            return result

        chain_sentences = {}
        chain_tokens = {}

        for chain_id in doc:
            chain_head = self.get_head(chain_id)

            for group_id in doc[chain_id]:
                entry = doc[chain_id][group_id]

                group_tokens = find_group_tokens(entry)
                head_tokens = find_head(entry)

                if not group_tokens:
                    continue

                for group_i, tok_i in enumerate(group_tokens):
                    sentence_id = sentence_by_token[tok_i]
                    if chain_id not in chain_sentences:
                        chain_sentences[chain_id] = set()

                    chain_sentences[chain_id].add(sentence_id)

                    if chain_id not in chain_tokens:
                        chain_tokens[chain_id] = []
                    if postag[sentence_id][token_in_sentence[tok_i]] in ['NOUN', 'PRON']:
                        chain_tokens[chain_id].append((
                            tok_i,
                            tokens[tok_i].text,
                            postag[sentence_id][token_in_sentence[tok_i]],
                            head_tokens[group_i]))

                    if head_tokens[group_i]:
                        token_coref[tok_i]['is_head'] = True
                    else:
                        token_coref[tok_i]['is_head'] = False

                    token_coref[tok_i]['chain_id'] = chain_id
                    token_coref[tok_i]['group_id'] = group_id
                    token_coref[tok_i]['head_group_id'] = chain_head
                    token_coref[tok_i]['link'] = entry['link']
                    if 'attributes' in entry:
                        token_coref[tok_i]['attributes'] = entry['attributes']
                    else:
                        token_coref[tok_i]['attributes'] = ""

        return {"coref": token_coref}

    def set_doc(self, doc_id):
        self._doc_id = doc_id

    def get_doc(self):
        return self._docs[self._doc_id]

    def get_head(self, chain_id):
        return self._docs['heads'][self._doc_id][chain_id]

def align(text):
    while True:
        new_text = re.sub("^(.*?)\s([^\s].*)", "\\1 \\2", text, 1)
        if new_text == text:
            return new_text
        text = new_text

# {"form": "волнуюсь", "lemma": "волноваться", "feat": "V ipf intr med sg praes 1p indic", "sem": "ca:noncaus t:move", "sem2": "ca:noncaus t:psych:emot", "rank": "Предикат", "pred": 203}

def get_doc_data(workdir, ppl, doc_id, doc_path):
    text = open(f"{workdir}/rucoref_texts/{doc_path}", mode='r', encoding='utf-8-sig').read()
    res = ppl('\n'.join([ x for x in text.split("\n")]))

    data = []

    sentence_by_token = {}
    token_in_sentence = {}
    for i, sentence in enumerate(res['sentences']):
        for j in range(sentence.begin, sentence.end):
            sentence_by_token[j] = i
            token_in_sentence[j] = j - sentence.begin

    groups = {}
    for i, coref in enumerate(res['coref']):
        if 'group_id' not in coref: continue
        if coref['group_id'] not in groups:
            groups[coref['group_id']] = []

        groups[coref['group_id']].append(i)

    for i, sentence in enumerate(res['sentences']):
        sdata = []

        srl = res['srl'][i]

        if not srl: continue

        preds = {}
        args = {}

        for event in srl:
            preds[event.pred[0]] = True
            for arg in event.args:
                args[arg.begin] = (event.pred[0], arg)

        j = sentence.begin
        while j < sentence.end:
            sent_i = j - sentence.begin

            token = res['tokens'][j]
            coref = res['coref'][j]

            if token.text == '':
                sdata.append({ "form": "" })
                j += 1
                continue

            sem = ""
            sem2 = ""
            if 'chain_id' in coref:
                attrs = [x.split(":") for x in coref["attributes"].split("|")]
                attrs = [ x[1] for x in attrs if len(x) > 1 ]

                sem = ' '.join(attrs)
                sem2 = coref['group_id']

            if res['mystem_postag'][i][sent_i].strip() == '':
                tok = {
                    "form": token.text
                }
            else:
                tok = {
                    "form": token.text,
                    "lemma": res['lemma'][i][sent_i],
                    "feat": ' '.join(res['morph'][i][sent_i].values()),
                    "sem": sem,
                    "sem2": sem2
                }

            if sent_i in preds:
                tok['rank'] = 'Предикат'
                tok['fillpred'] = sent_i

            if sent_i in args:
                tok['rank'] = 'Периферия'
                tok['rolepred1'] = args[sent_i][1].tag
                tok['fillpred'] = args[sent_i][0]

            if 'link' in coref and coref['link'] != '0' and coref['link'] in groups:
                grp_id = coref['group_id']
                l = j
                for k in range(j+1, sentence.end):
                    if 'group_id' not in res['coref'][k] or res['coref'][k]['group_id'] != grp_id:
                        break
                    l = k

                for _j in groups[coref['link']]:
                    new_tok = copy.deepcopy(tok)
                    new_tok['form'] = res['tokens'][_j].text
                    new_tok['lemma'] = res['lemma'][sentence_by_token[_j]][token_in_sentence[_j]]
                    new_tok['feat'] = ' '.join(res['morph'][sentence_by_token[_j]][token_in_sentence[_j]].values())
                    sdata.append(tok)

                j = l+1
                continue

            sdata.append(tok)

            j += 1

        data.append([f"{doc_id}_{i}", [sdata]])

    return data


def parse_coreference(workdir):

    return {}


def main(workdir):
    coref_processor = CorefProcessor(workdir)

    ppl = PipelineCommon([(ProcessorRemote('localhost', 3333, 'default'),
                           ['text'],
                           {'tokens': 'tokens',
                            'sentences': 'sentences',
                            'postag': 'mystem_postag',
                            'lemma': 'lemma'}),
                          (ProcessorSyntaxNetRemote('localhost', 3334),
                           ['tokens', 'sentences'],
                           {'syntax_dep_tree': 'syntax_dep_tree'}),
                          (ConverterMystemToUd(),
                           ['mystem_postag'],
                           {'morph': 'morph',
                            'postag': 'postag'}),
                          (ProcessorRemote('localhost', 3335, 'default'),
                           ['tokens', 'postag', 'morph', 'lemma', 'syntax_dep_tree'],
                           {'srl': 'srl'}),
                          (coref_processor, ['tokens', 'sentences', 'postag', 'srl'], {'coref': 'coref'})])

    docs = [ x.strip().split("\t") for x in open(f"{workdir}/Documents.txt").read().strip().split("\n") ][1:]

    result = []
    for doc in docs:
        if len(doc) < 2:
            break
        print(doc[0])
        coref_processor.set_doc(doc[0])
        doc_data = get_doc_data(workdir, ppl, doc[0], doc[1])

        result = result + doc_data

    out = open(f"{workdir}/cleared_corpus.json", "w")
    out.write(json.dumps(result, ensure_ascii=True))
    out.flush()
    out.close()


if __name__ == "__main__":
    fire.Fire(main)
