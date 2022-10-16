"""
A Dataset Reader for Universal Dependencies, with support for multiword tokens and special handling for NULL "_" tokens
"""

from typing import Dict, Tuple, List, Any, Callable

from allennlp.data.tokenizers.word_splitter import WordSplitter, SpacyWordSplitter
from overrides import overrides
from io_.dataset_readers.parser import parse_line, DEFAULT_FIELDS

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from io_.dataset_readers.lemma_edit import gen_lemma_rule

import logging

logger = logging.getLogger('mylog')

sample_id = 0

def lazy_parse(text: str, fields: Tuple[str, ...]=DEFAULT_FIELDS):
    for sentence in text.split("\n\n"):
        if sentence:
            # TODO: upgrade conllu library
            yield [parse_line(line, fields)
                   for line in sentence.split("\n")
                   if line and not line.strip().startswith("#")]

@DatasetReader.register("universal-dependencies-reader")
class UniversalDependenciesDatasetReader(DatasetReader):
    def __init__(self,
                 tasks: List[str] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 ) -> None:
        super().__init__(lazy)
        self.tasks = [task for task in tasks] if tasks is not None else []
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        global sample_id

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)
            for annotation in lazy_parse(conllu_file.read()):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                multiword_tokens = [x for x in annotation if x["multi_id"] is not None]
                annotation = [x for x in annotation if x["id"] is not None]

                if len(annotation) == 0:
                    continue

                def get_field(tag: str, map_fn: Callable[[Any], Any] = None) -> List[Any]:
                    map_fn = map_fn if map_fn is not None else lambda x: x
                    return [map_fn(x[tag]) if x[tag] is not None else "_" for x in annotation if tag in x]

                # Extract multiword token rows (not used for prediction, purely for evaluation)
                ids = [x["id"] for x in annotation]
                multiword_ids = [x["multi_id"] for x in multiword_tokens]
                multiword_forms = [x["form"] for x in multiword_tokens]

                words = get_field("form")
                lemmas = get_field("lemma") if "lemma" in self.tasks else None
                lemma_rules = [gen_lemma_rule(word, lemma)
                               if lemma != "_" else "_"
                               for word, lemma in zip(words, lemmas)] if "lemma" in self.tasks else None
                upos_tags = get_field("upos")
                xpos_tags = get_field("xpos")if "xpos" in self.tasks else None
                feats = get_field("feats", lambda x: "|".join(k + "=" + v for k, v in x.items())
                if hasattr(x, "items") else "_")  if "feats" in self.tasks else None
                heads = get_field("head")
                dep_rels = get_field("deprel")
                dependencies = list(zip(dep_rels, heads))
                ner = get_field("ner")
                yield self.text_to_instance(words, lemmas, lemma_rules, upos_tags, xpos_tags,
                                            feats, dependencies, ner, ids, multiword_ids, multiword_forms, sample_id)
                sample_id += 1

    @overrides
    def text_to_instance(self,  # type: ignore
                         words: List[str],
                         lemmas: List[str] = None,
                         lemma_rules: List[str] = None,
                         upos_tags: List[str] = None,
                         xpos_tags: List[str] = None,
                         feats: List[str] = None,
                         dependencies: List[Tuple[str, int]] = None,
                         ner: List[str] = None,
                         ids: List[str] = None,
                         multiword_ids: List[str] = None,
                         multiword_forms: List[str] = None,
                         sample_id: int = 0) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in words], self._token_indexers)
        fields["tokens"] = tokens

        names = ["upos", "xpos", "feats", "lemmas", "ner"]
        all_tags = [upos_tags, xpos_tags, feats, lemma_rules, ner]
        for name, field in zip(names, all_tags):
            if field:
                fields[name] = SequenceLabelField(field, tokens, label_namespace=name)

        head_tags = None
        head_indices = None
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField([x[0] for x in dependencies],
                                                     tokens,
                                                     label_namespace="head_tags")
            fields["head_indices"] = SequenceLabelField([int(x[1]) for x in dependencies],
                                                        tokens,
                                                        label_namespace="head_index_tags")
            head_tags = [x[0] for x in dependencies]
            head_indices = [x[1] for x in dependencies]

        fields["metadata"] = MetadataField({
            "words": words,
            "upos_tags": upos_tags,
            "xpos_tags": xpos_tags,
            "feats": feats,
            "lemmas": lemmas,
            "lemma_rules": lemma_rules,
            "head_tags":  head_tags,
            "head_indices": head_indices,
            "ner_tags": ner,
            "ids": ids,
            "multiword_ids": multiword_ids,
            "multiword_forms": multiword_forms,
            'sample_id': sample_id
        })

        return Instance(fields)

@DatasetReader.register("udify_universal_dependencies_raw")
class UniversalDependenciesRawDatasetReader(DatasetReader):
    """Like UniversalDependenciesDatasetReader, but reads raw sentences and tokenizes them first."""

    def __init__(self,
                 dataset_reader: DatasetReader,
                 tokenizer: WordSplitter = None) -> None:
        super().__init__(lazy=dataset_reader.lazy)
        self.dataset_reader = dataset_reader
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = SpacyWordSplitter(language="xx_ent_wiki_sm")

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, 'r') as conllu_file:
            for sentence in conllu_file:
                if sentence:
                    words = [word.text for word in self.tokenizer.split_words(sentence)]
                    yield self.text_to_instance(words)

    @overrides
    def text_to_instance(self,  words: List[str]) -> Instance:
        return self.dataset_reader.text_to_instance(words)