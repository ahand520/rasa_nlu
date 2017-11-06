# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os

import typing
from builtins import str
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text
from future.utils import PY3
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from sklearn.feature_extraction.text import TfidfVectorizer
logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import numpy as np
    from rasa_nlu.model import Metadata


class TfidfFeaturizer(Featurizer):
    name = "intent_featurizer_tfidf"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    def __init__(self, training_message=None):
        self.training_message = training_message if training_message else []
    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "sklearn"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        for example in training_data.training_examples:
            self.training_message.append(self.get_doc_without_stopword(example))
            
        self.tfidf_vectorizer = TfidfVectorizer(min_df = 1)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.training_message)
        
        for example in training_data.training_examples:
            updated = self._text_features_with_tfidf(example)
            example.set("text_features", updated)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated = self._text_features_with_tfidf(message)
        message.set("text_features", updated)

    def _text_features_with_tfidf(self, message):
        if self.training_message is not None:
            extras = self.features_for_tfidf(message)
            return self._combine_with_existing_text_features(message, extras)
        else:
            return message.get("text_features")

    def features_for_tfidf(self, message):
        """以tf-idf計算句子的特徵
        """

        import numpy as np
        t_doc = []
        t_doc.append(self.get_doc_without_stopword(message))
        tf_idf = self.tfidf_vectorizer.transform(t_doc)
        return np.array(np.ravel(tf_idf.todense()))
		
    def get_doc_without_stopword(self, message):
        """透過spacy_doc分出來的token,再利用is_stop跟is_punct來判斷是否為重要詞,重要詞才需要計算tf-idf,tf-idf是以一段話做為單位,所以需要把token先併成一句
        """
        tokens = []
        for token in message.get("spacy_doc"):
            if not token.is_stop and not token.is_punct:
                tokens.append(token)
        new_doc = ' '.join(token.text for token in tokens)
        logger.info(new_doc)
        return new_doc
    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SklearnIntentClassifier
        import cloudpickle

        if model_dir and model_metadata.get("intent_featurizer_tfidf"):
            tf_idf_file = os.path.join(model_dir, model_metadata.get("intent_featurizer_tfidf"))
            with io.open(tf_idf_file, 'rb') as f:  # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return TfidfFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        tf_idf_file = os.path.join(model_dir, "intent_featurizer_tfidf.pkl")
        with io.open(tf_idf_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_featurizer_tfidf": "intent_featurizer_tfidf.pkl"
        }
