from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.emulators import NoEmulator


class DomainswitchEmulator(NoEmulator):
    def __init__(self):
        # type: () -> None

        super(DomainswitchEmulator, self).__init__()
        self.name = "domainswitch"

    def normalise_request_json(self, data, multiple_project=False):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]

        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]

        if not data.get("project"):
            _data["project"] = ["default"] if multiple_project else "defalut"
        elif type(data["project"]) == list:
            _data["project"] = data["project"] if multiple_project else data["project"][0]
        else:
            _data["project"] = [data["project"]] if multiple_project else data["project"]

        if data.get("model"):
            _data["model"] = data["model"][0] if type(data["model"]) == list else data["model"]

        _data['time'] = data["time"] if "time" in data else None
        return _data

    def _ranking(self, data):
        return [{"project": el["project"],"intent": el["intent"], "entities": el["entities"]} for el in data]

    def normalise_response_json(self, data, multiple_project=False):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]
        """Transform data to luis.ai format."""
        if multiple_project == False:
            return data
        ranking = self._ranking(data)
        return {
            "text": data[0]["text"],
            "domain_ranking": ranking
        }
