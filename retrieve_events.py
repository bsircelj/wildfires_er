from eventregistry import *
import json
import os
import sys

API_KEY = "bb002e2c-2163-49b4-b40f-fd31da961d2f"
date_start = "2021-05-10"
date_end = "2021-06-10"
manual_query = """
{ 
    "$query": {
        "$and": [
            {"conceptUri": "http://en.wikipedia.org/wiki/Wildfire"},
            {
                "$or": [
                    {"categoryUri": "dmoz/Health/Public_Health_and_Safety/Emergency_Services"},
                    {"categoryUri": "dmoz/Science/Earth_Sciences/Natural_Disasters_and_Hazards"},
                ]
            },
            {
                "dateStart": "%s",
                "dateEnd": "%s",
            },
            {"lang": "eng"},
        ],
        "$not": {
            "$or": [
                {"conceptUri": "http://en.wikipedia.org/wiki/Controlled_burn"},
                {"conceptUri": "http://en.wikipedia.org/wiki/Insurance"},
                {"conceptUri": "http://en.wikipedia.org/wiki/Temperature"},
                {"categoryUri": "news/Technology"},
            ]
        }
    }
}
    """ % (date_start, date_end)


def retrieve_new_events(er):
    q = QueryEventsIter.initWithComplexQuery(manual_query)
    # res = er.execQuery(q)

    retInfo = ReturnInfo(eventInfo=EventInfoFlags(concepts=True, categories=True, stories=True))

    for event in q.execQuery(er, returnInfo=retInfo, maxItems=10):
        print(json.dumps(event, indent=2))
