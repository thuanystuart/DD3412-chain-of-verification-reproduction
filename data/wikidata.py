import pandas as pd
import json
import sys
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataQuery:
    def __init__(self, csv_path, sparql_endpoint):
        self.df = pd.read_csv(csv_path, sep=';')
        self.queries = {}
        self.sparql_endpoint = sparql_endpoint

    def generate_queries(self):
        for index, row in self.df.iterrows():
            query = f"""
            SELECT DISTINCT ?person ?personLabel ?birthdate ?birthplace ?birthplaceLabel ?sitelinks
            WHERE {{
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
                ?person wdt:P106 wd:{row['profession_code']};  
                wdt:P19 ?birthplace.
                ?birthplace (wdt:P131*) wd:{row['city_code']}.
                OPTIONAL {{
                    ?person wdt:P569 ?birthdate. 
                }}

                OPTIONAL {{
                    ?person wikibase:sitelinks ?sitelinks.
                }}
            }}
            ORDER BY DESC (?sitelinks)
            LIMIT 600"""
            self.queries[row['profession'], row['city']] = query

    def get_results(self, query):
        user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
        sparql = SPARQLWrapper(self.sparql_endpoint, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        return sparql.query().convert()

    def create_answer_questions(self):
        question_answerings = {}
        for q, v in self.queries.items():
            results = self.get_results(v)
            answers = [result['personLabel']['value'] for result in results["results"]["bindings"]]
            unique_answers = list(set(answers))  # Remove duplicates
            question_answerings[f"Who are some {q[0]} who were born in {q[1]}?"] = unique_answers

        with open('wikidata_questions.json', 'w') as fp:
            json.dump(question_answerings, fp, ensure_ascii=False)

if __name__ == "__main__":
    csv_path = "PATH_TO_THE_CSV_FILE"
    sparql_endpoint = "https://query.wikidata.org/sparql"

    wikidata_query = WikidataQuery(csv_path, sparql_endpoint)
    wikidata_query.generate_queries()
    wikidata_query.create_answer_questions()
