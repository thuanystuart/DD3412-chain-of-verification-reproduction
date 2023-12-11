import argparse
import json
import sys
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

class WikidataQuery:
    def __init__(self, csv_path, sparql_endpoint):
        self.df = pd.read_csv(csv_path, sep=';')
        self.queries = {}
        self.sparql_endpoint = sparql_endpoint

    def generate_queries(self):
        for _, row in self.df.iterrows():
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

    def create_answer_questions(self, output_path: str):
        question_answerings = {}
        for q, v in self.queries.items():
            results = self.get_results(v)
            answers = [result['personLabel']['value'] for result in results["results"]["bindings"]]
            unique_answers = list(set(answers))  # Remove duplicates
            question_answerings[f"Who are some {q[0]} who were born in {q[1]}?"] = unique_answers

        with open(output_path, 'w') as fp:
            json.dump(question_answerings, fp, ensure_ascii=False)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-csv",
        "--csv-path",
        type=str,
        help="Path to the csv file.",
        default="./dataset/source/wikidata_queries.csv",
    )
    argParser.add_argument(
        "-o",
        "--output-path",
        type=str,
        help="Path to the output dataset.",
        default="./dataset/wikidata_dataset.json",
    )
    args = argParser.parse_args()
    
    sparql_endpoint = "https://query.wikidata.org/sparql"

    wikidata_query = WikidataQuery(args.csv_path, sparql_endpoint)
    wikidata_query.generate_queries()
    wikidata_query.create_answer_questions(args.output_path)
