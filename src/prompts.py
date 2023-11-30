BASELINE_PROMPT_WIKI = """Answer the below question which is asking for a list of persons. Output should be a numbered list of maximum 10 persons and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Example Question: Whoe are some movie actors who were born in Boston?
Example Answer: 1. Matt Damon
2. Chris Evans
3. Mark Wahlberg
Example Question: Who are some football players who were born in Madrid?
Example Answer: 1. Sergio Ramos
2. Iker Casillas
3. Fernando Torres
Example Question: Who are some politicians who were born in Washington?
Example Answer: 1. Barack Obama
2. Bill Clinton
3. George Washington

Actual Question: {original_question}

Answer:"""


VERIFICATION_QUESTION_PROMPT_WIKI = """Your task is to create a series of verification questions based on the below question, the verfication question template and baseline response.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Where Was [movie actor] born?
Example Baseline Response: 1. Matt Damon 
2. Chris Evans 
3. Mark Wahlberg
Verification questions: 1. Where was Matt Damon born?
2. Where was Chirs Evans born?
3. Where was Mark Wahlberg born?

Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place) based on the template and substitutes entity values from the baseline response.
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and substitute the entity values from the baseline response to generate verification questions.

Actual Question: {original_question}
Baseline Response: {baseline_response}
Verification Questions:"""

EXECUTE_PLAN_PROMPT = """Answer the following questions. Think step by step and answer each question concisely.
Exampl Questions: 1. Where was Matt Damon born?
2. Where was Chirs Evans born?
3. Where was Mark Wahlberg born?
Example Answers: 1. Cambridge, Massachusetts
2. Sudbury, Massachusetts
3. Boston, Massachusetts

Actual Questions: {verification_questions}

Answers:"""

FINAL_VERIFIED_PROMPT = """Given the below `Original Question` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer. NO ADDITIONAL DETAILS.
Provide the answer as a numbered list of persons.
Example Context: 

Example Original Question: Who are some movie actors who were born in Boston?
Example Baseline Answer: 1. Matt Damon
2. Chris Evans
3. Mark Wahlberg
Example Verification Questions & Answer Pairs From another source: 
1. Where was Matt Damon born?
2. Where was Chirs Evans born?
3. Where was Mark Wahlberg born?
&
1. Cambridge, Massachusetts
2. Sudbury, Massachusetts
3. Boston, Massachusetts
Example Final Refined Answer: 1. Mark Wahlberg

Context:

Actual Original Question: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions} & {verification_answers}

Final Refined Answer:"""