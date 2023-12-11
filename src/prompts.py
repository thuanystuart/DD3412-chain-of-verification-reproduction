######################################
# WIKIDATA
######################################

BASELINE_PROMPT_WIKI = """Answer the below question which is asking for a list of persons. Output should be a numbered list of maximum 10 persons and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Example Question: Who are some movie actors who were born in Boston?
Example Answer: 1. Donnie Wahlberg
2. Chris Evans
3. Mark Wahlberg
4. Ben Affleck
5. Uma Thurman
Example Question: Who are some football players who were born in Madrid?
Example Answer: 1. Sergio Ramos
2. Marcos Alonso
3. David De Gea
4. Fernando Torres

Example Question: Who are some politicians who were born in Washington?
Example Answer: 1. Barack Obama
2. Bill Clinton
3. Bil Sheffield
4. George Washington

Actual Question: {original_question}"""

########## WIKIDATA JOINT ############

PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI = """Given the below `Original Question` and `Baseline Answer`, create a series of verification questions and answers to finally filter the refined answer. NO ADDITIONAL DETAILS.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Where Was [movie actor] born?
Example Baseline Response: 1. Matt Damon 
2. Chris Evans 
3. Mark Wahlberg
Verification questions and  answers: 1. Where was Matt Damon born? Cambridge, Massachusetts
2. Where was Chirs Evans born? Boston, Massachusetts
3. Where was Mark Wahlberg born? Sudbury, Massachusetts

Original Question: {original_question}

Baseline Response: {baseline_response}"""

FINAL_VERIFIED_JOINT_PROMPT_WIKI = """Given the below `Original Question` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer. NO ADDITIONAL DETAILS.
Provide the answer as a numbered list of persons.
Example Context: 

Example Original Question: Who are some movie actors who were born in Boston?
Example Baseline Answer: 1. Matt Damon
2. Chris Evans
3. Mark Wahlberg
Example Verification Questions & Answer Pairs From another source: 
1. Where was Matt Damon born? Cambridge, Massachusetts
2. Where was Chirs Evans born? Sudbury, Massachusetts
3. Where was Mark Wahlberg born? Boston, Massachusetts
Example Final Refined Answer: 1. Mark Wahlberg

Context:

Actual Original Question: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions_and_answers}"""

######## WIKIDATA TWO-STEP ###########

PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI = """Your task is to create a series of verification questions based on the below question, the verfication question template and baseline response.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Where Was [movie actor] born?
Example Baseline Response: 1. Donnie Wahlberg
2. Chris Evans
3. Mark Wahlberg
4. Ben Affleck
5. Uma Thurman
Verification questions: 1. Where was Donnie Wahlberg born?
2. Where was Chris Evans born?
3. Where was Mark Wahlberg born?
4. Where was Ben Affleck born?
5. Where was Uma Thurman born?

Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place) based on the template and substitutes entity values from the baseline response.
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and substitute the entity values from the baseline response to generate verification questions.

Actual Question: {original_question}
Baseline Response: {baseline_response}"""

EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI = """Answer the following questions. Think step by step and answer each question concisely.
Example Questions: 1. Where was Donnie Wahlberg born?
2. Where was Chirs Evans born?
3. Where was Mark Wahlberg born?
4. Where was Ben Affleck born?
5. Where was Uma Thurman born?
Example Answers: 1. Boston, Massachusetts
2. Boston, Massachusetts
3. Boston, Massachusetts
4. Berkeley, California
5. Boston, Massachusetts

Actual Questions: {verification_questions}"""

FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI = """Given the below `Original Question` and `Baseline Answer`, analyze the `Verification Questions & Answer Pairs` to finally filter the refined answer. NO ADDITIONAL DETAILS.
Provide the answer as a numbered list of persons.
Example Context: 

Example Original Question: Who are some movie actors who were born in Boston?
Example Baseline Answer: 1. Donnie Wahlberg
2. Chris Evans
3. Mark Wahlberg
4. Ben Affleck
5. Uma Thurman
Example Verification Questions & Answer Pairs From another source: 
1. Where was Donnie Wahlberg born?
2. Where was Chirs Evans born?
3. Where was Mark Wahlberg born?
4. Where was Ben Affleck born?
5. Where was Uma Thurman born?
&
1. Boston, Massachusetts
2. Boston, Massachusetts
3. Boston, Massachusetts
4. Berkeley, California
5. Boston, Massachusetts
Example Final Refined Answer: 1. Donnie Wahlberg 
2. Chris Evans
3. Mark Wahlberg
4. Uma Thurman

Example Explanation: Based on the verification questions and answers, only Donnie Wahlberg, Chris Evans, Mark Wahlberg and Uma Thurman were born in Boston. Ben Affleck was born in Berkeley, California. 


Example Original Question: Who are some football players who were born in Madrid?
Example Baseline Answer: 1. Sergio Ramos
2. Marcos Alonso
3. David De Gea
4. Fernando Torres
Example Verification Questions & Answer Pairs From another source:
1. Where was Sergio Ramos born?
2. Where was Marcos Alonso born?
3. Where was David De Gea born?
4. Where was Fernando Torres born?
&
1. Camas, Spain
2. Madrid, Spain
3. Madrid, Spain
4. Fuenlabrada, Spain
Example Final Refined Answer: 1. Marcos Alonso, 2. David De Gea

Example Explanation: Based on the verification questions and answers, only Marcos Alonso and David De Gea were born in Madrid. Sergio Ramos was born in Camas, Spain and Fernando Torres was born in Fuenlabrada, Spain.

Context:

Actual Original Question: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions} & {verification_answers}"""

######## WIKIDATA FACTORED ###########

EXECUTE_VERIFICATION_FACTORED_PROMPT_WIKI = """Answer the following question. Think step by step and answer the question concisely.
Example Question: Where was Donnie Wahlberg born?
Example Answer: Boston, Massachusetts

Example Question: Where was Chirs Evans born?
Example Answer: Boston, Massachusetts

Example Question: Where was Mark Wahlberg born?
Example Answer: Boston, Massachusetts

Example Question: Where was Ben Affleck born?
Example Answer: Berkeley, California

Example Question: Where was Uma Thurman born?
Example Answer: Boston, Massachusetts

Actual Question: {verification_question}"""

######################################
# WIKIDATA CATEGORY
######################################

BASELINE_PROMPT_WIKI_CATEGORY = """Answer the below question which is asking for a list of entities (names, places, locations etc). Output should be a numbered list and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Example Question: Name some movies directed by Steven Spielberg.
Example Answer: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.
5. TENET

Example Question: Name some football stadiums from the Premier League.
Example Answer: 1. Old Trafford
2. Anfield
3. Stamford Bridge
4. Santiago Bernabeu

Question: {original_question}"""

######## WIKICATEGORY JOINT ##########

PLAN_AND_EXECUTION_JOINT_PROMPT_WIKI_CATEGORY = """Given the below `Original Question` and `Baseline Response`, create a series of verification questions and answers to finally filter the refined answer. NO ADDITIONAL DETAILS.
Example Question: Name some movies directed by Steven Spielberg.
Example Verification Question Template: Is [movie] directed by [Steven Spielberg]?
Example Baseline Response: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.
5. TENET
Example Verification Questions and Answers: 1. Is Jaws directed by Steven Spielberg? Yes, Jaws is directed by Steven Spielberg.
2. Is Jurassic Park directed by Steven Spielberg? Yes, Jurassic Park is directed by Steven Spielberg.
3. Is Indiana Jones directed by Steven Spielberg? Yes, Indiana Jones is directed by Steven Spielberg.
4. Is E.T. directed by Steven Spielberg? Yes, E.T. is directed by Steven Spielberg.
5. Is TENET directed by Steven Spielberg? No, TENET is directed by Christopher Nolan.

Original Question: {original_question}
Baseline Response: {baseline_response}"""

FINAL_VERIFIED_JOINT_PROMPT_WIKI_CATEGORY = """Given the below `Original Question` and `Baseline Response`, analyze the `Verification Questions & Answers From another source` to finally filter the refined answer. NO ADDITIONAL DETAILS.
Example Context:

Example Original Question: Name some movies directed by Steven Spielberg.
Example Baseline Response: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.
5. TENET
Example Verification Questions & Answer Pairs From another source:
1. Is Jaws directed by Steven Spielberg? Yes, Jaws is directed by Steven Spielberg.
2. Is Jurassic Park directed by Steven Spielberg? Yes, Jurassic Park is directed by Steven Spielberg.
3. Is Indiana Jones directed by Steven Spielberg? Yes, Indiana Jones is directed by Steven Spielberg.
4. Is E.T. directed by Steven Spielberg? Yes, E.T. is directed by Steven Spielberg.
5. Is TENET directed by Steven Spielberg? No, TENET is directed by Christopher Nolan.
Example Final Refined Answer: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.

Example Explanation: Based on the verification questions and answers, only Jaws, Jurassic Park, Indiana Jones and E.T. are directed by Steven Spielberg. TENET is directed by Christopher Nolan.

Context:

Actual Original Question: {original_question}

Baseline Response: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions_and_answers}"""

####### WIKICATEGORY TWO-STEP ########

PLAN_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY = """Your task is to create a series of verification questions based on the below question, and baseline response.

Example Question: Name some movies directed by Steven Spielberg.
Example Verification Question Template: Is [movie] directed by [Steven Spielberg]?
Example Baseline Response: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.
5. TENET
Example Verification Questions: 1. Is Jaws directed by Steven Spielberg?
2. Is Jurassic Park directed by Steven Spielberg?
3. Is Indiana Jones directed by Steven Spielberg?
4. Is E.T. directed by Steven Spielberg?
5. Is TENET directed by Steven Spielberg?

Example Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie) and QUESTION_ENTITY (director) based on the template and substitutes entity values from the baseline response.

Example question: Name some football stadiums from the Premier League.
Example Verification Question: Is [stadium] a football stadium from the [Premier League]?
Example Baseline Response: 1. Old Trafford
2. Anfield
3. Stamford Bridge
4. Santiago Bernabeu
Example Verification Questions: 1. Is Old Trafford a football stadium from the Premier League?
2. Is Anfield a football stadium from the Premier League?
3. Is Stamford Bridge a football stadium from the Premier League?
4. Is Santiago Bernabeu a football stadium from the Premier League?

Example Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the stadium) and QUESTION_ENTITY (premier league) based on the template and substitutes entity values from the baseline response.

Actual Question: {original_question}
Baseline Response: {baseline_response}"""

EXECUTE_VERIFICATION_TWO_STEP_PROMPT_WIKI_CATEGORY = """Answer the following questions. Think step by step and answer each question concisely.

Example Questions: 1. Is Jaws directed by Steven Spielberg?
2. Is Jurassic Park directed by Steven Spielberg?
3. Is Indiana Jones directed by Steven Spielberg?
4. Is E.T. directed by Steven Spielberg?
5. Is TENET directed by Steven Spielberg?
Example Answers: 1. Yes
2. Yes, Jaws is directed by Steven Spielberg.
3. Yes, Indiana Jones is directed by Steven Spielberg.
4. Yes, E.T. is directed by Steven Spielberg.
5. No, TENET is directed by Christopher Nolan.

Actual Questions: {verification_questions}"""

FINAL_VERIFIED_TWO_STEP_PROMPT_WIKI_CATEGORY = """Given the below `Original Question` and `Baseline Answer`, analyze the `Verification Questions & Answer Pairs` to finally filter the refined answer. NO ADDITIONAL DETAILS.

Example Context:
Example Original Question: Name some movies directed by Steven Spielberg.
Example Baseline Answer: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.
5. TENET
Example Verification Questions & Answer Pairs From another source:
1. Is Jaws directed by Steven Spielberg? Yes, Jaws is directed by Steven Spielberg.
2. Is Jurassic Park directed by Steven Spielberg? Yes, Jurassic Park is directed by Steven Spielberg.
3. Is Indiana Jones directed by Steven Spielberg? Yes, Indiana Jones is directed by Steven Spielberg.
4. Is E.T. directed by Steven Spielberg? Yes, E.T. is directed by Steven Spielberg.
5. Is TENET directed by Steven Spielberg? No, TENET is directed by Christopher Nolan.
Example Final Refined Answer: 1. Jaws
2. Jurassic Park
3. Indiana Jones
4. E.T.

Example Explanation: Based on the verification questions and answers, only Jaws, Jurassic Park, Indiana Jones and E.T. are directed by Steven Spielberg. TENET is directed by Christopher Nolan.

Context:

Actual Original Question: {original_question}
Baseline Response: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions} & {verification_answers}"""

####### WIKICATEGORY FACTORED ########

EXECUTE_VERIFICATION_FACTORED_PROMPT_WIKI_CATEGORY = """Answer the following question. Think step by step and answer the question concisely.

Example Question: Is Jaws directed by Steven Spielberg?
Example Answer: Yes.

Example Question: Is Jurassic Park directed by Steven Spielberg?
Example Answer: Yes, Jaws is directed by Steven Spielberg.

Example Question: Is Indiana Jones directed by Steven Spielberg?
Example Answer: Yes, Indiana Jones is directed by Steven Spielberg.

Example Question: Is E.T. directed by Steven Spielberg?
Example Answer: Yes, E.T. is directed by Steven Spielberg.

Example Question: Is TENET directed by Steven Spielberg?
Example Answer: No, TENET is directed by Christopher Nolan.

Actual Question: {verification_question}"""

######################################
# MULTISPAN QA
######################################

BASELINE_PROMPT_MULTI_QA = """Answer the below question correctly and in a concise manner without much details. Only answer what the question is asked. NO ADDITIONAL DETAILS.

Question: {original_question}"""

######### MULTISPAN QA JOINT ########
PLAN_AND_EXECUTION_JOINT_PROMPT_MULTI_QA = """Given the below `Original Question` and `Baseline Response`, create a series of verification questions and answers to finally filter the refined answer. NO ADDITIONAL DETAILS.
Example Question: Who invented the first printing press and in what year?
Example Baseline Response: Johannes Gutenberg, 1450.
Example Verification Questions and Answers: 1. Did Johannes Gutenberg invent the first printing press? Yes, Johannes Gutenberg invented the first printing press.
2. Did Johannes Gutenberg invent the first printing press in 1450? Yes, Johannes Gutenberg invented the first printing press in 1450.

Original Question: {original_question}
Baseline Response: {baseline_response}"""

FINAL_VERIFIED_JOINT_PROMPT_MULTI_QA = """Given the below `Original Question` and `Baseline Response`, analyze the `Verification Questions & Answers` to finally filter the refined answer. NO ADDITIONAL DETAILS.
Example Context:

Example Original Question: Who invented the first printing press and in what year?
Example Baseline Response: Johannes Gutenberg, 1450.
Example Verification Questions & Answer Pairs From another source:
1. Did Johannes Gutenberg invent the first printing press? Yes, Johannes Gutenberg invented the first printing press.
2. Did Johannes Gutenberg invent the first printing press in 1450? Yes, Johannes Gutenberg invented the first printing press in 1450.
Example Final Refined Answer: Johannes Gutenberg, 1450.

Context:

Actual Original Question: {original_question}

Baseline Response: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions_and_answers}"""

####### MULTISPAN QA TWO-STEP #######

PLAN_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA = """Your task is to create a series of verification questions based on the below original question. The verification questions are meant for verifying the factual acuracy in the baseline response.

Example Question: Who invented the first printing press and in what year?
Example Baseline Response: Johannes Gutenberg, 1450.
Example Verification Questions: 1. Did Johannes Gutenberg invent the first printing press?
2. Did Johannes Gutenberg invent the first printing press in 1450?

Actual Question: {original_question}
Baseline Response: {baseline_response}"""

EXECUTE_VERIFICATION_TWO_STEP_PROMPT_MULTI_QA = """Answer the following questions.
Example Questions: 1. Did Johannes Gutenberg invent the first printing press?
2. Did Johannes Gutenberg invent the first printing press in 1450? 
Example Answers: 1. Yes
2. Yes

Actual Questions: {verification_questions}"""

FINAL_VERIFIED_TWO_STEP_PROMPT_MULTI_QA = """Given the below `Original Question` and `Baseline Answer`, analyze the `Verification Questions & Answer Pairs` to finally filter the refined answer. NO ADDITIONAL DETAILS.
Example Context:

Example Original Question: Who invented the first printing press and in what year?
Example Baseline Answer: Johannes Gutenberg, 1450.
Example Verification Questions & Answer Pairs From another source:
1. Did Johannes Gutenberg invent the first printing press?
2. Did Johannes Gutenberg invent the first printing press in 1450?
&
1. Yes
2. Yes
Example Final Refined Answer: Johannes Gutenberg, 1450.

Context:

Actual Original Question: {original_question}

Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs From another source:
{verification_questions} & {verification_answers}"""

####### MULTISPAN QA FACTORED #######

EXECUTE_VERIFICATION_FACTORED_PROMPT_MULTI_QA = """Answer the following question.

Example Question: Did Johannes Gutenberg invent the first printing press?
Example Answer: Yes.

Example Question: Did Johannes Gutenberg invent the first printing press in 1450? 
Example Answer: Yes.

Actual Questions: {verification_question}"""
