BASELINE_PROMPT_WIKI = """Answer the below question which is asking for a list of persons. Output should be a numbered list and only contains the relevant & concise enitites as answer. NO ADDITIONAL DETAILS.

Question: {original_question}

Answer:"""


VERIFICATION_QUESTION_PROMPT_WIKI = """Your task is to create a series of verification questions based on the below question, the verfication question template and baseline response.
Example Question: Who are some movie actors who were born in Boston?
Example Verification Question Template: Was [movie actor] born in Boston?
Example Baseline Response: 1. Matt Damon 
2. Chris Evans 
Verification questions: 1. Where was Matt Damon born?
2. Where was Chirs Evans born?

Explanation: In the above example the verification questions focused only on the ANSWER_ENTITY (name of the movie actor) and QUESTION_ENTITY (birth place) based on the template and substitutes entity values from the baseline response.
Similarly you need to focus on the ANSWER_ENTITY and QUESTION_ENTITY from the actual question and substitute the entity values from the baseline response to generate verification questions.

Actual Question: {original_question}
Baseline Response: {baseline_response}
Verification Questions:"""

EXECUTE_PLAN_PROMPT = """Answer the following questions. Think step by step and answer each question concisely.

Questions: {verification_questions}

Answers:"""

FINAL_VERIFIED_PROMPT = """Given the below `Original Query` and `Baseline Answer`, analyze the `Verification Questions & Answers` to finally filter the refined answer.
Original Query: {original_question}
Baseline Answer: {baseline_response}

Verification Questions & Answer Pairs:
{verification_questions} & {verification_answers}

Final Refined Answer:"""