"""
Evaluation metrics for FHIR Agent Benchmark.

Provides retrieval metrics (precision/recall) and LLM-based answer correctness checking.
"""
import asyncio
import logging

import litellm
import numpy as np

logger = logging.getLogger("fhir_common")


def retrieval_recall(pred: list, true: list) -> float:
    """
    Calculate retrieval recall.

    Returns NaN when no ground truth exists (to exclude from averages).
    """
    pred_set = set(pred)

    if len(true) == 0 and len(pred) == 0:
        return 1.0
    if len(true) == 0 and len(pred) > 0:
        return None
    if len(true) > 0 and len(pred) == 0:
        return 0.0

    return np.mean([t in pred_set for t in true])


def retrieval_precision(pred: list, true: list) -> float:
    """
    Calculate retrieval precision.

    Returns NaN when no predictions made (to exclude from averages).
    """
    true_set = set(true)

    if len(true) == 0 and len(pred) == 0:
        return 1.0
    if len(true) == 0 and len(pred) > 0:
        return 0.0
    if len(true) > 0 and len(pred) == 0:
        return None

    return np.mean([p in true_set for p in pred])


NORMALIZE_ANSWER_PROMPT = """You are a helpful assistant that converts free-text answers into a standardized format.

Your task: Determine if the model's answer successfully answers the question or if it indicates insufficient information.

### Instructions:
1. If the answer indicates a FAILURE or ERROR (e.g., "unable to calculate", "execution error", "failed to determine", "encountered an error"), return exactly:
    question answered
    
2. If the answer shows that the question CANNOT be answered (e.g., not enough data, unknown, missing information), return exactly:
    no answer

3. If the answer DOES answer the question, return exactly:
    question answered

Important:
- Return ONLY one of these two values: 'no answer' or 'question answered'.
- Do NOT include any additional text, punctuation, or explanation.

---

### Input:

Question:
{question}

Answer:
{answer}

---

### Examples:

EXAMPLE 1:
Question: Please tell me the sex of patient C  
Answer: The patient's sex is female.  
Return: question answered

---

EXAMPLE 2:
Question: Compared to last measured on the first ICU visit, is the glucose measurement value of patient B less than second to last measured on the first ICU visit?  
Answer: The patient does not have glucose measurements.
Return: no answer

---

EXAMPLE 3:
Question: What is the heart rate of patient C?
Answer: Unable to determine the answer due to execution errors.
Return: question answered

"""


ANSWER_COMPARISON_PROMPT = """You are a helpful assistant that evaluates whether a model answer to a question is correct, by comparing it to the true answer.

Your task:
- Return 1 if the model answer is correct.
- Return 0 if the model answer is incorrect.
- Never return anything other than 0 or 1.

The model answer may be more verbose or formatted differently from the true answer. Focus on correctness of content, not exact formatting.

---

### Core Rules:

1. Yes/No answers 
   - True answers may appear as `[[0]]` (No) or `[[1]]` (Yes), with flexible formatting (e.g., `[0]`, `'[[0]]'` are equivalent).
   - Evaluate based on meaning, not syntax.
   - Ignore differences in variable names or context if the Yes/No meaning aligns.

2. Numerical answers
   - Match on value, rounding both sides to the nearest integer, except for days (which should be rounded to the nearest whole day).
   - Be lenient if the model answer has the true answer in the breakdown but returns a different total aggregated value.
   - Ignore decimal formatting (`1.0` = `1.` = `1`).
   - Ignore units (`1850` = `1850 mL`).

3. Date answers
    - Match on dates, you can ignore time and timezone differences unless specifically stated in the question.
    - A time difference of up to a minute is acceptable, as the values may be rounded.

4. List answers
   - If the true answer is a list (e.g., `['or ebl', 'or urine']`), the model must include all listed values and no extra medical values.  
   - Ignore harmless extra context (e.g., time references, phrasing).

5. Detail and verbosity
   - Extra correct details in the model answer are fine if they align with the true answer.

6. Formatting leniency
   - Be lenient with brackets, quotes, spacing, and style.
   - As long as the model’s answer semantically matches the true answer, return 1.

7. Drug interaction questions
   - Answer has two parts: [list of current medications, interaction status]
   - Expected format: [['med1', 'med2', ...], 'interaction exists'] or [['med1', ...], 'no interactions']
   - Part 1 (medications): Model must list all current medications. Use list matching rules.
   - Part 2 (interaction): Must be 'interaction exists' or 'no interactions'
   - Model may output in exact structured format OR natural language - both are acceptable.
   - If interaction exists, model is correct if it identifies ANY risk, caution, warning, or potential interaction.
   - If no interaction exists, model is correct if its FINAL CONCLUSION states no interactions.
   - Ignore general drug warnings about medications not in the patient's current list.

---

### Examples:

EXAMPLE 1:
- question : Please tell me the sex of patient C  
- true answer : [['F']]  
- model answer : The patient’s sex is female.  
- reasoning : The true answer indicates "female" (F). The model states the patient’s sex is female, which directly matches.  
You will return : 1

---

EXAMPLE 2:  
- question : Compared to last measured on the first icu visit, is the glucose measurement value of patient B less than second to last measured on the first icu visit?  
- true answer : '
- model answer : The patient’s last glucose measurement on the current hospital visit (87 mg/dL) was greater than the first measurement (86 mg/dL).  
- reasoning : The true answer is No (`[[0]]`). The model states the value is greater, which also implies it is not less. This aligns with the true answer meaning.  
You will return : 1

---

EXAMPLE 3:
- question : Are patient A's bicarbonate levels second measured on the first hospital visit less than the levels first measured on the first hospital visit?  
- true answer : [[0]]
- model answer: The second bicarbonate measurement (26 mEq/L at 2161-04-28 02:15) is not less than the first measurement (23 mEq/L at 2161-04-27 12:09).  
- reasoning : The true answer is No (`[[0]]`). The model explicitly says the second value is not less than the first, which matches the true answer.  
You will return : 1

---

EXAMPLE 4:
- question : "Hey, what was patient 10019172's minimum ['fibrinogen, functional'] value in 11/this year?"  
- true answer : [[201.]]  
- model answer: The minimum fibrinogen functional value for patient 10019172 in November of this year is 201.  
- reasoning : The true answer is 201. The model provides the same value (ignoring decimals). They match.  
You will return : 1

---

EXAMPLE 5:  
- question : Show me the total amount of patient's output on this month/02.  
- true answer : [[1850]] 
- model answer: The total patient output on 2110-12-02 was 1850 mL.  
- reasoning : The true answer is 1850. The model answer gives the same value and adds "mL", which is acceptable because units are disregarded.  
You will return : 1

---

EXAMPLE 6:
- question : What was the change in ['platelet count'] in patient 10029484 second measured on the first hospital visit compared to the first value measured on the first hospital visit?
- true answer : [[-38]]
- model answer : The platelet count on the first hospital visit went from 261 on the first measurement to 223 on the second—a drop of 38.
- reasoning : The true answer is -38, indicating a decrease. The model states the count dropped by 38, which matches the true answer meaning.
You will return : 1

---

EXAMPLE 7:
- question : When was patient 10004422's last maximum ['cholesterol, total'] value on their last hospital encounter?
- true answer : [['2111-01-16 07:02:00']]
- model answer : The highest total cholesterol value on the patient’s last hospital encounter was 121 mg/dL, recorded on 2111-01-16.
- reasoning : The true answer is a timestamp. The model provides the same timestamp and mentions the maximum cholesterol value, which aligns with the true answer context. Here there is additional information (the max value) but the answer is still correct. It is okay that the answer does not contain the exact time. 
You will return : 1

---

EXAMPLE 8:
- question : Tell me the total ['acetaminophen'] dose that patient 10037975 has been prescribed?
- true answer : [[1300]]
- model answer : "Total acetaminophen prescribed: 3,300 mg (3.3 g).

Breakdown:
- Acetaminophen IV: 2 prescriptions × 1,000 mg = 2,000 mg  
- Acetaminophen: 2 prescriptions × 650 mg = 1,300 mg"
- reasoning : The true answer only shows the total dose for the basic variation of Acetaminophen, whereas the model provides the total dose for all variations, breaking down the count for the basic one.
You will return : 1

---

EXAMPLE 9:
- question: I'm considering prescribing nifedipine to Patient 10012552. First list their current medications, then tell me if there are any drug interactions.
- true answer: [['docusate sodium (liquid)', 'glucose gel', 'heparin'], 'interaction exists']
- model answer: [['docusate sodium (liquid)', 'glucose gel', 'heparin'], 'interaction exists']
- reasoning: Model provides exact structured format matching the true answer. Both medication list and interaction status match.
You will return: 1

---

EXAMPLE 10:
- question: Patient 10020740 is set to begin furosemide. Does it interact with current medications?
- true answer: [['insulin'], 'interaction exists']
- model answer: Current medications: insulin. Furosemide may affect blood glucose levels and could require adjustment of insulin dosing. Caution is advised.
- reasoning: Model identifies insulin and notes a potential interaction concern. Natural language format is acceptable.
You will return: 1

---

### Final Input:
- Question: {question}
- True answer: {ref_answer}
- Model answer: {answer}

### Output:
Return only 0 or 1. Do not explain your reasoning.
"""


async def normalize_answer(answer: str, question: str, model: str) -> str:
    """Returns 'no answer' or 'question answered'"""
    prompt = NORMALIZE_ANSWER_PROMPT.format(question=question, answer=answer)

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=0,
    )

    result = response.choices[0].message.content.strip().lower()

    if "no answer" in result:
        return "no answer"
    return "question answered"


async def compare_answers(answer: str, ref_answer: str, question: str, model: str) -> int:
    """Compare two answers that both contain data."""
    prompt = ANSWER_COMPARISON_PROMPT.format(
        question=question,
        ref_answer=ref_answer,
        answer=answer,
    )

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result = response.choices[0].message.content.strip()

    if result in ["0", "1"]:
        return int(result)

    logger.warning(f"Unexpected LLM response: {result}, defaulting to 0")
    return 0


def is_null_answer(ref_answer: str) -> bool:
    null_values = ['[]', '[[]]', '[[None]]', 'null', "['null']", '[[null]]', "['None']", '[None]']
    return ref_answer.strip() in null_values


async def check_answer_correctness(
    answer: str,
    ref_answer: str,
    question: str,
    model: str,
    true_fhir_ids: dict | None = None,
) -> int:
    """Two-step answer correctness check."""

    # Step 1: Normalize agent answer
    agent_normalized = await normalize_answer(answer, question, model)
    true_is_null = is_null_answer(ref_answer)

    # Step 2: Apply logic
    if agent_normalized == "no answer":
        if true_is_null:
            return 1
        # Special case: boolean with no resources - accept "no data" as valid
        if true_fhir_ids is not None and not true_fhir_ids and ref_answer == "[[0]]":
            return 1
        return 0

    if true_is_null:
        return 0  # agent gave answer but no data should exist

    # Both have answers - compare
    return await compare_answers(answer, ref_answer, question, model)