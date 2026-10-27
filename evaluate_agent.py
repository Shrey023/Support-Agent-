import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from agent import handle_customer_question, load_support_knowledge


def evaluate_rag_answer(actual_answer, test_case):
    answer_lower = actual_answer.lower()

    required_phrases = test_case.get("required_phrases", [])

    return all(
        phrase.lower() in answer_lower
        for phrase in required_phrases
    )

def main():
    load_dotenv()

    client = genai.Client()

    # Load our predefined evaluation questions.
    cases_path = Path(__file__).parent / "evaluation_cases.json"

    with open(cases_path, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    # Build the knowledge base once for all tests.
    knowledge_base = load_support_knowledge()

    passed = 0
    failed = 0

    print("\n=== Support Agent Evaluation ===")

    for index, case in enumerate(test_cases, start=1):
        question = case["question"]
        expected_answer = case["expected_answer"]
        test_type = case["test_type"]

        print(f"\nTest {index}: {question}")

        try:
            actual_answer = handle_customer_question(
                client,
                question,
                knowledge_base
            )

            if test_type == "ticket_status":
                # Ticket-status answers should match exactly.
                success = actual_answer.strip() == expected_answer.strip()

            elif test_type == "rag":
                success = evaluate_rag_answer(
                    actual_answer,
                    case
                )

            else:
                print(f"Unknown test type: {test_type}")
                failed += 1
                continue

            print(f"Expected: {expected_answer}")
            print(f"Actual:   {actual_answer}")

            if success is True:
                print("Result: PASS")
                passed += 1

            elif success is False:
                print("Result: FAIL")
                failed += 1

            else:
                print("Result: REVIEW REQUIRED")
                print("Check whether the answer follows the support policy.")

        except Exception as error:
            print(f"Result: ERROR ({error})")
            failed += 1

    print("\n=== Evaluation Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("RAG cases require manual review in this version.")


if __name__ == "__main__":
    main()