
from tools import get_ticket_status
from agent import is_ticket_status_question, extract_ticket_id


def test_ticket_status():
    assert get_ticket_status(101) == "open"
    assert get_ticket_status(102) == "closed"
    assert get_ticket_status(103) == "in progress"
    assert get_ticket_status(999) == "Ticket not found"


def test_ticket_status_question_detection():
    assert is_ticket_status_question(
        "What is the status of ticket 101?"
    ) is True

    assert is_ticket_status_question(
        "What is the status of my ticket?"
    ) is True

    assert is_ticket_status_question(
        "My ticket has been delayed. What should I do?"
    ) is False


def test_ticket_id_extraction():
    assert extract_ticket_id(
        "What is the status of ticket 101?"
    ) == 101

    assert extract_ticket_id(
        "Check ticket ID #102"
    ) == 102

    assert extract_ticket_id(
        "What should I do about my delayed ticket?"
    ) is None


if __name__ == "__main__":
    test_ticket_status()
    test_ticket_status_question_detection()
    test_ticket_id_extraction()

    print("All tests passed!")
