from tools import get_ticket_status

MAX_STEPS = 3
ticket_id = 102

for step in range(MAX_STEPS):
    print("Agent step:", step + 1)

    status = get_ticket_status(ticket_id)
    print("Ticket status:", status)

    print("Task completed!")
    break
