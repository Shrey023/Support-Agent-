def get_ticket_status(ticket_id: int) -> str:
    """Get the status of a customer support ticket.

    Args:
        ticket_id: The ID of the ticket to look up.
    """
    tickets = {
        101: "open",
        102: "closed",
        103: "in progress"
    }

    return tickets.get(ticket_id, "Ticket not found")
