from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Leave Request Server")

EMPLOYEES_DB = {
    "EMP001": {"name": "Abdullah Elgabry", "department": "Engineering", "balance": {"annual": 21, "sick": 10}},
    "EMP002": {"name": "Laila Zaki", "department": "HR", "balance": {"annual": 18, "sick": 10}},
    "EMP003": {"name": "Amr Mohamed", "department": "Finance", "balance": {"annual": 25, "sick": 10}},
}

LEAVE_REQUESTS_DB = []
PENDING_REQUESTS = {}  # temp storage
REQUEST_COUNTER = 1000


@mcp.tool()
def get_current_date() -> dict:
    """Get today's current date. Use this when user mentions relative dates like 'today', 'tomorrow', '2 days later', 'next week', etc."""
    today = datetime.now()
    return {
        "today": today.strftime("%Y-%m-%d"),
        "day_name": today.strftime("%A"),
        "message": f"Today is {today.strftime('%A, %Y-%m-%d')}"
    }


@mcp.tool()
def check_leave_balance(employee_id: str) -> dict:
    """Check the remaining leave balance for an employee. Returns annual and sick leave balances."""
    if employee_id not in EMPLOYEES_DB:
        return {"error": f"Employee {employee_id} not found"}
    
    emp = EMPLOYEES_DB[employee_id]
    return {
        "employee_id": employee_id,
        "employee_name": emp["name"],
        "balance": emp["balance"],
        "message": f"{emp['name']} has {emp['balance']['annual']} annual and {emp['balance']['sick']} sick leave days remaining."
    }


@mcp.tool()
def prepare_leave_request(employee_id: str, leave_type: str, start_date: str, end_date: str, reason: str) -> dict:
    """
    Prepare a leave request for review. This validates all data and returns a summary.
    The user must then type 'confirm' and you must call confirm_leave_request to actually submit it.
    leave_type must be: annual or sick. Dates in YYYY-MM-DD format.
    """
    global REQUEST_COUNTER
    
    if employee_id not in EMPLOYEES_DB:
        return {"error": f"Employee {employee_id} not found"}
    
    if leave_type not in ["annual", "sick"]:
        return {"error": "leave_type must be: annual or sick"}
    
    emp = EMPLOYEES_DB[employee_id]
    
    # clac days
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days_requested = (end - start).days + 1
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}
    
    if days_requested <= 0:
        return {"error": "End date must be after start date"}
    
    # check balance
    if emp["balance"][leave_type] < days_requested:
        return {"error": f"Insufficient {leave_type} leave. Requested: {days_requested}, Available: {emp['balance'][leave_type]}"}
    
    # create pending request
    REQUEST_COUNTER += 1
    request_id = f"REQ{REQUEST_COUNTER}"
    
    # store in pending requests
    PENDING_REQUESTS[request_id] = {
        "request_id": request_id,
        "employee_id": employee_id,
        "employee_name": emp["name"],
        "leave_type": leave_type,
        "start_date": start_date,
        "end_date": end_date,
        "days": days_requested,
        "reason": reason
    }
    
    return {
        "status": "pending_confirmation",
        "request_id": request_id,
        "summary": {
            "employee": emp["name"],
            "leave_type": leave_type,
            "start_date": start_date,
            "end_date": end_date,
            "days": days_requested,
            "reason": reason,
            "remaining_balance_after": emp["balance"][leave_type] - days_requested
        },
        "message": f"Please review your leave request:\n- Employee: {emp['name']}\n- Type: {leave_type}\n- From: {start_date} to {end_date}\n- Days: {days_requested}\n- Reason: {reason}\n\nType 'confirm' to submit this request."
    }


@mcp.tool()
def confirm_leave_request(request_id: str) -> dict:
    """
    Confirm and submit a prepared leave request. 
    Only call this after the user explicitly types 'confirm'.
    Use the request_id from prepare_leave_request.
    """
    if request_id not in PENDING_REQUESTS:
        return {"error": f"No pending request found with ID {request_id}. Please prepare a new request."}
    
    # get pending request
    request_data = PENDING_REQUESTS.pop(request_id)
    
    # add to confirm requests
    request_data["status"] = "submitted"
    request_data["submitted_at"] = datetime.now().isoformat()
    LEAVE_REQUESTS_DB.append(request_data)
    
    # update employee balance
    emp = EMPLOYEES_DB[request_data["employee_id"]]
    emp["balance"][request_data["leave_type"]] -= request_data["days"]
    
    return {
        "success": True,
        "request_id": request_id,
        "message": f"Leave request {request_id} confirmed and submitted successfully!\n{request_data['days']} days of {request_data['leave_type']} leave from {request_data['start_date']} to {request_data['end_date']}."
    }


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8001)