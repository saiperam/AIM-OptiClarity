# app.py - FastAPI Backend
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import uvicorn
from datetime import datetime, time

app = FastAPI()

# Allow CORS from the frontend (localhost:5173)
origins = [
    "http://localhost:5173",  # Your frontend URL
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Models
class TimeSlot(BaseModel):
    day: str
    start: str  # Format: "09:00am" or "02:30pm"
    end: str  # Format: "09:30am" or "03:30pm"


class PatientTime(BaseModel):
    patient_id: str
    name: str
    availability: List[TimeSlot]


class BlockTime(BaseModel):
    block_id: str
    reason: str
    times: List[TimeSlot]


class ScheduleRequest(BaseModel):
    buffer_time: int  # Buffer time in minutes
    patient_times: Dict[str, List[Dict]]
    block_times: Dict[str, List[Dict]]


class ScheduleResponse(BaseModel):
    schedule: Dict[str, Dict]
    success: bool
    message: str


# Scheduler Logic
def overlap(time1, time2, buffer_time):
    day1, start1, end1 = time1["day"], time1["start"], time1["end"]
    day2, start2, end2 = time2["day"], time2["start"], time2["end"]
    if day1 != day2:
        return False
    # check if actual overlap
    if start1 <= end2 and start1 >= start2:
        return True
    elif end1 <= end2 and end1 >= start2:
        return True
    elif start2 <= end1 and start2 >= start1:
        return True
    elif end2 <= end1 and end2 >= start1:
        return True
    # if no overlap, check if violating buffer time
    if end1 <= start2 and start2 - end1 < buffer_time:
        return True
    elif end2 <= start1 and start1 - end2 < buffer_time:
        return True
    return False


def constraints_satisfied(assigned_variables: dict, buffer_time: int):
    assigned_variable_key_value_pairs = assigned_variables.items()
    for variable1, variable1_domain in assigned_variable_key_value_pairs:
        for variable2, variable2_domain in assigned_variable_key_value_pairs:
            if variable1 != variable2 and overlap(variable1_domain, variable2_domain, buffer_time):
                return False
    return True


def forward_checking(buffer_time, variable_domains, variable_names, curr_uv_index, new_assigned_variable_value):
    updated_variable_domains = dict()
    # Keep assigned variables as having the same domain values
    for i in range(curr_uv_index + 1):
        curr_assigned_variable = variable_names[i][0]
        updated_variable_domains[curr_assigned_variable] = variable_domains[curr_assigned_variable]
    # perform forward checking on unassigned variable domains
    for idx in range(curr_uv_index + 1, len(variable_names)):
        current_unassigned_variable = variable_names[idx][0]
        updated_domain_values_for_curr_variable = []
        for domain_dict_value in variable_domains[current_unassigned_variable]:
            if not overlap(domain_dict_value, new_assigned_variable_value, buffer_time):
                updated_domain_values_for_curr_variable.append(domain_dict_value)
        updated_variable_domains[current_unassigned_variable] = updated_domain_values_for_curr_variable
    return updated_variable_domains


def backtrack(buffer_time: int, variables: dict, assigned_variables: dict, variable_names: list, curr_uv_index: int):
    global assigned_variables_global
    if curr_uv_index >= len(variable_names):
        assigned_variables_global = assigned_variables.copy()
        return True

    curr_unassigned_variable = variable_names[curr_uv_index][0]
    domain_values = variables[curr_unassigned_variable]

    for domain_value_dictionary in domain_values:
        assigned_variables[curr_unassigned_variable] = domain_value_dictionary
        if curr_uv_index == len(variable_names) - 1:
            if constraints_satisfied(assigned_variables, buffer_time):
                assigned_variables_global = assigned_variables.copy()
                return True
        else:
            updated_variable_domains = forward_checking(buffer_time, variables, variable_names, curr_uv_index,
                                                        domain_value_dictionary)
            if all(len(domains) > 0 for domains in updated_variable_domains.values()):
                if backtrack(buffer_time, updated_variable_domains, assigned_variables.copy(), variable_names,
                             curr_uv_index + 1):
                    return True

    return False


def convert_to_24hr_clock(time_str: str) -> int:
    """Convert time like '09:00am' to 900 or '01:30pm' to 1330"""
    if time_str[-2:].lower() == 'am':
        time_str = time_str[:-2]
        hour, minute = map(int, time_str.split(':'))
        if hour == 12:
            hour = 0
    else:  # pm
        time_str = time_str[:-2]
        hour, minute = map(int, time_str.split(':'))
        if hour != 12:
            hour += 12

    return hour * 100 + minute


def convert_from_24hr_clock(time_int: int) -> str:
    """Convert time like 900 to '09:00am' or 1330 to '01:30pm'"""
    hour = time_int // 100
    minute = time_int % 100

    if hour < 12:
        am_pm = 'am'
        if hour == 0:
            hour = 12
    else:
        am_pm = 'pm'
        if hour > 12:
            hour -= 12

    return f"{hour:02d}:{minute:02d}{am_pm}"


def process_schedule_request(request_data):
    buffer_time = request_data["buffer_time"]
    patient_times = request_data["patient_times"]
    block_times = request_data["block_times"]

    day_to_index = {
        "Sunday": 0,
        "Monday": 1,
        "Tuesday": 2,
        "Wednesday": 3,
        "Thursday": 4,
        "Friday": 5,
        "Saturday": 6
    }
    index_to_day = {v: k for k, v in day_to_index.items()}

    variables, assigned_variables, variable_names = dict(), dict(), []

    # Process patient times
    for patient_number, patient_info in patient_times.items():
        temp_info = []
        for slot in patient_info:
            # Convert to internal format
            temp_slot = {
                "day": day_to_index[slot["day"]],
                "start": convert_to_24hr_clock(slot["start"]),
                "end": convert_to_24hr_clock(slot["end"]),
                "original_day": slot["day"],
                "original_start": slot["start"],
                "original_end": slot["end"]
            }
            temp_info.append(temp_slot)
        variables[patient_number] = temp_info
        variable_names.append((patient_number, len(temp_info)))

    # Process block times
    for block_number, block_info in block_times.items():
        temp_info = []
        for slot in block_info:
            # Convert to internal format
            temp_slot = {
                "day": day_to_index[slot["day"]],
                "start": convert_to_24hr_clock(slot["start"]),
                "end": convert_to_24hr_clock(slot["end"]),
                "original_day": slot["day"],
                "original_start": slot["start"],
                "original_end": slot["end"]
            }
            temp_info.append(temp_slot)
        assigned_variables[block_number] = temp_slot  # Block times are pre-assigned

    # Sort by most constrained variable (least number of available slots)
    variable_names = sorted(variable_names, key=lambda t: t[1])

    # Run the scheduling algorithm
    global assigned_variables_global
    assigned_variables_global = None
    success = backtrack(buffer_time, variables, assigned_variables, variable_names, 0)

    if assigned_variables_global:
        # Convert back to user-friendly format
        result = {}
        for patient_id, appointment in assigned_variables_global.items():
            if "original_day" in appointment:
                result[patient_id] = {
                    "day": appointment["original_day"],
                    "start": appointment["original_start"],
                    "end": appointment["original_end"]
                }
            else:
                result[patient_id] = {
                    "day": index_to_day[appointment["day"]],
                    "start": convert_from_24hr_clock(appointment["start"]),
                    "end": convert_from_24hr_clock(appointment["end"])
                }
        return {"schedule": result, "success": True, "message": "Schedule created successfully"}
    else:
        return {"schedule": {}, "success": False, "message": "Could not create a valid schedule with given constraints"}


# Routes
@app.get("/")
def read_root():
    return {"message": "Appointment Scheduler API"}


@app.post("/api/schedule", response_model=ScheduleResponse)
def create_schedule(request: ScheduleRequest):
    try:
        assigned_variables_global = None
        result = process_schedule_request(request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheduling failed: {str(e)}")


@app.get("/api/patients")
def get_patients():
    # In a real app, fetch from database
    return {
        "patients": [
            {"id": "patient1", "name": "John Doe"},
            {"id": "patient2", "name": "Jane Smith"},
            {"id": "patient3", "name": "Bob Johnson"}
        ]
    }


# Global variable for storing result
assigned_variables_global = None

if __name__ == "__main__":
    uvicorn.run("scheduler:scheduler", host="0.0.0.0", port=8006, reload=True)
