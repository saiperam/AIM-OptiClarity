import React, { useState, useEffect } from "react";
import { Calendar, momentLocalizer } from "react-big-calendar";
import moment from "moment";
import "react-big-calendar/lib/css/react-big-calendar.css";

// Initialize the calendar localizer
const localizer = momentLocalizer(moment);

export default function AppointmentScheduler() {
  const [view, setView] = useState("schedule"); // schedule, patients, blocks
  const [patients, setPatients] = useState([]);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [notification, setNotification] = useState(null);
  const [scheduleData, setScheduleData] = useState({
    buffer_time: 15, // 15 minutes default
    patient_times: {},
    block_times: {},
  });
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedDateEvents, setSelectedDateEvents] = useState([]);

  // Fetch patients on component mount
  useEffect(() => {
    // For demo, using mock data
    setPatients([
      { id: "patient1", name: "John Doe" },
      { id: "patient2", name: "Jane Smith" },
      { id: "patient3", name: "Bob Johnson" },
    ]);
  }, []);

  // Update calendar events whenever block times or patient availability changes
  useEffect(() => {
    updateCalendarEvents();
  }, [scheduleData.block_times, scheduleData.patient_times]);

  const parseTimeToDate = (day, timeStr) => {
    // Parse day and time strings to create a Date object
    const dayMapping = {
      Sunday: 0,
      Monday: 1,
      Tuesday: 2,
      Wednesday: 3,
      Thursday: 4,
      Friday: 5,
      Saturday: 6,
    };

    const today = new Date();
    const dayOfWeek = today.getDay(); // 0 = Sunday, 6 = Saturday
    const daysUntilAppointment = (7 + dayMapping[day] - dayOfWeek) % 7;

    const appointmentDate = new Date(today);
    appointmentDate.setDate(today.getDate() + daysUntilAppointment);

    // Parse time
    const [hour, minute] = timeStr.slice(0, -2).split(":");
    const amPm = timeStr.slice(-2);
    let hour24 = parseInt(hour);
    if (amPm.toLowerCase() === "pm" && hour24 < 12) hour24 += 12;
    if (amPm.toLowerCase() === "am" && hour24 === 12) hour24 = 0;

    const result = new Date(appointmentDate);
    result.setHours(hour24, parseInt(minute), 0);

    return result;
  };

  const updateCalendarEvents = () => {
    const newEvents = [];

    // Convert block times to calendar events
    for (const [blockId, times] of Object.entries(scheduleData.block_times)) {
      times.forEach((blockTime) => {
        const start = parseTimeToDate(blockTime.day, blockTime.start);
        const end = parseTimeToDate(blockTime.day, blockTime.end);

        newEvents.push({
          title: `BLOCKED: ${blockId.replace("block_", "").replace(/_/g, " ")}`,
          start,
          end,
          blockId,
          isBlockTime: true,
        });
      });
    }

    // Convert patient availability to calendar events
    for (const [patientId, availabilityTimes] of Object.entries(
      scheduleData.patient_times
    )) {
      const patientData = patients.find((p) => p.id === patientId) || {
        name: patientId,
      };

      availabilityTimes.forEach((availability) => {
        const start = parseTimeToDate(availability.day, availability.start);
        const end = parseTimeToDate(availability.day, availability.end);

        newEvents.push({
          title: `AVAILABLE: ${patientData.name}`,
          start,
          end,
          patientId,
          isAvailability: true,
        });
      });
    }

    // Keep any existing patient appointment events that aren't block times or availability
    const appointmentEvents = events.filter(
      (event) => !event.isBlockTime && !event.isAvailability
    );

    // Update events with block times, availability, and appointments
    setEvents([...appointmentEvents, ...newEvents]);
  };

  const generateSchedule = async () => {
    setLoading(true);
    try {
      // In production, replace with your API endpoint
      const response = await fetch("http://localhost:8000/api/schedule", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(scheduleData),
      });

      const result = await response.json();

      if (result.success) {
        // Convert the schedule to calendar events
        const newEvents = [];
        for (const [patientId, appointment] of Object.entries(
          result.schedule
        )) {
          if (patientId.startsWith("block_")) continue; // Skip block times

          const patientData = patients.find((p) => p.id === patientId) || {
            name: patientId,
          };
          const start = parseTimeToDate(appointment.day, appointment.start);
          const end = parseTimeToDate(appointment.day, appointment.end);

          newEvents.push({
            title: `APPOINTMENT: ${patientData.name}`,
            start,
            end,
            patientId,
            isAppointment: true,
          });
        }

        // Filter out events that are block times or availability but keep other appointments
        const otherEvents = events.filter(
          (event) => !event.isBlockTime && !event.isAvailability
        );

        // Update the calendar
        setEvents([...otherEvents, ...newEvents]);

        // Re-run updateCalendarEvents to add back block times and availabilities
        updateCalendarEvents();

        setNotification({
          type: "success",
          message: "Schedule created successfully!",
        });
      } else {
        setNotification({ type: "error", message: result.message });
      }
    } catch (error) {
      console.error("Error generating schedule:", error);
      setNotification({
        type: "error",
        message: "Failed to generate schedule",
      });
    } finally {
      setLoading(false);
    }
  };

  const checkForOverlaps = (newTime) => {
    // Check if new availability overlaps with any block times
    for (const blockTimes of Object.values(scheduleData.block_times)) {
      for (const blockTime of blockTimes) {
        if (blockTime.day === newTime.day) {
          const newStart = convertTimeStringToMinutes(newTime.start);
          const newEnd = convertTimeStringToMinutes(newTime.end);
          const blockStart = convertTimeStringToMinutes(blockTime.start);
          const blockEnd = convertTimeStringToMinutes(blockTime.end);

          // Check for overlap
          if (
            (newStart <= blockEnd && newStart >= blockStart) ||
            (newEnd <= blockEnd && newEnd >= blockStart) ||
            (blockStart <= newEnd && blockStart >= newStart) ||
            (blockEnd <= newEnd && blockEnd >= newStart)
          ) {
            return true;
          }
        }
      }
    }
    return false;
  };

  const convertTimeStringToMinutes = (timeStr) => {
    const [hourStr, rest] = timeStr.split(":");
    const minuteStr = rest.slice(0, 2);
    const amPm = rest.slice(2);

    let hour = parseInt(hourStr);
    if (amPm.toLowerCase() === "pm" && hour < 12) hour += 12;
    if (amPm.toLowerCase() === "am" && hour === 12) hour = 0;

    return hour * 60 + parseInt(minuteStr);
  };

  const addPatientAvailability = (patientId, availability) => {
    // Check if this availability overlaps with any block times
    if (checkForOverlaps(availability)) {
      setNotification({
        type: "error",
        message:
          "This availability overlaps with a blocked time period. Please choose another time.",
      });
      return false;
    }

    setScheduleData((prev) => ({
      ...prev,
      patient_times: {
        ...prev.patient_times,
        [patientId]: [...(prev.patient_times[patientId] || []), availability],
      },
    }));
    return true;
  };

  const addBlockTime = (blockId, blockTime) => {
    setScheduleData((prev) => ({
      ...prev,
      block_times: {
        ...prev.block_times,
        [blockId]: [...(prev.block_times[blockId] || []), blockTime],
      },
    }));
  };

  const handleSetBufferTime = (minutes) => {
    setScheduleData((prev) => ({
      ...prev,
      buffer_time: parseInt(minutes),
    }));
  };

  const eventStyleGetter = (event) => {
    if (event.isBlockTime) {
      return {
        style: {
          backgroundColor: "#EF4444", // Red for blocked times
          color: "white",
          borderRadius: "4px",
        },
      };
    } else if (event.isAvailability) {
      return {
        style: {
          backgroundColor: "#10B981", // Green for available times
          color: "white",
          borderRadius: "4px",
        },
      };
    }
    return {
      style: {
        backgroundColor: "#3B82F6", // Blue for appointments
        color: "white",
        borderRadius: "4px",
      },
    };
  };

  const handleDateClick = (date) => {
    setSelectedDate(date);
    const dayStart = new Date(date);
    dayStart.setHours(0, 0, 0, 0);
    const dayEnd = new Date(date);
    dayEnd.setHours(23, 59, 59, 999);

    const dayEvents = events.filter((event) => {
      return (
        (event.start >= dayStart && event.start <= dayEnd) ||
        (event.end >= dayStart && event.end <= dayEnd)
      );
    });

    setSelectedDateEvents(dayEvents);
  };

  const formatTime = (date) => {
    return moment(date).format("h:mm A");
  };

  return (
    <div className="space-y-6 sm:mx-16 my-8">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <h2
          className="sm:text-3xl text-2xl font-bold text-center text-blue-900"
          data-aos="fade-right"
          data-aos-delay="300"
        >
          Appointment Scheduler
        </h2>
        <span className="block w-16 h-1 bg-gradient-to-r from-blue-300 to-blue-900 mx-auto mt-4 rounded-full"></span>
      </div>
      <div className="flex sm:space-x-4 sm:text-base text-sm space-x-2 px-4 sm:px-0">
        <button
          className={`px-4 py-2 rounded ${
            view === "schedule" ? "bg-blue-800 text-white" : "bg-gray-200"
          }`}
          onClick={() => setView("schedule")}
        >
          Calendar View
        </button>
        <button
          className={`px-4 py-2 rounded ${
            view === "patients" ? "bg-blue-800 text-white" : "bg-gray-200"
          }`}
          onClick={() => setView("patients")}
        >
          Patient Availability
        </button>
        <button
          className={`px-4 py-2 rounded ${
            view === "blocks" ? "bg-blue-800 text-white" : "bg-gray-200"
          }`}
          onClick={() => setView("blocks")}
        >
          Block Times
        </button>
      </div>

      {notification && (
        <div
          className={`p-4 rounded ${
            notification.type === "success"
              ? "bg-green-100 text-green-800"
              : "bg-red-100 text-red-800"
          }`}
        >
          {notification.message}
          <button
            className="ml-2 text-sm"
            onClick={() => setNotification(null)}
          >
            ×
          </button>
        </div>
      )}

      {view === "schedule" && (
        <div className="space-y-4 px-4">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Appointment Schedule</h2>
            <div className="flex sm:flex-row flex-col items-start space-x-2 space-y-2 sm:space-y-0 sm:items-center">
              <div className="flex items-center ml-2">
                <div className="w-4 h-4 bg-red-500 rounded mr-1"></div>
                <span className="text-sm text-gray-600">Blocked</span>
              </div>
              <div className="flex items-center sm:ml-2">
                <div className="w-4 h-4 bg-green-500 rounded mr-1"></div>
                <span className="text-sm text-gray-600">Available</span>
              </div>
              <div className="flex items-center sm:ml-2">
                <div className="w-4 h-4 bg-blue-500 rounded mr-1"></div>
                <span className="text-sm text-gray-600">Appointment</span>
              </div>
              <button
                className="sm:text-base text-sm sm:px-4 sm:py-2 px-2 py-2 bg-blue-800 text-white rounded hover:bg-blue-600 transition sm:ml-4"
                onClick={generateSchedule}
                disabled={loading}
              >
                {loading ? "Generating..." : "Generate Schedule"}
              </button>
            </div>
          </div>

          <div className="bg-white p-4 rounded shadow-md">
            <div style={{ height: "600px" }}>
              <Calendar
                localizer={localizer}
                events={events}
                startAccessor="start"
                endAccessor="end"
                eventPropGetter={eventStyleGetter}
                style={{ height: "100%" }}
                onNavigate={(date) => setSelectedDate(null)}
                onSelectSlot={(slotInfo) => handleDateClick(slotInfo.start)}
                onSelectEvent={(event) => handleDateClick(event.start)} //added later on
                selectable
                views={["month", "week", "day", "agenda"]}
                defaultView="month"
              />
            </div>
          </div>

          {selectedDate && (
            <div className="bg-white p-6 rounded shadow-md">
              <h3 className="text-lg font-semibold mb-4">
                Schedule for {moment(selectedDate).format("MMMM D, YYYY")}
              </h3>

              {selectedDateEvents.length === 0 ? (
                <p className="text-gray-500">
                  No events scheduled for this day.
                </p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Type
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Title
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Time
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Duration
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {selectedDateEvents.map((event, index) => (
                        <tr
                          key={index}
                          className={
                            index % 2 === 0 ? "bg-white" : "bg-gray-50"
                          }
                        >
                          <td className="px-6 py-4 whitespace-nowrap">
                            {event.isBlockTime ? (
                              <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                                Blocked
                              </span>
                            ) : event.isAvailability ? (
                              <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                                Available
                              </span>
                            ) : (
                              <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                                Appointment
                              </span>
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {event.title.replace(/^[^:]+: /, "")}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {formatTime(event.start)} - {formatTime(event.end)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {moment
                              .duration(
                                moment(event.end).diff(moment(event.start))
                              )
                              .asMinutes()}{" "}
                            minutes
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {view === "patients" && (
        <PatientAvailability
          patients={patients}
          addAvailability={addPatientAvailability}
          currentAvailability={scheduleData.patient_times}
        />
      )}

      {view === "blocks" && (
        <BlockTimeManager
          addBlockTime={addBlockTime}
          currentBlockTimes={scheduleData.block_times}
          bufferTime={scheduleData.buffer_time}
          setBufferTime={handleSetBufferTime}
        />
      )}
    </div>
  );
}

function PatientAvailability({
  patients,
  addAvailability,
  currentAvailability,
}) {
  const [selectedPatient, setSelectedPatient] = useState("");
  const [day, setDay] = useState("Monday");
  const [startTime, setStartTime] = useState("09:00am");
  const [endTime, setEndTime] = useState("09:30am");
  const [formError, setFormError] = useState("");

  const daysOfWeek = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ];
  const timeOptions = [
    "08:00am",
    "08:30am",
    "09:00am",
    "09:30am",
    "10:00am",
    "10:30am",
    "11:00am",
    "11:30am",
    "12:00pm",
    "12:30pm",
    "01:00pm",
    "01:30pm",
    "02:00pm",
    "02:30pm",
    "03:00pm",
    "03:30pm",
    "04:00pm",
    "04:30pm",
    "05:00pm",
    "05:30pm",
    "06:00pm",
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    setFormError("");

    if (!selectedPatient) {
      setFormError("Please select a patient");
      return;
    }

    // Convert times to validate end time is after start time
    const startTimeVal = timeToMinutes(startTime);
    const endTimeVal = timeToMinutes(endTime);

    if (endTimeVal <= startTimeVal) {
      setFormError("End time must be after start time");
      return;
    }

    const success = addAvailability(selectedPatient, {
      day,
      start: startTime,
      end: endTime,
    });

    if (success) {
      // Reset form after successful submission
      setStartTime("09:00am");
      setEndTime("09:30am");
    }
  };

  const timeToMinutes = (timeStr) => {
    const [hour, minuteAmPm] = timeStr.split(":");
    const minute = minuteAmPm.substring(0, 2);
    const amPm = minuteAmPm.substring(2);

    let hourVal = parseInt(hour);
    if (amPm === "pm" && hourVal !== 12) hourVal += 12;
    if (amPm === "am" && hourVal === 12) hourVal = 0;

    return hourVal * 60 + parseInt(minute);
  };

  return (
    <div className="space-y-6 sm:mx-16 my-8 px-4">
      <h2 className="text-xl font-semibold">Patient Availability</h2>

      <div className="bg-white p-6 rounded shadow-md">
        <form onSubmit={handleSubmit} className="space-y-4">
          {formError && (
            <div className="p-3 bg-red-100 text-red-700 rounded">
              {formError}
            </div>
          )}

          <div>
            <label className="block text-gray-700 mb-2">Patient</label>
            <select
              className="w-full p-2 border rounded"
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              required
            >
              <option value="">Select Patient</option>
              {patients.map((patient) => (
                <option key={patient.id} value={patient.id}>
                  {patient.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Day</label>
            <select
              className="w-full p-2 border rounded"
              value={day}
              onChange={(e) => setDay(e.target.value)}
            >
              {daysOfWeek.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 mb-2">Start Time</label>
              <select
                className="w-full p-2 border rounded"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
              >
                {timeOptions.map((time) => (
                  <option key={time} value={time}>
                    {time}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-gray-700 mb-2">End Time</label>
              <select
                className="w-full p-2 border rounded"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
              >
                {timeOptions.map((time) => (
                  <option key={time} value={time}>
                    {time}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            className="w-full py-2 bg-blue-800 text-white rounded hover:bg-blue-600 transition"
          >
            Add Availability
          </button>
        </form>
      </div>

      <div className="bg-white p-6 rounded shadow-md">
        <h3 className="text-lg font-medium mb-4">Current Availability</h3>

        {Object.entries(currentAvailability).length === 0 ? (
          <p className="text-gray-500">No availability added yet.</p>
        ) : (
          <div className="space-y-4">
            {Object.entries(currentAvailability).map(([patientId, times]) => {
              const patient = patients.find((p) => p.id === patientId) || {
                name: patientId,
              };

              return (
                <div key={patientId} className="border-b pb-4">
                  <h4 className="font-medium">{patient.name}</h4>
                  <ul className="mt-2 space-y-1">
                    {times.map((time, index) => (
                      <li key={index} className="text-sm">
                        {time.day} {time.start} - {time.end}
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function BlockTimeManager({
  addBlockTime,
  currentBlockTimes,
  bufferTime,
  setBufferTime,
}) {
  const [blockName, setBlockName] = useState("lunch");
  const [day, setDay] = useState("Monday");
  const [startTime, setStartTime] = useState("12:00pm");
  const [endTime, setEndTime] = useState("01:00pm");
  const [formError, setFormError] = useState("");

  const daysOfWeek = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
  ];
  const timeOptions = [
    "08:00am",
    "08:30am",
    "09:00am",
    "09:30am",
    "10:00am",
    "10:30am",
    "11:00am",
    "11:30am",
    "12:00pm",
    "12:30pm",
    "01:00pm",
    "01:30pm",
    "02:00pm",
    "02:30pm",
    "03:00pm",
    "03:30pm",
    "04:00pm",
    "04:30pm",
    "05:00pm",
    "05:30pm",
    "06:00pm",
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    setFormError("");

    // Convert times to validate end time is after start time
    const startTimeVal = timeToMinutes(startTime);
    const endTimeVal = timeToMinutes(endTime);

    if (endTimeVal <= startTimeVal) {
      setFormError("End time must be after start time");
      return;
    }

    const blockId = `block_${blockName.replace(/\s+/g, "_").toLowerCase()}`;
    addBlockTime(blockId, {
      day,
      start: startTime,
      end: endTime,
    });

    // Reset form after submission
    setBlockName("lunch");
    setStartTime("12:00pm");
    setEndTime("01:00pm");
  };

  const timeToMinutes = (timeStr) => {
    const [hour, minuteAmPm] = timeStr.split(":");
    const minute = minuteAmPm.substring(0, 2);
    const amPm = minuteAmPm.substring(2);

    let hourVal = parseInt(hour);
    if (amPm === "pm" && hourVal !== 12) hourVal += 12;
    if (amPm === "am" && hourVal === 12) hourVal = 0;

    return hourVal * 60 + parseInt(minute);
  };

  return (
    <div className="space-y-6 sm:mx-16 my-8 px-4">
      <h2 className="text-xl font-semibold">Block Times & Buffer Settings</h2>

      <div className="bg-white p-6 rounded shadow-md">
        <div className="mb-6">
          <label className="block text-gray-700 mb-2">
            Buffer Time Between Appointments (minutes)
          </label>
          <input
            type="number"
            className="w-full p-2 border rounded"
            value={bufferTime}
            onChange={(e) => setBufferTime(e.target.value)}
            min="0"
            max="60"
          />
          <p className="text-sm text-gray-500 mt-1">
            Minimum time required between any two appointments
          </p>
        </div>

        <h3 className="text-lg font-medium mb-4">Add Blocked Time</h3>
        <form onSubmit={handleSubmit} className="space-y-4">
          {formError && (
            <div className="p-3 bg-red-100 text-red-700 rounded">
              {formError}
            </div>
          )}

          <div>
            <label className="block text-gray-700 mb-2">Block Reason</label>
            <input
              type="text"
              className="w-full p-2 border rounded"
              value={blockName}
              onChange={(e) => setBlockName(e.target.value)}
              placeholder="Lunch, Meeting, etc."
              required
            />
          </div>

          <div>
            <label className="block text-gray-700 mb-2">Day</label>
            <select
              className="w-full p-2 border rounded"
              value={day}
              onChange={(e) => setDay(e.target.value)}
            >
              {daysOfWeek.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 mb-2">Start Time</label>
              <select
                className="w-full p-2 border rounded"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
              >
                {timeOptions.map((time) => (
                  <option key={time} value={time}>
                    {time}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-gray-700 mb-2">End Time</label>
              <select
                className="w-full p-2 border rounded"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
              >
                {timeOptions.map((time) => (
                  <option key={time} value={time}>
                    {time}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <button
            type="submit"
            className="w-full py-2 bg-blue-800 text-white rounded hover:bg-blue-600 transition"
          >
            Add Block Time
          </button>
        </form>
      </div>

      <div className="bg-white p-6 rounded shadow-md">
        <h3 className="text-lg font-medium mb-4">Current Block Times</h3>

        {Object.entries(currentBlockTimes).length === 0 ? (
          <p className="text-gray-500">No block times added yet.</p>
        ) : (
          <div className="space-y-4">
            {Object.entries(currentBlockTimes).map(([blockId, times]) => (
              <div key={blockId} className="border-b pb-4">
                <h4 className="font-medium">
                  Block: {blockId.replace("block_", "").replace(/_/g, " ")}
                </h4>
                <ul className="mt-2 space-y-1">
                  {times.map((time, index) => (
                    <li key={index} className="text-sm">
                      {time.day} {time.start} - {time.end}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
