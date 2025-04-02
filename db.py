import sqlite3
from twilio.rest import Client

with open("env_variables.txt", 'r') as file:
    RECOVERY_CODE, AUTH_TOKEN, ACCOUNT_SID = tuple(file.readlines())
    RECOVERY_CODE = RECOVERY_CODE[:-1]
    AUTH_TOKEN = AUTH_TOKEN[:-1]
    ACCOUNT_SID = ACCOUNT_SID
    

connection = sqlite3.connect("patients.db")
cursor = connection.cursor()
client = Client(ACCOUNT_SID, AUTH_TOKEN)

def add_doctor(fname, lname, ssn, email, phone):
    cursor.execute('''
        INSERT INTO DOCTORS (ssn, fname, lname, email, phone) VALUES (?, ?, ?, ?, ?)''', (ssn, fname, lname, email, phone)
    )
    connection.commit()


def delete_doctor(ssn):
    cursor.execute('''SELECT EXISTS (
        SELECT 1 FROM DOCTORS WHERE ssn = ?)''', (ssn,))
    result = cursor.fetchone()[0]
    if result == 1:
        cursor.execute("DELETE FROM DOCTORS WHERE ssn = ?", (ssn,))
        connection.commit()
    else:
        print("Could not find a doctor with the given ssn within the database")


def add_patient(fname, lname, ssn, email, phone):
    cursor.execute('''
        INSERT INTO PATIENTS (ssn, fname, lname, email, phone) VALUES (?, ?, ?, ?, ?)''', (ssn, fname, lname, email, phone)
    )
    connection.commit()


def delete_patient(ssn):
    cursor.execute('''SELECT EXISTS (
        SELECT 1 FROM PATIENTS WHERE ssn = ?)''', (ssn,))
    result = cursor.fetchone()[0]
    if result == 1:
        cursor.execute("DELETE FROM PATIENTS WHERE ssn = ?", (ssn,))
        connection.commit()
    else:
        print("Could not find a patient with the given ssn within the database")


def display_patients():
    cursor.execute('''SELECT * FROM PATIENTS''')
    patients = cursor.fetchall()
    if patients:
        for patient_row in patients:
            print(f'''SSN: {patient_row[0]}, 
                  First Name: {patient_row[1]}, 
                  Last Name: {patient_row[2]}, 
                  Email: {patient_row[3]}, 
                  Phone: {patient_row[4]}''')


def display_doctors():
    cursor.execute('''SELECT * FROM DOCTORS''')
    patients = cursor.fetchall()
    if patients:
        for patient_row in patients:
            print(f'''SSN: {patient_row[0]}, 
                  First Name: {patient_row[1]}, 
                  Last Name: {patient_row[2]}, 
                  Email: {patient_row[3]}, 
                  Phone: {patient_row[4]}''')


def send_text_to_patient(doctor_ssn, patient_ssn, message_body):
    cursor.execute('''
        SELECT phone FROM PATIENTS WHERE ssn = ?
    ''', (patient_ssn,))
    patient_phone = f"+1{cursor.fetchone()[0]}"
    cursor.execute('''
        SELECT phone FROM DOCTORS WHERE ssn = ?
    ''', (doctor_ssn,))
    doctor_phone = f"+1{cursor.fetchone()[0]}"
    try: 
        message = client.messages.create(
            body=message_body, 
            from_="+18554648586",
            to="+15106736549"
        )
        print(f"Message sent! SID: {message.sid}")
    except Exception as e:
        print(f"Error sending your message: {str(e)}")




# create patients relation 
cursor.execute('''
    CREATE TABLE IF NOT EXISTS PATIENTS (
        ssn INT PRIMARY KEY,
        fname VARCHAR(50) NOT NULL,
        lname VARCHAR(50) NOT NULL,
        email VARCHAR(50) NOT NULL,
        phone INT NOT NULL        
    )
''')
# create doctor relation
cursor.execute('''
    CREATE TABLE IF NOT EXISTS DOCTORS (
        ssn INT PRIMARY KEY,
        fname VARCHAR(50) NOT NULL,
        lname VARCHAR(50) NOT NULL,
        email VARCHAR(50) NOT NULL,
        phone INT NOT NULL
    )
''')
# create relation describing which employees (doctors) manage which patients
cursor.execute('''
    CREATE TABLE IF NOT EXISTS MANAGES (
        doctor_ssn INT NOT NULL,
        patient_ssn INT NOT NULL,
        FOREIGN KEY(doctor_ssn) REFERENCES DOCTORS(ssn)
               ON DELETE CASCADE
               ON UPDATE CASCADE,
        FOREIGN KEY(patient_ssn) REFERENCES PATIENTS(ssn)
               ON DELETE CASCADE
               ON UPDATE CASCADE,
        PRIMARY KEY(doctor_ssn, patient_ssn)
    )
''')
# comit table creations
connection.commit()

add_patient("Joel", "Gurivireddy", 1111, "joelpreetam@gmail.com", 5106736549)
add_patient("Jane", "Doe", 1112, "janedoe@gmail.com", 1234567891)
add_patient("John", "Smith", 1113, "joelpreetam@gmail.com", 1234567892)
add_patient("Joel", "Gurivireddy", 1114, "joelpreetam@gmail.com", 1234567893)
add_patient("Joel", "Gurivireddy", 1115, "joelpreetam@gmail.com", 1234567894)
display_patients()


add_doctor("Joel", "Gurivireddy", 2221, "joelpreetam@gmail.com", 5106736549)
add_doctor("Jane", "Doe", 22212, "janedoe@gmail.com", 1234567891)
add_doctor("John", "Smith", 22213, "joelpreetam@gmail.com", 1234567892)
add_doctor("Joel", "Gurivireddy", 22214, "joelpreetam@gmail.com", 1234567893)
add_doctor("Joel", "Gurivireddy", 22215, "joelpreetam@gmail.com", 1234567894)
display_doctors()

send_text_to_patient(2221, 1111, "HELLO! This is a twilio test message")


connection.close()