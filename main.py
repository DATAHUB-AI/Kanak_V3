

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
import logging

# Configure logging
logging.basicConfig(filename='app_errors.log', level=logging.ERROR,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Page configuration
st.set_page_config(page_title="Salary Prediction App", layout="wide")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Step 1: Functions for Model and Data
@st.cache_data
def load_data(file_path):
    """Loads dataset from the given CSV file path."""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        st.error("An error occurred while loading the data. Please try again later.")
        return pd.DataFrame()

@st.cache_resource
def load_trained_model(model_path):
    """Loads the trained model or preprocessing objects from a file."""
    try:
        return joblib.load(model_path)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        st.error("An error occurred while loading the model. Please try again later.")
        return None

# Load data and model
df = load_data("application.csv")

# Ensure necessary columns are present
required_columns = [
    "Candidate Age", "Candidate Gender", "Candidate City", "Disability Status",
    "Marriage Status", "Highest Qualification", "Present Village", "Present District",
    "Present State", "Present Country", "Employment Type", "Job Title",
    "Job Designation", "Job Location", "Course Name", "Course Sector", "CTC (Yearly)"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    logging.error(f"Missing columns in dataset: {', '.join(missing_columns)}")
    st.error("An error occurred while processing the data. Please try again later.")

# Load trained model and encoders
model = load_trained_model("salary_prediction_model_feb9.pkl")
label_encoders = load_trained_model("label_encoders.pkl")
scaler = load_trained_model("scaler.pkl")

if not model or not label_encoders or not scaler:
    st.stop()

def encode_inputs(input_data):
    """Encodes input data using label encoders and scaler."""
    encoded_data = []
    for key, value in input_data.items():
        if key in label_encoders:
            if value in label_encoders[key].classes_:
                encoded_data.append(label_encoders[key].transform([value])[0])
            else:
                logging.warning(f"Value '{value}' for {key} not found in training data. Defaulting to unknown (-1).")
                encoded_data.append(-1)
        elif key == "Candidate Age":
            encoded_data.append(scaler.transform([[value]])[0][0])
        else:
            encoded_data.append(value)
    logging.info(f"Encoded Data: {encoded_data}")  # Log encoded data for debugging
    return np.array(encoded_data)

# Function to calculate salary range
def calculate_salary_range(predicted_salary, percentage_deviation=0.1):
    lower_bound = predicted_salary * (1 - percentage_deviation)
    upper_bound = predicted_salary * (1 + percentage_deviation)
    return lower_bound, upper_bound

# Course sector and course name data
course_data = {
    "Apparel": [
        'Certificate Course for Sewing Machine Operator - 2',
        'Certificate Course on Aari and Embroidery Work-I',
        'Certification Course for Self Employed Tailor -1- I',
        'Self Employed Tailor'
    ],
    "Automotive": [
        'Certificate Course for CNC Operator',
        'Auto Sales Consultant',
        'Auto Service Technician (2-wheelers)',
        'CNC Operator',
        'Certificate Course for Automotive Assembly Operator - 4 Wheelers',
        'Certificate Course for Automotive Service Technician - 4 Wheeler',
        'Electric Vehicle Service Technician',
        'Foundation Skills for Automotive Manufacturing Industry',
        'Kia HMV Drivers Training',
        'Kia HMV Drivers Training_APSRTC',
        'SLMT - Automotive Service Technician (Upskilling)- 2, 3 and 4 wheeler'
    ],
    "BFSI": [
        'Certificate Course for Life Insurance Agent',
        'Certificate Course for Loan Approval Officer',
        'Certification Course for Business Correspondent & Business Facilitator',
        'Certification Course for Equity Dealer'
    ],
    "Beauty & Wellness": [
        'Assistant Hair Stylist',
        'Beauty Entrepreneurship Program',
        'Certificate Course for Assistant Beauty Therapist',
        'Certificate Course for Assistant Beauty Therapist (Alankrit)',
        'Certificate Course for Assistant Beauty Therapist - Alankrit- I',
        'Certificate Course for Assistant Beauty Therapist - Saundarya- I',
        'Loreal Hair stylist',
        'Mehendi and Bridal MakeUp'
    ],
    "Capital Goods": [
        'Certificate Course for Fitter Electrical and Electronic Assembly',
        'Certificate Course for Fitter Fabrication - 2 ',
        'Certificate Course for Welding Operator',
        'Fitter Electrical and Electronic Assembly (NSDC)',
        'Fitter Fabrication ',
        'Welding Operator'
    ],
    "Construction": [
        'Certificate Course for Assistant Construction Painter & Decorator',
        'Assistant Electrician',
        'Certificate Course for Assistant Construction Painter & Decorator',
        'Certificate Course for Assistant Shuttering Carpenter',
        'Certificate Course on Shuttering Carpenter System',
        'Certification Course for Assistant Electrician-1',
        'Energy Efficient Electrician',
        'RPL Certification for Construction Painter & Decorator'
    ],
    "Electronics": [
        'Mobile Phone Hardware Repair Technician'
    ],
    "Furniture & Fittings Skill Council": [
        'Certificate Course for Assistant Carpenter – Wooden Furniture'
    ],
    "Healthcare": [
        'Certification Course for General Duty Assistant -1',
        'Certification Course on Refractionist for Essilor- Eye Mitra'
    ],
    "IT-ITeS": [
        'Associate Desktop Publishing ',
        'CRM Domestic Non Voice',
        'CRM Domestic Voice ',
        'Certificate Course for Data Entry Operator - 2- 1',
        'Certificate Course in Tally - 2- I',
        'Certificate Course on Associate Desktop Publishing',
        'Certificate Course on CRM Domestic Voice - 1-1',
        'Certification Course on CRM Domestic Non Voice ',
        'Certification Course on CRM Domestic Non Voice - 2- I',
        'Data Analytics',
        'Data Entry Operator'
    ],
    "Logistics": [
        'Certificate Course for Courier Delivery Executive - 2- I',
        'Warehouse Picker '
    ],
    "Media & Entertainment Skill Council": [
        'Certificate Course on Digital Marketing - 2- I',
        'Digital Marketing '
    ],
    "Others": [
        'Certificate Course for Data Entry Operator - 2',
        'Cyber Security',
        'Cybersecurity',
        'Saksham 21st Century Skills ',
        'Work Enhancement Skills',
        'Work Place Skills',
        'Workshop on Employability Skills and Career Counselling for College Graduates'
    ],
    "Plumbing": [
        'RPL Certification for Plumber General'
    ],
    "Retail": [
        'Certification Course for Retail Sales Associate-2_RSA-2',
        'Certificate Course For Retail Sales Associate 1',
        'Certification Course for Retail Sales Associate-1_ RSA 1',
        'Certification Course for Retail Sales Associate-1_RSA-1',
        'Certification Course for Retail Sales Associate-2_RSA-2'
    ],
    "Tourism & Hospitality": [
        'Certificate Course for Front Office Associate - 2',
        'Food & Beverage Service Steward',
        'Front Office associate'
    ]
}

# Sidebar Navigation
with st.sidebar:
    st.title("Prediction Apps")
    page = option_menu(
        menu_title=None,
        options=["Home", "Student Salary Prediction", "Counselor Page"],
        icons=["house", "bar-chart-line", "graph-up-arrow", "person-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Home Page
if page == "Home":
    st.header("Employee Salary Prediction")
    st.write("### Sample Data")
    if not df.empty:
        st.dataframe(df.sample(frac=0.3, random_state=42).reset_index(drop=True), use_container_width=True)

# Relations & Correlations Page deleted 

# Student Salary Prediction Page
if page == "Student Salary Prediction":
    st.header("Candidate Salary Prediction")
    with st.form("Predict Form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Candidate Age", min_value=18, max_value=80, value=25)
            candidate_name = st.text_input("Candidate Name")
            gender = st.selectbox("Candidate Gender", options=["Male", "Female", "Transgender"])
            disability = st.selectbox("Disability Status", options=["No", "Yes"])
            marriage_status = st.selectbox("Marriage Status", options=["Unmarried", "Married", "Separated", "Widowed"])
            qualification = st.selectbox(
                "Highest Qualification",
                options=[
                    "Higher Secondary Pass", "10th Pass", "Graduation", "6 to 9", "ITI pass",
                    "Diploma Pass", "Post Graduation", "1 to 5", "Certification"
                ]
            )
            employment_type = st.selectbox(
                "Employment Type",
                options=[
                    "Salaried", "Wage Employment- Full Time", "Self Employed", "Part Time",
                    "Professional", "Nano Contractor", "Gig", "Unpaid worker in household enterprise"
                ]
            )
            job_title = st.text_input("Job Title")
            job_designation = st.text_input("Job Designation")
            job_location = st.text_input("Job Location")

        with col2:
            city = st.text_input("Candidate City")
            village = st.text_input("Present Village")
            district = st.text_input("Present District")
            state = st.text_input("Present State")
            country = st.text_input("Present Country")
            course_sector = st.selectbox("Course Sector", options=list(course_data.keys()))
            course_name = st.selectbox("Course Name", options=course_data[course_sector])

        # Submit button
        submitted = st.form_submit_button("Predict Salary")
        if submitted:
            try:
                input_data = {
                    "Candidate Age": age,
                    "Candidate Gender": gender,
                    "Candidate City": city,
                    "Disability Status": disability,
                    "Marriage Status": marriage_status,
                    "Highest Qualification": qualification,
                    "Present Village": village,
                    "Present District": district,
                    "Present State": state,
                    "Present Country": country,
                    "Employment Type": employment_type,
                    "Job Title": job_title,
                    "Job Designation": job_designation,
                    "Job Location": job_location,
                    "Course Name": course_name,
                    "Course Sector": course_sector
                }
                encoded_data = encode_inputs(input_data)
                salary_prediction = model.predict([encoded_data])[0]

                # Calculate salary range (10% deviation)
                lower_salary, upper_salary = calculate_salary_range(salary_prediction, 0.1)

                st.success("Prediction Successful!")
                st.write(f"### Predicted Salary Range: ₹{lower_salary:,.2f} to ₹{upper_salary:,.2f}")
            except Exception as e:
                logging.error(f"Prediction Error: {e}")
                st.error("An error occurred while making the prediction. Please try again later.")

# Counselor Page
# Counselor Page
if page == "Counselor Page":
    st.header("Career Counseling Page")

    # Mapping of qualifications to courses, sectors, and salary ranges
    qualification_course_mapping = {
        "10th Pass": [
            ("Self Employed Tailor", "Apparel", (8000, 20000)),
            ("Assistant Hair Stylist", "Beauty & Wellness", (10000, 25000)),
            ("Certificate Course for Assistant Beauty Therapist", "Beauty & Wellness", (12000, 30000)),
            ("Certificate Course for Assistant Shuttering Carpenter", "Construction", (15000, 35000)),
            ("Certificate Course for Assistant Construction Painter & Decorator", "Construction", (18000, 40000)),
            ("Assistant Electrician", "Construction", (20000, 45000)),
            ("Energy Efficient Electrician", "Construction", (22000, 50000)),
            ("Auto Service Technician (2-wheelers)", "Automotive", (25000, 60000)),
            ("Certificate Course for Welding Operator", "Capital Goods", (30000, 70000))
        ],
        "12th Pass": [
            ("Certificate Course For Retail Sales Associate", "Retail", (10000, 30000)),
            ("Certificate Course for Front Office Associate", "Tourism & Hospitality", (12000, 35000)),
            ("CRM Domestic Voice", "IT-ITeS", (15000, 40000)),
            ("CRM Domestic Non-Voice", "IT-ITeS", (18000, 45000)),
            ("Food & Beverage Service Steward", "Tourism & Hospitality", (20000, 50000)),
            ("Front Office Associate", "Tourism & Hospitality", (25000, 60000)),
            ("Auto Sales Consultant", "Automotive", (30000, 70000))
        ],
        "Diploma": [
            ("Certificate Course in Tally", "IT-ITeS", (20000, 50000)),
            ("Certificate Course for Data Entry Operator", "IT-ITeS", (25000, 60000)),
            ("Data Entry Operator", "IT-ITeS", (30000, 70000)),
            ("Certificate Course for Fitter Electrical and Electronic Assembly", "Capital Goods", (35000, 80000)),
            ("Mobile Phone Hardware Repair Technician", "Electronics", (40000, 90000)),
            ("Certificate Course for CNC Operator", "Automotive", (45000, 100000)),
            ("Certificate Course for Automotive Assembly Operator - 4 Wheelers", "Automotive", (50000, 120000))
        ],
        "Certification": [
            ("Certificate Course for Automotive Service Technician - 4 Wheeler", "Automotive", (40000, 100000)),
            ("Certificate Course for Loan Approval Officer", "BFSI", (50000, 120000)),
            ("Certificate Course for Life Insurance Agent", "BFSI", (60000, 140000)),
            ("Certification Course for Equity Dealer", "BFSI", (70000, 160000)),
            ("Certification Course for Business Correspondent & Business Facilitator", "BFSI", (80000, 180000)),
            ("Certificate Course for Assistant Carpenter – Wooden Furniture", "Furniture & Fittings Skill Council", (90000, 200000))
        ],
        "Graduate": [
            ("Data Analytics", "IT-ITeS", (60000, 160000)),
            ("Cyber Security", "IT-ITeS", (70000, 180000)),
            ("Associate Desktop Publishing", "IT-ITeS", (80000, 200000)),
            ("Electric Vehicle Service Technician", "Automotive", (100000, 240000))
        ]
    }

    with st.form("Counseling Form"):
        candidate_name = st.text_input("Candidate Name")
        age = st.number_input("Age", min_value=18, max_value=80, value=22)
        gender = st.selectbox("Gender", options=["Male", "Female", "Transgender"])
        qualification = st.selectbox("Highest Qualification", options=["10th Pass", "12th Pass", "Diploma", "Certification", "Graduate"])
        
        # Display min and max salary for the selected qualification
        if qualification in qualification_course_mapping:
            min_salary = min([course[2][0] for course in qualification_course_mapping[qualification]])
            max_salary = max([course[2][1] for course in qualification_course_mapping[qualification]])
            st.write(f"**Salary Range for {qualification}:** ₹{min_salary:,} - ₹{max_salary:,}")

        field_of_interest = st.text_input("Field of Interest")
        job_preference = st.selectbox("Job Preference", options=["Full-time", "Part-time", "Self-employed", "Freelancer"])
        expected_salary = st.number_input("Expected Salary (in INR)", min_value=5000, max_value=1000000, step=1000, value=10000)
        willing_to_relocate = st.radio("Willing to Relocate?", options=["Yes", "No"])
        additional_skills = st.text_area("Additional Skills")

        submitted = st.form_submit_button("Get Counseling Advice")

        if submitted:
            # Get suggested courses based on qualification and expected salary
            suggested_courses = []
            for course in qualification_course_mapping.get(qualification, []):
                course_name, course_sector, salary_range = course
                if salary_range[0] <= expected_salary <= salary_range[1]:
                    suggested_courses.append({"Course Name": course_name, "Course Sector": course_sector})  # Removed Salary Range

            if suggested_courses:
                st.success("Here are some suggested courses:")
                suggested_courses_df = pd.DataFrame(suggested_courses)
                st.table(suggested_courses_df)
            else:
                st.warning("No courses match your expected salary. Please adjust your expected salary or qualification.")

            counseling_advice = "We recommend exploring roles aligned with your interests and upskilling in areas where demand is high. "
            if willing_to_relocate == "Yes":
                counseling_advice += "Relocation can significantly increase job opportunities."
            else:
                counseling_advice += "Consider remote job options if relocation is not feasible."

            st.info(
                f"""
                Candidate Name: {candidate_name}
                Age: {age} years
                Field of Interest: {field_of_interest}
                Expected Salary: ₹{expected_salary:,.2f}
                Relocation Option: {willing_to_relocate}

                **Counseling Advice:**
                {counseling_advice}
                """
            )
